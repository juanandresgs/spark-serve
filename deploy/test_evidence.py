"""Schema, provenance, immutability and real runner HTTP-boundary checks."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from jsonschema import ValidationError
from spark_serve import evidence as e
from spark_serve.evidence_bench import metadata, publish_group

ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable

class EvidenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.records=e.load(ROOT)

    def sample(self):return copy.deepcopy(self.records['import-20260930-qwen-0-1-v1'])

    def test_schema_rejects_units_unknown_fields_and_broken_fingerprint(self):
        r=self.sample();r['measurements'][0]['unit']='s'
        with self.assertRaisesRegex(ValueError,'unit'):e.validate_record(e.seal(r),ROOT)
        r=self.sample();r['secret']='not allowed'
        with self.assertRaises(ValidationError):e.validate_record(e.seal(r),ROOT)
        r=self.sample();r['measurements'][0]['value']=123
        with self.assertRaisesRegex(ValueError,'fingerprint'):e.validate_record(r,ROOT)
        with self.assertRaisesRegex(ValueError,'source'):e.validate_record(e.seal(r),ROOT)

    def test_receipt_snapshot_survives_original_updates_and_rejects_path_escape(self):
        r=self.sample();source=r['measurements'][0]['source']
        self.assertTrue(source['file'].startswith('evidence/sources/'))
        self.assertNotEqual(source['file'],source['original_file'])
        self.assertEqual(e.resolve(ROOT,source),r['measurements'][0]['value'])
        with self.assertRaisesRegex(ValueError,'Unsafe'):e.safe_file(ROOT,'../outside.json')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'duplicate.json';p.write_text('{"id":1,"id":2}')
            with self.assertRaisesRegex(ValueError,'Duplicate'):e.read(p)

    def test_external_reports_cannot_qualify_local_recipe(self):
        r=copy.deepcopy(self.records['import-20260930-public-0-reported-v1'])
        r['qualification']['reboot']['status']='passed'
        with self.assertRaisesRegex(ValueError,'cannot qualify'):e.validate_record(e.seal(r),ROOT)

    def test_actual_mismatch_missing_refs_and_recommendation_hardware(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['evidence','recipes','comparisons','experiments','cluster']:
                shutil.copytree(ROOT/name,root/name)
            path=root/'evidence/recommendations/qwen-recommendation-20260930-v1.json'
            r=e.read(path);r['hardware_nodes']=2;path.write_text(json.dumps(e.seal(r)))
            with self.assertRaisesRegex(ValueError,'hardware'):e.load(root)
            r['hardware_nodes']=1;r['evidence_runs'][0]['fingerprint']='0'*64;path.write_text(json.dumps(e.seal(r)))
            with self.assertRaisesRegex(ValueError,'reference'):e.load(root)

    def test_comparability_is_fail_closed(self):
        r=self.sample();m=r['measurements'][0]
        with self.assertRaisesRegex(ValueError,'Incomparable'):e.percent_change(r,m,r,m)
        r['origin']='local';r['identity']['relationship']='attested_exact';r['environment']['unknowns']=[]
        r['identity'].update(image_digest='sha256:'+'a'*64,runtime_revision='a'*40,model_revision='b'*40)
        m['conditions']['cache']='cold';m['conditions']['fixture_sha256']='a'*64
        b=copy.deepcopy(r);b['measurements'][0]['value']=m['value']*1.2
        self.assertAlmostEqual(e.percent_change(r,m,b,b['measurements'][0]),20)
        for field,value in [('concurrency',1),('cache','warm'),('sampling',{'temperature':1}),('fixture_sha256','b'*64)]:
            changed=copy.deepcopy(b);changed['measurements'][0]['conditions'][field]=value
            with self.assertRaisesRegex(ValueError,'Incomparable'):e.percent_change(r,m,changed,changed['measurements'][0])
        changed=copy.deepcopy(b);changed['measurements'][0]['metric']='decode_rate_proxy'
        with self.assertRaisesRegex(ValueError,'Incomparable'):e.percent_change(r,m,changed,changed['measurements'][0])
        b['origin']='external_report'
        with self.assertRaisesRegex(ValueError,'external'):e.percent_change(r,m,b,b['measurements'][0])

    def test_importer_preserves_existing_records_and_refuses_changed_configuration(self):
        spec=importlib.util.spec_from_file_location('import_evidence',ROOT/'deploy/import_evidence.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['evidence','recipes','comparisons','experiments','cluster']:
                shutil.copytree(ROOT/name,root/name)
            module.ROOT=root
            # Recreate the migration's original published inputs from retained snapshots.
            for r in self.records.values():
                for source in r.get('sources',[]):
                    if 'original_file' in source:
                        shutil.copyfile(root/source['file'],root/source['original_file'])
            record=root/'evidence/runs/import-20260930-qwen-0-1-v1.json'
            before=record.stat().st_mtime_ns
            config=root/'comparisons/models.json';data=e.read(config);data['presentation_note']='owned by another editor'
            body=json.dumps(data);config.write_text(body)
            with self.assertRaisesRegex(ValueError,'configuration changed'):module.main()
            self.assertEqual(config.read_text(),body)
            self.assertEqual(record.stat().st_mtime_ns,before)

    def test_create_only_and_history_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=e.write_immutable(self.sample(),d,ROOT)
            before=p.read_bytes()
            with self.assertRaises(FileExistsError):e.write_immutable(self.sample(),d,ROOT)
            self.assertEqual(p.read_bytes(),before)
            root=Path(d);(root/'evidence/runs').mkdir(parents=True)
            target=root/'evidence/runs/a.json';target.write_text('{}')
            def git(*args):return subprocess.run(['git','-C',d,*args],check=True,capture_output=True)
            git('init');git('add','.');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','base')
            e.check_immutable(root,'HEAD');target.write_text('{"edited":true}')
            with self.assertRaisesRegex(ValueError,'append-only'):e.check_immutable(root,'HEAD')

    def test_http_runner_emits_sanitized_records_and_missing_usage(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                text='{"sum":465,"sorted":[1,3,7,9],"marker":"TF-CHECK-20260927"}'
                if 'context fixture' in body['messages'][0]['content']:
                    text=json.dumps({'ALPHA':'amber-7419-lake','BETA':'silver-3821-oak','OMEGA':'violet-2863-moon'})
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                event={'choices':[{'delta':{'content':text},'finish_reason':'stop'}], 'usage':{'completion_tokens':16,'prompt_tokens':10,'prompt_tokens_details':{'cached_tokens':10 if len(body['messages'])>1 else 0}}}
                self.wfile.write(('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode())
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                path=Path(d);recipe=self.records['qwen-affine4-source-20260930-v1'];pins=recipe['pins']
                meta={'recipe_id':recipe['id'],'image_digest':'sha256:'+'1'*64,'runtime_revision':pins['runtime_revision'],'model_revision':pins['target_revision'],'hardware':recipe['hardware'],'software':{'os':'test','kernel':'test','driver':'test','cuda':'test','runtime_version':'test'}}
                (path/'metadata.json').write_text(json.dumps(meta))
                invalid=copy.deepcopy(meta);invalid['recipe_id']='glm-adaptive-source-20260930-v1';invalid['hardware']=self.records[invalid['recipe_id']]['hardware']
                (path/'wrong-model.json').write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError,'Qwen Affine4'):metadata(path/'wrong-model.json',ROOT)
                result=subprocess.run([PYTHON,str(ROOT/'recipes/qwen38-flash-affine4-1spark/bench.py'),'--base',f'http://127.0.0.1:{server.server_port}/v1','--label','test','--out',str(path/'raw'),'--quick','--clients','4','--reps','1','--evidence-metadata',str(path/'metadata.json'),'--evidence-dir',str(path/'share')],env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                records=[e.validate_record(e.read(p),ROOT) for p in (path/'share').glob('*.json')]
                self.assertEqual(len(records),2)
                self.assertNotIn('127.0.0.1',json.dumps(records))
                self.assertNotIn('TF-CHECK',json.dumps(records))
                for r in records:
                    self.assertEqual(r['measurements'][0]['samples'],4)
                    self.assertIsNone(r['measurements'][3]['value']) # one delta cannot yield a decode window
                    self.assertEqual(r['qualification']['correctness']['status'],'not_tested')
                    serialized=json.dumps(r)
                    for forbidden in ['127.0.0.1','TF-CHECK','messages','output','reasoning_content']:
                        self.assertNotIn('\"'+forbidden+'\"',serialized)
                (path/'raw/fixtures.json').write_text(json.dumps([{'target':128,'messages':[{'role':'user','content':'context fixture'}]}]))
                context=subprocess.run([PYTHON,str(ROOT/'recipes/qwen38-flash-affine4-1spark/bench.py'),'--base',f'http://127.0.0.1:{server.server_port}/v1','--label','context','--suite','context','--out',str(path/'raw'),'--reps','1','--evidence-metadata',str(path/'metadata.json'),'--evidence-dir',str(path/'context-share')],env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True)
                self.assertEqual(context.returncode,0,context.stderr)
                context_records=[e.validate_record(e.read(p),ROOT) for p in (path/'context-share').glob('*.json')]
                self.assertEqual(len(context_records),2)
                self.assertEqual({r['measurements'][0]['conditions']['cache'] for r in context_records},{'cold','warm'})
                for r in context_records:
                    self.assertEqual(r['accounting']['requests_attempted'],1)
                    self.assertEqual(r['qualification']['correctness']['status'],'passed')
                    self.assertEqual(r['measurements'][-2]['value'],1)
                    self.assertNotIn('amber-7419',json.dumps(r))
                raw=e.read(path/'raw/test-speed.json');group=raw['groups'][0]
                group['rows'][0]['completion_tokens']=None;group['rows'][0]['error']='SECRET endpoint and output'
                p=publish_group(ROOT,path/'share',meta,recipe,group,[ROOT/'recipes/qwen38-flash-affine4-1spark/bench.py'],'2026-09-30')
                failed=e.read(p);self.assertIsNone(failed['measurements'][0]['value']);self.assertEqual(failed['qualification']['performance']['status'],'failed');self.assertNotIn('SECRET',p.read_text())
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
