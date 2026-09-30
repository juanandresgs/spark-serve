#!/usr/bin/env python3
"""One-time, deterministic migration of published receipts; never executes a benchmark.

Existing records are immutable. A repeated import verifies byte identity; corrections
require a new migration/version. Legacy receipts remain the provenance authority.
"""
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from spark_serve.evidence import canonical, read, seal, write_bytes_once, write_immutable
ROOT = Path(__file__).resolve().parents[1]
STAMP = '2026-09-30'

def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def source(file,path):
 sha=digest(file);target=ROOT/'evidence/sources'/(sha+'.json');target.parent.mkdir(exist_ok=True)
 body=(ROOT/file).read_bytes()
 if target.exists() and target.read_bytes()!=body:raise ValueError('Content-addressed source collision')
 if not target.exists():
  try:write_bytes_once(target,body)
  except FileExistsError:
   if target.read_bytes()!=body:raise ValueError('Conflicting source snapshot')
 return {'file':str(target.relative_to(ROOT)),'original_file':file,'path':path,'sha256':sha}
def ref(r):return {'id':r['id'],'fingerprint':r['fingerprint']}
def save(r):
 r=seal(r);p=ROOT/'evidence'/(r['kind']+'s')/(r['id']+'.json');body=json.dumps(r,indent=2)+'\n'
 if p.exists() and p.read_text()!=body:raise ValueError('Refusing to rewrite immutable '+str(p))
 if not p.exists():write_immutable(r,p.parent,ROOT)
 return r

def recipe(id,catalog,model,nodes,pins,settings,files,unknowns,scope):
 return save(dict(schema_version=1,kind='recipe',id=id,catalog_id=catalog,model=model,
 hardware={'accelerator':'NVIDIA GB10','nodes':nodes,'memory_gb_per_node':128,'topology':'single GPU' if nodes==1 else 'tensor parallel pair'},
 pins=pins,settings=settings,requirements={'destination_acceptance_required':True,'available_memory_gib_per_node':100,'disk_free':{'value':180 if model=='qwen' else 190,'unit':'GB' if model=='qwen' else 'GiB'},'runtime':'NVIDIA-enabled Docker; serving GPU free of other workloads'},
 source_files={f:digest(f) for f in files},unknowns=unknowns,scope=scope))

def qualifications(scope):return {k:{'status':'unknown','scope':scope} for k in ['build','api_tool','correctness','clean_install','restart','reboot','performance','endurance']}

def main():
 models=read(ROOT/'comparisons/models.json')
 # Retain the exact pre-migration table specification as an auditable input.
 old=ROOT/'evidence/legacy-comparisons-v1.json'
 if old.exists():models=read(old)
 else:write_bytes_once(old,(json.dumps(models,indent=2)+'\n').encode())
 q='recipes/qwen38-flash-affine4-1spark/'
 e='experiments/qwen-tensorfold-cooperative/'
 g='recipes/glm53-flash-adaptive-2spark/'
 affine=recipe('qwen-affine4-source-20260930-v1','qwen38-flash-affine4-1spark','qwen',1,read(ROOT/(q+'pins.json')),read(ROOT/(q+'presets.json')),
 [q+x for x in ['pins.json','Dockerfile','run.sh','serve.py','cooperative.py','reliability.py','tool_adapter.py','selection.py','ssd_pipeline.py','reader.cpp','stage.py']],
 ['Source snapshot is not proof of the actual identity of every historical run.'],
 'Published source kit. Historical runs link by association unless actual identity was captured.')
 exl=recipe('qwen-exl3-source-20260930-v1','qwen-cooperative-exl3','qwen',1,read(ROOT/(e+'pins.json')),{'context_tokens':262144,'slots':4},[e+'pins.json'],
 ['Complete image and adapter identity of historical control not retained in these receipts.'],'Historical EXL3 alternative; not a new build qualification.')
 gp=read(ROOT/(g+'recipe.json'))
 glm=recipe('glm-adaptive-source-20260930-v1','glm53-flash-adaptive-2spark','glm',2,gp['profile_pins'],read(ROOT/(g+'profile-0.json')),
 [g+'profile-0.json','cluster/glm-throughput/Dockerfile','cluster/glm-throughput/overlay/patch_adaptive_k.py'],
 ['Historical installed image digest, rank software parity and benchmark fixture identity not retained.'],'Portable source snapshot; original-site evidence does not qualify a destination install.')
 fixed=recipe('glm-fixed-historical-20260911-v1','glm53-flash-2spark','glm',2,{k:{p:v for p,v in pin.items() if p!='image'} for k,pin in gp['profile_pins'].items()},{'draft_policy':'fixed k=7, full graph','dense_precision':'BF16'},[],
 ['Exact image digest and full runtime settings of matched fixed-k control not retained.'],'Reconstructed historical control, not a current source-build identity.')
 publicq=recipe('external-qwen-report-20260930-v1','external-qwen','qwen',1,{}, {},[],['Reported configuration is incomplete.'],'External report only.')
 publicg=recipe('external-glm-report-20260930-v1','external-glm','glm',2,{}, {'dense_precision':'FP8'},[],['Reported configuration is incomplete.'],'External report only.')
 runs=[]; mappings={}
 def create_run(key,spec,table,row_index,arm,rec,external=False):
  id=f'import-20260930-{key}-{row_index}-{arm}-v1'
  metrics=[]
  for i,r in enumerate(spec['refs']):
   v=read(ROOT/r['file'])
   for part in r['path']:v=v[part]
   leaf=str(r['path'][-1]); label=table['rows'][row_index]['metric']
   metric='reported_throughput';unit='tokens/s';agg='reported_unknown';samples=None;sample_unit='unknown';definition='Output tokens per second as reported; exact timing denominator not retained.'
   cache='unknown';c=None;request={};sampling={};warmup=None
   if key=='qwen':
    if 'throughput' in label:
     metric='end_to_end_throughput';agg='median';samples=3;sample_unit='request groups';c=4
     definition='Median across repetitions of total completion tokens divided by whole request-group wall seconds, including prefill.'
     request={'max_tokens':512,'thinking':False};sampling={'temperature':0,'seed_policy':'1000 + fixture number'};warmup='One excluded exact-response request';cache='uncontrolled'
    elif 'correct' in label:
     metric='correct' if leaf=='correct' else 'cases';unit='count';agg='count';samples=296 if 'reasoning' in label else 20;sample_unit='distinct questions' if samples==296 else 'coding tasks'
     definition='Count of correct answers' if metric=='correct' else 'Count of evaluated distinct cases'
     c=4;request={'max_tokens':8192,'thinking':True,'reasoning_cap':None};sampling={'temperature':1.0,'top_p':0.95,'top_k':20}
    else:
     metric='response_latency';unit='s';agg='p95' if 'p95' in leaf else ('median' if 'median' in leaf else 'single');sample_unit='requests'
     definition='Client elapsed seconds from request start until the complete response, including validation/buffering where enabled.'
     if 'Cold' in label or 'Cached' in label:
      cache='cold' if 'Cold' in label else 'warm';samples=1;c=1;request={'max_tokens':8192,'prompt_tokens':253843 if cache=='cold' else None};sampling={'temperature':0}
     elif 'mixed' in label:
      cache='mixed';samples=20;c=4
     else:
      samples=196 if 'reasoning' in label else 20;c=4;request={'max_tokens':8192,'thinking':True,'reasoning_cap':None};sampling={'temperature':1.0,'top_p':0.95,'top_k':20}
   elif key=='glm':c=8 if 'Eight' in label else 1
   elif external:
    c=4 if row_index==0 else (1 if row_index in [1,2] else None)
    if row_index==1:
     metric='ttft' if i==0 else 'reported_prompt_tokens';unit='s' if i==0 else 'tokens';definition='Externally reported time to first token.' if i==0 else 'Externally reported prompt tokens.'
    if row_index==3 and i in [1,3]:metric='cases';unit='count';definition='Externally reported concurrency (not a sample count).'
   conditions={'workload':key+'-'+str(row_index),'fixture_sha256':None,'concurrency':c,'cache':cache,'request':request,'sampling':sampling,'warmup':warmup}
   metrics.append(dict(id='m'+str(i),metric=metric,value=v,unit=unit,definition=definition,aggregation=agg,samples=samples,sample_unit=sample_unit,population='all reported observations; raw failure exclusion policy unknown' if key!='qwen' else 'published cohort; see original receipt',conditions=conditions,source=source(r['file'],r['path'])))
  srcs=list({(r['file'],digest(r['file'])):source(r['file'],[]) for r in spec['refs']}.values())
  quals=qualifications('Not established by this imported measurement cohort. See separate qualification receipts.')
  if key=='qwen' and not external:
   quals['correctness' if 'correct' in table['rows'][row_index]['metric'] else 'performance']={'status':'passed','scope':'Historical bounded workload only; does not qualify this published kit on a fresh machine.'}
  failures=[]
  if key=='qwen' and row_index==5 and arm=='0':failures=[{'category':'wrong_answer','count':2,'detail':'Two failures among 296 distinct reasoning questions; duplicate cases excluded as recorded in quality.json.'}]
  if failures:quals['correctness']={'status':'failed','scope':'Bounded reasoning evaluation includes wrong answers; see failure count.'}
  r=dict(accounting={k:None for k in ['requests_attempted','requests_completed','prompt_tokens','completion_tokens','cached_tokens','wall_seconds']},observations=[],schema_version=1,kind='run',id=id,origin='external_report' if external else 'historical_import',recorded_at=STAMP,executed_at=None if external else table.get('date'),recipe=ref(rec),
   identity={'relationship':'external_report' if external else 'historical_association','image_digest':None,'runtime_revision':None,'model_revision':None,'attestation':'Imported from pinned public receipts; no new execution or actual-identity attestation.'},
   environment={'hardware':rec['hardware'],'software':{},'unknowns':['Exact installed software, runtime/image identity and physical host parity not established by the numeric receipt.']},
   workload_version='historical-'+key+'-v1',measurements=metrics,failures=failures,qualification=quals,sources=srcs,
   external_sources=[],
   limitations=['Imported aggregate/cohort, not a newly executed run. Unknown metadata is deliberately not reconstructed from current defaults.','No percentage comparison permitted until required matching metadata is available.'])
  if external:
   u=read(ROOT/'comparisons/public-references.json')['sources'][0 if row_index<2 else 1];r['external_sources']=[{'url':u,'revision':u.split('/blob/')[1].split('/')[0]}]
  r=save(r);runs.append(r)
  for i,oldref in enumerate(spec['refs']):mappings[json.dumps(oldref,sort_keys=True)]={'file':f'evidence/runs/{id}.json','path':['measurements',i,'value']}
 for key in ['qwen','glm']:
  table=models['tables'][key]
  for i,row in enumerate(table['rows']):
   for a,spec in enumerate(row['cells']):create_run(key,spec,table,i,str(a),[exl,affine][a] if key=='qwen' else [fixed,glm][a])
 table=models['tables']['public']
 for i,row in enumerate(table['rows']):create_run('public',row['public'],table,i,'reported',publicq if i<2 else publicg,True)
 # Preserve historical failed mitigation separately, not hidden behind selected rows.
 file=q+'results.json';failed=read(ROOT/file)['reliability-affine-bounded-greedy']
 r=copy.deepcopy(runs[0]);r.update(id='import-20260930-qwen-bounded-greedy-failures-v1',recipe=ref(affine),measurements=[],sources=[source(file,['reliability-affine-bounded-greedy'])],failures=[{'category':'wrong_answer','count':2,'detail':'2048-token cap with greedy sampling answered 4/6 known failures correctly; not the recommended sampling policy.'}]);r['qualification']=qualifications('Not tested by this imported cohort.');r['qualification']['correctness']={'status':'failed','scope':'Greedy bounded replay: 4/6 correct.'};runs.append(save(r))
 file=q+'build-validation.json';build=read(ROOT/file)
 r=copy.deepcopy(runs[0]);r.update(id='import-20260930-qwen-public-build-v1',recipe=ref(affine),executed_at='2026-09-30',measurements=[],sources=[source(file,[])],failures=[])
 r['identity'].update(image_digest=build['image'],attestation='Independent public dependency build receipt; existing checksum-verified weights reused. Other identity metadata not inferred.')
 r['qualification']=qualifications('Not tested by the independent public build receipt.')
 r['qualification']['build']={'status':'passed','scope':'Fresh dependency build and kernel compilation with existing verified checkpoint.'}
 r['qualification']['api_tool']={'status':'passed','scope':'11 HTTP and 36 typed API checks; six exact sampled replays.'}
 for k in ['clean_install','restart','reboot','performance','endurance']:r['qualification'][k]={'status':'not_tested','scope':'Not repeated by this bounded build qualification.'}
 r['limitations']=['Existing weights reused. No clean-machine download, full performance repeat, reboot or endurance qualification.']
 runs.append(save(r))
 for model,rec in [('qwen',affine),('glm',glm)]:
  m=next(m for m in models['models'] if m['id']==model)
  recommendation=save(dict(schema_version=1,kind='recommendation',id=model+'-recommendation-20260930-v1',model=model,hardware_nodes=m['sparks'],selected=ref(rec),alternatives=[{'catalog_id':o['id'],'rationale':o['reason']} for o in m['options'] if o['id']!=m['recommended']],evidence_runs=[ref(r) for r in runs if r['recipe']['id'] in ([affine['id'],exl['id']] if model=='qwen' else [glm['id'],fixed['id']])],rationale=next(o['reason'] for o in m['options'] if o['id']==m['recommended']),review={'status':'reviewed','basis':'Imported existing published selection at public commit 2a4d956; this migration makes no new recommendation decision.','date':STAMP},limitations=['Destination install and reboot qualification remain separate.','Historical incomplete metadata prevents percentage claims from these imported records.']))
  m['recommendation_record']=recommendation['id']
 def redirect(value):
  if isinstance(value,dict):
   if set(value)=={'file','path'}:return mappings.get(json.dumps(value,sort_keys=True),value)
   return {k:redirect(v) for k,v in value.items()}
  if isinstance(value,list):return [redirect(v) for v in value]
  return value
 target=ROOT/'comparisons/models.json';before=target.read_bytes();body=(json.dumps(redirect(models),indent=2)+'\n').encode()
 if before!=body:
  if read(target)!=read(old):raise ValueError('Comparison configuration changed; reconcile manually instead of overwriting it')
  with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as stream:
   temporary=Path(stream.name);stream.write(body)
  try:
   if target.read_bytes()!=before:raise ValueError('Concurrent comparison edit')
   temporary.chmod(target.stat().st_mode & 0o777);os.replace(temporary,target)
  finally:temporary.unlink(missing_ok=True)
 print(f'Imported {len(runs)} immutable run cohorts; originals retained.')
if __name__=='__main__':main()
