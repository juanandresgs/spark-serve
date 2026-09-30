"""Sanitized v1 evidence writer for the existing synthetic benchmark runner."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import re
import statistics
import uuid
from pathlib import Path

from spark_serve.evidence import canonical, load, read, safe_file, write_immutable

METADATA_KEYS = {'recipe_id', 'image_digest', 'runtime_revision', 'model_revision', 'hardware', 'software'}
HARDWARE_KEYS = {'accelerator', 'nodes', 'memory_gb_per_node', 'topology'}
SOFTWARE_KEYS = {'os', 'kernel', 'driver', 'cuda', 'runtime_version'}


def metadata(path, root):
    """Caller attests actual server identity; never mistake client inventory for server."""
    data = read(path)
    if set(data) != METADATA_KEYS or set(data['hardware']) != HARDWARE_KEYS or set(data['software']) != SOFTWARE_KEYS:
        raise ValueError('Metadata must use the exact sanitized server-identity fields from evidence/README.md')
    for key, pattern in [('image_digest', r'sha256:[a-f0-9]{64}'),
                         ('runtime_revision', r'[a-f0-9]{40}'), ('model_revision', r'[a-f0-9]{40}')]:
        if not re.fullmatch(pattern, data[key]):
            raise ValueError('Actual immutable identity required: ' + key)
    for value in data['software'].values():
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9 ._+()/-]{1,100}', value):
            raise ValueError('Software metadata must contain short version strings only')
    records = load(root)
    recipe = records[data['recipe_id']]
    if recipe['kind'] != 'recipe' or data['hardware'] != recipe['hardware']:
        raise ValueError('Actual hardware differs from the selected recipe')
    if recipe['catalog_id'] != 'qwen38-flash-affine4-1spark':
        raise ValueError('This runner emits evidence only for its Qwen Affine4 request contract')
    pins = recipe['pins']
    for actual, pin in [('runtime_revision', 'runtime_revision'), ('model_revision', 'target_revision')]:
        if pin in pins and data[actual] != pins[pin]:
            raise ValueError('Actual identity differs from recipe pin: ' + actual)
    # Source identity is checked before execution; image digest remains operator-attested.
    for rel, expected in recipe['source_files'].items():
        if hashlib.sha256(safe_file(root,rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Recipe source changed; create a new recipe record: ' + rel)
    return data, recipe


def publish_group(root, directory, meta, recipe, group, workload_files, started_at):
    """Only measurements/counts/hashes leave the raw result directory. No output/error text."""
    suite = group.get('suite', 'speed')
    if suite not in {'speed', 'context'}:
        raise ValueError('Unsupported benchmark evidence suite')
    rows = group['rows']
    n = len(rows)
    if not n:
        raise ValueError('Cannot publish an empty request group')
    counts_ok = all(type(r['completion_tokens']) is int and r['completion_tokens'] >= 0 for r in rows)
    failures = []
    if not group.get('warmup_passed',False):
        failures.append({'category':'warmup_failure','count':1,'detail':'Excluded warmup failed its exact-response gate; retain this qualification failure.'})
    bad = sum(bool(r['error']) or not r['gate'] for r in rows)
    if bad:
        failures.append({'category':'request_or_gate_failure','count':bad,'detail':'Request error or basic response gate failed; raw text retained locally only.'})
    if not counts_ok:
        failures.append({'category':'missing_token_accounting','count':sum(r['completion_tokens'] is None for r in rows),'detail':'Throughput unavailable; missing token usage is never treated as zero.'})
    fixture_hash = hashlib.sha256(canonical([r['prompt_sha256'] for r in rows])).hexdigest()
    code_hash = hashlib.sha256(canonical({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in workload_files})).hexdigest()
    conditions = {'workload':group['kind'], 'fixture_sha256':fixture_hash,
                  'concurrency':group['c'], 'cache':'uncontrolled',
                  'request':{'max_tokens':group['cap'],'stream':True,'thinking':False},
                  'sampling':{'temperature':0,'seed_policy':'1000 + fixture number',
                              'fixture_numbers':[r['fixture'] for r in rows]},
                  'warmup':'One excluded exact-response request before the suite; prefix cache not reset between groups.'}
    if suite == 'context':
        cached = rows[0]['usage'].get('prompt_tokens_details', {}).get('cached_tokens')
        conditions['cache'] = ('cold' if cached == 0 else 'warm') if type(cached) is int else 'unknown'
        conditions['request'].update(phase=group['phase'], target_tokens=group['target'])
        conditions['warmup'] = 'One excluded exact-response request; appended request follows its priming response. Cache state uses API-reported cached prompt tokens only.'
    metrics = []
    def add(id, metric, value, unit, definition, aggregation, samples, sample_unit, population):
        metrics.append(dict(id=id,metric=metric,value=value,unit=unit,definition=definition,
                            aggregation=aggregation,samples=samples,sample_unit=sample_unit,
                            population=population,conditions=conditions,source=None))
    add('throughput','end_to_end_throughput',sum(r['completion_tokens'] for r in rows)/group['wall_s'] if counts_ok else None,
        'tokens/s','Sum of all reported completion tokens divided by request-group wall seconds, including prefill, queueing and failed-request time.',
        'single',n,'requests','all attempted requests; unavailable when any token count is missing')
    for key, metric, definition, unit in [
        ('ttft','ttft','Client seconds from request start until first nonempty content, reasoning or tool delta; includes network and queueing.','s'),
        ('seconds','response_latency','Client seconds from request start until stream completion or error; includes network and queueing.','s'),
        ('decode_tps','decode_rate_proxy','Per-request (completion_tokens - 1) / (last nonempty delta time - first nonempty delta time); chunk-based proxy, not exact token timestamps.','tokens/s')]:
        values=[r[key] for r in rows if r[key] is not None and r['error'] is None]
        add(key,metric,statistics.median(values) if values else None,unit,definition,'median',len(values) or None,'requests','successful transport requests with available measurement; basic gate failures retained')
    if suite == 'context':
        add('correct','correct',n-bad,'count','Strict expected JSON retrieval answers correct.','count',n,'requests','all attempted retrieval requests')
        add('cases','cases',n,'count','Attempted retrieval requests.','count',n,'requests','all attempted retrieval requests')
    now=datetime.now(timezone.utc).isoformat()
    qualification={k:{'status':'not_tested','scope':'This throughput run does not qualify '+k+'.'} for k in ['build','api_tool','correctness','clean_install','restart','reboot','performance','endurance']}
    qualification['performance']={'status':'failed' if failures else 'passed','scope':'This synthetic request group only; response gate is not a correctness evaluation.'}
    if suite == 'context':
        qualification['correctness'] = {'status':'failed' if bad else 'passed','scope':'This synthetic three-marker strict JSON retrieval request only; not broad model quality.'}
        qualification['performance']['scope'] = 'This single context request only; cold/warm labels require API-reported cache accounting.'
    def token_sum(key):
        values=[r['usage'].get(key) for r in rows]
        return sum(values) if all(type(v) is int and v >= 0 for v in values) else None
    observations=[{'fixture_sha256':r['prompt_sha256'],'prompt_tokens':r['usage'].get('prompt_tokens'),
                   'completion_tokens':r['completion_tokens'],
                   'cached_tokens':r['usage'].get('prompt_tokens_details',{}).get('cached_tokens'),
                   'ttft_seconds':r['ttft'],'response_seconds':r['seconds'],
                   'decode_rate_proxy':r['decode_tps'],'outcome':'passed_basic_gate' if r['gate'] and not r['error'] else 'failed'} for r in rows]
    cached=[o['cached_tokens'] for o in observations]
    accounting={'requests_attempted':n,'requests_completed':sum(r['error'] is None for r in rows),
                'prompt_tokens':token_sum('prompt_tokens'),'completion_tokens':sum(r['completion_tokens'] for r in rows) if counts_ok else None,
                'cached_tokens':sum(cached) if all(type(v) is int for v in cached) else None,'wall_seconds':group['wall_s']}
    record=dict(accounting=accounting,observations=observations,schema_version=1,kind='run',id='run-'+uuid.uuid4().hex,origin='local',recorded_at=now,executed_at=started_at,
                recipe={'id':recipe['id'],'fingerprint':recipe['fingerprint']},
                identity={'relationship':'attested_exact','image_digest':meta['image_digest'],
                          'runtime_revision':meta['runtime_revision'],'model_revision':meta['model_revision'],
                          'attestation':'Operator-supplied server identity; local recipe source hashes checked before benchmark. Server image identity is not remotely verified by this client.'},
                environment={'hardware':meta['hardware'],'software':meta['software'],'unknowns':[]},
                workload_version='qwen-'+suite+'-v1-sha256-'+code_hash,measurements=metrics,failures=failures,
                qualification=qualification,sources=[],external_sources=[],
                limitations=['Synthetic workload only; no broad correctness or lifecycle qualification.',
                             'Speed cache is uncontrolled; context cache state is unknown unless the API reports cached prompt tokens. No process restart or cold OS-cache test is implied.',
                             'Decode rate is a streaming-chunk proxy; report separately from end-to-end throughput.'])
    return write_immutable(record,directory,root)
