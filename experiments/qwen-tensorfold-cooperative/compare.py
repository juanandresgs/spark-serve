#!/usr/bin/env python3
"""Independent fixed-fixture API comparison; never execute model output."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

FIXTURES = {
 'prose': 'Explain the tradeoffs between prefix caching and speculative decoding for a local multi-agent inference service. Cover correctness, memory use, queueing, and an example. Write approximately 800 words.',
 'code': 'Write a complete Python LRU cache using OrderedDict, a reentrant lock, positive capacity validation, get, put, and delete. Include unittest cases for eviction, updates, missing keys, and capacity one. Output code only and aim for a complete implementation.',
 'exact': 'Return JSON only: {"sum":465,"sorted":[1,3,7,9],"marker":"TF-CHECK-20260927"}. Do not add any other text.',
}

def request(base, model, kind, number, tokens, extra=None):
    prompt = f'Independent benchmark fixture {number:04d}.\n' + FIXTURES[kind]
    body = dict(model=model, messages=[dict(role='user', content=prompt)], temperature=0,
                max_tokens=tokens, stream=True, stream_options={'include_usage':True},
                chat_template_kwargs={'enable_thinking':False}, seed=1000+number)
    body.update(extra or {})
    started=time.monotonic(); first=last=None; content=[]; thinking=[]; usage={}; finish=None; calls={}; error=None; runtime_stats={}
    try:
        req=urllib.request.Request(base.rstrip('/')+'/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=1200) as response:
            for line in response:
                if not line.startswith(b'data:'): continue
                payload=line[5:].strip()
                if payload==b'[DONE]': break
                event=json.loads(payload)
                usage=event.get('usage') or usage
                runtime_stats=event.get('tensorfold') or runtime_stats
                for choice in event.get('choices',[]):
                    delta=choice.get('delta',{})
                    text=delta.get('content') or ''
                    reason=delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if text or reason or delta.get('tool_calls'):
                        last=time.monotonic()
                        if first is None: first=last
                    content.append(text);thinking.append(reason)
                    finish=choice.get('finish_reason') or finish
                    for call in delta.get('tool_calls',[]):
                        i=call.get('index',0); acc=calls.setdefault(i,{'name':'','arguments':''})
                        for key in acc: acc[key]+=call.get('function',{}).get(key) or ''
    except Exception as e: error=repr(e)+((': '+e.read().decode()[:1000]) if hasattr(e,'read') else '')
    elapsed=time.monotonic()-started; output=''.join(content); n=usage.get('completion_tokens')
    gate=bool(output) and error is None
    if kind=='exact':
        try: gate=json.loads(output)=={'sum':465,'sorted':[1,3,7,9],'marker':'TF-CHECK-20260927'} and error is None
        except Exception: gate=False
    if extra and extra.get('tools'):
        try:
            c=next(iter(calls.values())); gate=c['name']=='record_probe' and json.loads(c['arguments'])=={'count':7,'enabled':True,'label':'spark'} and error is None
        except Exception: gate=False
    return dict(kind=kind, fixture=number, prompt_sha256=hashlib.sha256(json.dumps(body['messages'],sort_keys=True).encode()).hexdigest(), max_tokens=tokens,
                seconds=elapsed, ttft=None if first is None else first-started,
                decode_tps=None if not n or first is None or last==first else (n-1)/(last-first),
                completion_tokens=n, usage=usage, finish=finish, error=error, gate=gate,
                output=output, reasoning=''.join(thinking), calls=calls, runtime_stats=runtime_stats)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--model',required=True)
    p.add_argument('--out',required=True);p.add_argument('--label',required=True)
    p.add_argument('--suite',choices=['speed','checks','exactness','continuation'],default='speed');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True); rows=[]; groups=[]
    def save():
        target=out/(a.label+'.json');tmp=target.with_suffix('.next')
        tmp.write_text(json.dumps(dict(parameters=vars(a), rows=rows, groups=groups),indent=2));tmp.replace(target)
    def run(kind, ids, tokens, c, extra=None):
        start=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
            results=list(pool.map(lambda i:request(a.base,a.model,kind,i,tokens,extra),ids))
        elapsed=time.monotonic()-start;rows.extend(results)
        rates=[r['decode_tps'] for r in results if r['decode_tps'] is not None]
        ttfts=[r['ttft'] for r in results if r['ttft'] is not None]
        group=dict(kind=kind,c=c,tokens=tokens,n=len(results),passed=sum(r['gate'] for r in results),seconds=elapsed,
                   aggregate_tps=sum(r['completion_tokens'] or 0 for r in results)/elapsed,
                   median_decode_tps=statistics.median(rates) if rates else None,
                   median_ttft=statistics.median(ttfts) if ttfts else None,
                   median_seconds=statistics.median(r['seconds'] for r in results))
        groups.append(group);save();print(json.dumps(group),flush=True)
    if a.suite in ['exactness','continuation']:
        fixture=json.loads((out.parent/'fixtures.json').read_text())[-1]
        # Explicit continuation test, independent of drafted/serial equality.
        # Reuse the immediately preceding completed context request when present;
        # a standalone continuation run primes the freshly restored baseline.
        prior=out/(a.label.removesuffix('-exactness')+'-context.json')
        reused=a.suite=='exactness' and prior.exists()
        prime=json.loads(prior.read_text())[-1] if reused else request(a.base,a.model,'prose',fixture['target'],8192,{'messages':fixture['messages']})
        messages=fixture['messages']+[{'role':'assistant','content':prime['output']},{'role':'user','content':'Return the same JSON.'}]
        follow=request(a.base,a.model,'prose',fixture['target'],8192,{'messages':messages})
        expected={'ALPHA':'amber-7419-lake','BETA':'silver-3821-oak','OMEGA':'violet-2863-moon'}
        for mode,row in [('prime',prime),('followup',follow)]:
            try:passed=json.loads(row['output'])==expected and row['error'] is None
            except Exception:passed=False
            rows.append(row);groups.append({k:row.get(k) for k in ['seconds','ttft','usage','error','runtime_stats']}|dict(kind='continuation',mode=mode,passed=passed,reused_request=reused and mode=='prime'));save();print(json.dumps(groups[-1]),flush=True)
    if a.suite=='continuation':return
    if a.suite=='speed':
        # Excluded warm-up before timing; fresh numbered fixtures identical across arms.
        print('warmup',request(a.base,a.model,'exact',999,64)['gate'],flush=True)
        for tokens in [512,1024,4096]:
            for kind in ['prose','code']:run(kind,range(tokens,tokens+3),tokens,1)
        for c in [4,8]:
            for kind in ['prose','code']:run(kind,range(2000,2000+c),512,c)
    elif a.suite=='exactness':
        for kind in ['code','prose']:
            for temperature in [0,0.7]:
                pair=[]
                for draft in [True,False]:
                    body=dict(model=a.model,messages=[dict(role='user',content=FIXTURES[kind])],temperature=temperature,seed=1234,max_tokens=256,stream=False,draft=draft,return_token_ids=True,chat_template_kwargs={'enable_thinking':False})
                    try:
                        req=urllib.request.Request(a.base+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                        with urllib.request.urlopen(req,timeout=300) as response:r=json.load(response)
                        pair.append(r)
                    except Exception as e:pair.append({'error':repr(e)})
                ids=[r.get('tensorfold',{}).get('token_ids') for r in pair]
                passed=bool(ids[0]) and ids[0]==ids[1]
                rows.extend(pair);groups.append(dict(kind=kind,temperature=temperature,passed=passed,token_counts=[len(v) if v else None for v in ids]));save();print(json.dumps(groups[-1]),flush=True)
    else:
        run('exact',range(3000,3003),128,1)
        tool={'type':'function','function':{'name':'record_probe','description':'Record the probe','parameters':{'type':'object','properties':{'count':{'type':'integer'},'enabled':{'type':'boolean'},'label':{'type':'string'}},'required':['count','enabled','label'],'additionalProperties':False}}}
        extra={'messages':[{'role':'user','content':'Call record_probe with count 7, enabled true, label spark. Do not answer in text.'}], 'tools':[tool], 'tool_choice':'auto'}
        run('exact',range(3100,3103),256,1,extra)
if __name__=='__main__': main()
