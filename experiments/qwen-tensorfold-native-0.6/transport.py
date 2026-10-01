#!/usr/bin/env python3
"""Independent fixed-fixture API comparison; never execute model output."""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request

FIXTURES = {
 'prose': 'Explain the tradeoffs between prefix caching and speculative decoding for a local multi-agent inference service. Cover correctness, memory use, queueing, and an example. Write approximately 800 words.',
 'code': 'Write a complete Python LRU cache using OrderedDict, a reentrant lock, positive capacity validation, get, put, and delete. Include unittest cases for eviction, updates, missing keys, and capacity one. Output code only and aim for a complete implementation.',
 'exact': 'Return JSON only: {"sum":465,"sorted":[1,3,7,9],"marker":"TF-CHECK-20260927"}. Do not add any other text.',
}

def request(base, model, kind, number, tokens, extra=None, on_first=None, api_key=None):
    prompt = f'Independent benchmark fixture {number:04d}.\n' + FIXTURES[kind]
    body = dict(model=model, messages=[dict(role='user', content=prompt)], temperature=0,
                max_tokens=tokens, reasoning_effort='medium', stream=True, stream_options={'include_usage':True},
                chat_template_kwargs={'enable_thinking':False,'reasoning_effort':'medium'}, seed=1000+number)
    body.update(extra or {})
    started_wall=time.time(); started=time.monotonic(); first=last=None; content=[]; thinking=[]; usage={}; finish=None; calls={}; error=None; runtime_stats={}; arrivals=[]; first_chunk_chars=None; visible=[]; http_status=None
    try:
        headers={'Content-Type':'application/json'}
        if api_key: headers['Authorization']='Bearer '+api_key
        req=urllib.request.Request(base.rstrip('/')+'/chat/completions', data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=600) as response:
            http_status=response.status
            for line in response:
                if not line.startswith(b'data:'): continue
                payload=line[5:].strip()
                if payload==b'[DONE]': break
                event=json.loads(payload)
                if event.get('error') is not None:
                    error=json.dumps(event['error'],sort_keys=True)[:2000]
                    break
                usage=event.get('usage') or usage
                runtime_stats=event.get('tensorfold') or runtime_stats
                for choice in event.get('choices',[]):
                    delta=choice.get('delta',{})
                    text=delta.get('content') or ''
                    reason=delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if text or reason or delta.get('tool_calls'):
                        last=time.monotonic(); arrivals.append(last)
                        if first is None:
                            first=last; first_chunk_chars=len(text)+len(reason)
                            if on_first: on_first()
                    if text: visible.append(time.monotonic())
                    content.append(text);thinking.append(reason)
                    finish=choice.get('finish_reason') or finish
                    for call in delta.get('tool_calls',[]):
                        i=call.get('index',0); acc=calls.setdefault(i,{'name':'','arguments':'','id':''})
                        for key in ['name','arguments']: acc[key]+=call.get('function',{}).get(key) or ''
                        acc['id']+=call.get('id') or ''
    except Exception as e:
        http_status=getattr(e,'code',http_status)
        error=repr(e)+((': '+e.read().decode()[:1000]) if hasattr(e,'read') else '')
    elapsed=time.monotonic()-started; output=''.join(content); n=usage.get('completion_tokens')
    gate=bool(output) and error is None
    if kind=='exact':
        try: gate=json.loads(output)=={'sum':465,'sorted':[1,3,7,9],'marker':'TF-CHECK-20260927'} and error is None
        except Exception: gate=False
    if extra and extra.get('tools'):
        try:
            c=next(iter(calls.values())); gate=c['name']=='record_probe' and json.loads(c['arguments'])=={'count':7,'enabled':True,'label':'spark'} and error is None
        except Exception: gate=False
    gaps=[b-a for a,b in zip(arrivals,arrivals[1:])];visible_gaps=[b-a for a,b in zip(visible,visible[1:])]
    return dict(first_output_event_to_visible_s=(visible[0]-first) if visible and first is not None else None,reasoning_characters=len(''.join(thinking)),started_at=started_wall,finished_at=time.time(),http_status=http_status,ttft_metric='first_reasoning_content_or_tool_event',visible_ttft=visible[0]-started if visible else None,visible_max_output_gap=max(visible_gaps,default=0),visible_output_chunks=len(visible),decode_metric='completion_tokens_minus_one_per_first_to_last_output_event_proxy',output_chunks=len(arrivals),first_chunk_chars=first_chunk_chars,request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(), max_output_gap=max(gaps,default=0), p95_output_gap=sorted(gaps)[max(0,math.ceil(len(gaps)*.95)-1)] if gaps else 0, output_sha256=hashlib.sha256(output.encode()).hexdigest(), kind=kind, fixture=number, prompt_sha256=hashlib.sha256(json.dumps(body['messages'],sort_keys=True).encode()).hexdigest(), max_tokens=tokens,
                seconds=elapsed, ttft=None if first is None else first-started,
                decode_tps=None if not n or first is None or last==first else (n-1)/(last-first),
                completion_tokens=n, usage=usage, finish=finish, error=error, gate=gate,
                output=output, reasoning=''.join(thinking), calls=calls, runtime_stats=runtime_stats)
