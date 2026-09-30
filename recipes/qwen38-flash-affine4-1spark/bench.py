"""Matched end-to-end and decode measurements; synthetic fixtures, no execution."""
import argparse,concurrent.futures,json,statistics,sys,time
from pathlib import Path
from compare import request
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--label',required=True);p.add_argument('--reps',type=int,default=3);p.add_argument('--clients',type=int,nargs='+',choices=[1,4,8]);p.add_argument('--suite',choices=['speed','context'],default='speed');p.add_argument('--quick',action='store_true',help='Crossover subset: same fixtures, 512-token groups only');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=a.out;root.mkdir(parents=True,exist_ok=True);out=root/(a.label+'-'+a.suite+'.json')
data={'parameters':vars(a),'groups':[]}
def save():
 t=out.with_suffix('.next');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(out)
print('warmup',request(a.base,'qwen3.8-flash-next','exact',99001,128)['gate'],flush=True)
if a.suite=='speed':
 for rep in range(a.reps):
  for c,tokens in ([(1,512),(4,512),(8,512)] if a.quick else [(1,512),(1,1024),(1,4096),(4,512),(8,512)]):
   if a.clients is not None and c not in a.clients:continue
   for kind in ['code','prose']:
    n=max(4,c);start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(c) as pool:
     rows=list(pool.map(lambda i:request(a.base,'qwen3.8-flash-next',kind,50000+rep*10000+tokens+i,tokens),range(n)))
    elapsed=time.monotonic()-start
    result=dict(rep=rep,c=c,cap=tokens,kind=kind,rows=rows,wall_s=elapsed,aggregate_tps=sum(r['completion_tokens'] or 0 for r in rows)/elapsed,median_decode_tps=statistics.median([r['decode_tps'] or 0 for r in rows]),median_ttft=statistics.median([r['ttft'] or 0 for r in rows]),passed=all(r['gate'] for r in rows))
    data['groups'].append(result);save();print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
else:
 fixtures=json.loads((root/'fixtures.json').read_text())
 expected={'ALPHA':'amber-7419-lake','BETA':'silver-3821-oak','OMEGA':'violet-2863-moon'}
 for rep in range(a.reps):
  for fixture in fixtures:
   m=json.loads(json.dumps(fixture['messages']));m[0]['content']=f'AB147 fresh context repetition {rep}.\n'+m[0]['content']
   cold=request(a.base,'qwen3.8-flash-next','prose',0,8192,{'messages':m})
   m += [{'role':'assistant','content':cold['output']},{'role':'user','content':'Return the same JSON.'}]
   warm=request(a.base,'qwen3.8-flash-next','prose',0,8192,{'messages':m})
   try: passed=all(json.loads(x['output'])==expected and not x['error'] for x in (cold,warm))
   except Exception:passed=False
   result=dict(rep=rep,target=fixture['target'],cold=cold,warm=warm,passed=passed)
   data['groups'].append(result);save();print(json.dumps({'rep':rep,'target':fixture['target'],'tokens':cold['usage'].get('prompt_tokens'),'cold_ttft':cold['ttft'],'warm_ttft':warm['ttft'],'passed':passed}),flush=True)
assert all(g['passed'] for g in data['groups'])
