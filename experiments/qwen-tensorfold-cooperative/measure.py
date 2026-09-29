"""Matched public-API fixtures with fresh prefixes and bounded load; no generated execution."""
import argparse,concurrent.futures,copy,json,sys,time,statistics
from pathlib import Path
from compare import request
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--label',required=True);p.add_argument('--suite',choices=['mixed','context','short'],required=True);p.add_argument('--rep',type=int,default=0);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--out-directory',type=Path,required=True);a=p.parse_args()
root=a.out_directory;root.mkdir(parents=True,exist_ok=True);out=root/(a.label+'-'+a.suite+f'-{a.rep}.json')
fixtures=json.loads(a.fixtures.read_text());expected={'ALPHA':'amber-7419-lake','BETA':'silver-3821-oak','OMEGA':'violet-2863-moon'}
def messages(i,index):
 m=copy.deepcopy(fixtures[index]['messages']);m[0]['content']=f'Performance trial independent lane {i}, repetition {a.rep}.\n'+m[0]['content'];return m
rows=[]
def save():
 tmp=out.with_suffix('.next');tmp.write_text(json.dumps({'label':a.label,'suite':a.suite,'rep':a.rep,'rows':rows},indent=2)+'\n');tmp.replace(out)
def retrieval(i,index):
 r=request(a.base,'qwen3.8-flash-next','prose',i,8192,{'messages':messages(i,index)})
 try:r['passed']=json.loads(r['output'])==expected and not r['error']
 except Exception:r['passed']=False
 return r
if a.suite=='mixed':
 start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(24) as pool:
  longs=[pool.submit(retrieval,9000+i,2) for i in range(2)]
  shorts=[]
  for i in range(20):
   deadline=start+2+i*6
   time.sleep(max(0,deadline-time.monotonic()))
   shorts.append(pool.submit(request,a.base,'qwen3.8-flash-next','exact',8000+i,128))
  for f in concurrent.futures.as_completed(longs+shorts):
   r=f.result();r['role']='long' if f in longs else 'short';rows.append(r);save()
 elapsed=time.monotonic()-start
 short=[r for r in rows if r['role']=='short'];lat=sorted(r['ttft'] for r in short if r['ttft'] is not None)
 result={'passed':all(r.get('passed',r['gate']) for r in rows),'wall_s':elapsed,'short_p50':statistics.median(lat),'short_p95':lat[18],'short_max':max(lat),'long_seconds':[r['seconds'] for r in rows if r['role']=='long']}
elif a.suite=='context':
 prime=retrieval(9200,-1);rows.append(prime);save()
 m=messages(9200,-1)+[{'role':'assistant','content':prime['output']},{'role':'user','content':'Return the same JSON.'}]
 follow=request(a.base,'qwen3.8-flash-next','prose',9201,8192,{'messages':m});rows.append(follow);save()
 result={'passed':prime['passed'] and json.loads(follow['output'])==expected,'cold':prime['ttft'],'warm':follow['ttft'],'cached':follow['runtime_stats'].get('cached')}
else:
 for kind in ('code','prose'):
  for i in range(3):rows.append(request(a.base,'qwen3.8-flash-next',kind,7000+i,512));save()
 result={'passed':all(r['gate'] for r in rows),'median_decode':{k:statistics.median(r['decode_tps'] for r in rows if r['kind']==k) for k in ['code','prose']}}
data=json.loads(out.read_text());data['summary']=result;out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(result),flush=True)
assert result['passed']
