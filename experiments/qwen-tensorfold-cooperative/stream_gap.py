"""Measure an already-streaming response while two fresh long prefills arrive."""
import argparse,concurrent.futures,copy,hashlib,json,statistics,sys,threading,time,urllib.request
from pathlib import Path
from compare import FIXTURES
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--label',required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--out-directory',type=Path,required=True);a=p.parse_args()
root=a.out_directory;root.mkdir(parents=True,exist_ok=True)
fixture=json.loads(a.fixtures.read_text())[2]
started=threading.Event()
def call(role,index):
 messages=copy.deepcopy(fixture['messages']) if role=='long' else [{'role':'user','content':FIXTURES['code']}]
 messages[0]['content']=f'Active-stream independent fixture {role} {index}.\n'+messages[0]['content']
 body={'model':'qwen3.8-flash-next','messages':messages,'max_tokens':8192 if role=='long' else 1024,'stream':True,'temperature':0,'seed':9191,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
 begin=time.monotonic();arrivals=[];text=[];stats={};usage={};finish=None
 req=urllib.request.Request(a.base+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=1200) as response:
  for line in response:
   if not line.startswith(b'data:'):continue
   payload=line[5:].strip()
   if payload==b'[DONE]':break
   event=json.loads(payload);stats=event.get('tensorfold') or stats;usage=event.get('usage') or usage
   for choice in event.get('choices',[]):
    delta=choice.get('delta',{});content=delta.get('content') or '';reason=delta.get('reasoning_content') or ''
    if content or reason:
     arrivals.append(time.monotonic());text.append(content)
     if role=='stream':started.set()
    finish=choice.get('finish_reason') or finish
 output=''.join(text);gaps=[y-x for x,y in zip(arrivals,arrivals[1:])]
 if role=='long':passed=json.loads(output)=={'ALPHA':'amber-7419-lake','BETA':'silver-3821-oak','OMEGA':'violet-2863-moon'}
 else:passed=bool(output) and 'OrderedDict' in output and 'def ' in output
 return {'role':role,'index':index,'passed':passed,'seconds':time.monotonic()-begin,'ttft':arrivals[0]-begin,'max_output_gap':max(gaps,default=0),'p95_output_gap':sorted(gaps)[max(0,int(len(gaps)*.95)-1)] if gaps else 0,'output_chunks':len(arrivals),'usage':usage,'runtime_stats':stats,'finish':finish,'prompt_sha256':hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest(),'output_sha256':hashlib.sha256(output.encode()).hexdigest()}
begin=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(3) as pool:
 stream=pool.submit(call,'stream',0)
 assert started.wait(30),'No initial stream'
 time.sleep(.2)
 longs=[pool.submit(call,'long',i) for i in range(2)]
 rows=[f.result() for f in [stream]+longs]
result={'label':a.label,'rows':rows,'seconds':time.monotonic()-begin,'passed':all(r['passed'] for r in rows)}
(root/(a.label+'-stream-gap.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
assert result['passed']
