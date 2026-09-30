#!/usr/bin/env python3
"""Shared synthetic full-context fixtures; admission is not a retrieval result."""
import argparse, hashlib, json, time, urllib.request
from pathlib import Path
from compare import request
p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--model-dir',type=Path);p.add_argument('--model',required=True);p.add_argument('--fixtures',required=True);p.add_argument('--out');p.add_argument('--prepare',action='store_true');p.add_argument('--all-rungs',action='store_true');p.add_argument('--repeat',action='store_true');a=p.parse_args()
path=Path(a.fixtures)
def messages(lines,nonce):
 rows=[f'Record {i:06d}: routine blue widget inventory checked; no special access code.' for i in range(lines)]
 for pos,key,value in reversed([(lines//8,'ALPHA','amber-7419-lake'),(lines//2,'BETA','silver-3821-oak'),(7*lines//8,'OMEGA','violet-2863-moon')]):
  rows.insert(pos,f'IMPORTANT: The unique {key} access code is {value}.')
 return [dict(role='user',content=f'Independent retrieval fixture {nonce}. Read these records.\n'+'\n'.join(rows)+'\nReturn JSON only with exactly keys ALPHA, BETA, OMEGA and their access codes.')]
if a.prepare:
 rows=[]
 for target in [8192,65536,131072,253824]:
  lines=target//18
  for _ in range(5):
   msg=messages(lines,target)
   body=dict(model=a.model,messages=msg,add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False})
   if a.model_dir:
    from tensorfold.cuda.server import ChatTemplate
    from tokenizers import Tokenizer
    template=ChatTemplate(a.model_dir)
    tok=Tokenizer.from_file(str(a.model_dir/'tokenizer.json'))
    count=len(tok.encode(template.render(msg,tools=None,enable_thinking=False),add_special_tokens=False).ids)
   else:
    if not a.base: p.error('--base or --model-dir is required')
    req=urllib.request.Request(a.base+'/tokenize',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as r: count=json.load(r)['count']
   if abs(count-target)<32:break
   lines=max(1,int(lines*target/count))
  assert count+8192<=262144
  rows.append(dict(target=target,input_tokens=count,messages=msg,sha256=hashlib.sha256(json.dumps(msg).encode()).hexdigest()))
 path.write_text(json.dumps(rows))
else:
 results=[]
 fixtures=json.loads(path.read_text())
 for fixture in fixtures if a.all_rungs else fixtures[-1:]:
  for warm in ([False,True] if a.repeat else [False]):
   row=request(a.base+'/v1',a.model,'prose',fixture['target'],8192,{'messages':fixture['messages']})
   try: row['retrieval_pass']=json.loads(row['output'])==dict(ALPHA='amber-7419-lake',BETA='silver-3821-oak',OMEGA='violet-2863-moon') and row['error'] is None
   except Exception:row['retrieval_pass']=False
   row.update(target=fixture['target'],tokenizer_input_tokens=fixture['input_tokens'],fixture_sha256=fixture['sha256'],warm_repeat=warm)
   results.append(row);out=Path(a.out);pending=out.with_suffix('.next');pending.write_text(json.dumps(results,indent=2));pending.replace(out)
   print(json.dumps({k:v for k,v in row.items() if k not in ['output','reasoning','calls']}),flush=True)
   if row['error']:raise SystemExit('Context request failed; remaining sizes not tested')
