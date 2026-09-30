"""Independent replay and held-out evaluation; raw outputs stay outside Git."""
import argparse,concurrent.futures,json,sys,time
from pathlib import Path
from compare import request
from fixtures import make
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--indices',default='all');p.add_argument('--draft',choices=['on','off'],default='on');p.add_argument('--temperature',type=float,default=0);p.add_argument('--clients',type=int,default=4);p.add_argument('--budget',type=int,default=0);p.add_argument('--seed',type=int,default=61000);p.add_argument('--schema',action='store_true');p.add_argument('--fixture-seed',type=int,default=147);a=p.parse_args();fixtures=make(a.fixture_seed)
indices=range(len(fixtures)) if a.indices=='all' else [int(x) for x in a.indices.split(',')]
rows=[];out=Path(a.out)
def call(i):
 kind,prompt,answer=fixtures[i]
 body={'messages':[{'role':'user','content':prompt+' Return only a JSON object with one integer field named answer.'}],'chat_template_kwargs':{'enable_thinking':True},'draft':a.draft=='on','temperature':a.temperature,'top_k':20,'top_p':.95,'seed':a.seed+i,'return_token_ids':True}
 if a.budget:body['spark_reliability']={'thinking_budget':a.budget,'answer_reserve':1024}
 if a.schema:body['response_format']={'type':'json_schema','json_schema':{'name':'answer','strict':True,'schema':{'type':'object','properties':{'answer':{'type':'integer'}},'required':['answer'],'additionalProperties':False}}}
 start=time.time();r=request(a.base,'qwen3.8-flash-next','prose',60000+i,8192,body)
 if not r['finish'] and not r['output'] and not r['error']:
  r['error']='No terminal completion received; inspect server/API error evidence'
 try:passed=json.loads(r['output'])=={'answer':answer} and not r['error']
 except ValueError:passed=False
 return dict(index=i,category=kind,expected=answer,passed=passed,start=start,response=r)
with concurrent.futures.ThreadPoolExecutor(a.clients) as pool:
 for r in pool.map(call,indices):
  rows.append(r);tmp=out.with_suffix('.next');tmp.write_text(json.dumps({'parameters':vars(a),'rows':rows},indent=2)+'\n');tmp.replace(out)
  print(json.dumps({k:r[k] for k in ['index','passed']}|{'seconds':r['response']['seconds'],'tokens':r['response']['completion_tokens'],'finish':r['response']['finish'],'error':r['response']['error']}),flush=True)
