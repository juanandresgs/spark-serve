"""Real HTTP contract checks, including SSE validation failures."""
import argparse,json,urllib.request,urllib.error
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);a=p.parse_args()
rows=[]
def call(name,extra,expected=200,stream=False):
 body={'model':'qwen3.8-flash-next','messages':[{'role':'user','content':'Return exactly this JSON object: {"answer":7}'}],'temperature':0,'max_tokens':128,'stream':stream,'chat_template_kwargs':{'enable_thinking':False},**extra}
 req=urllib.request.Request(a.base+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=180) as r:status=r.status;raw=r.read().decode()
 except urllib.error.HTTPError as e:status=e.code;raw=e.read().decode()
 record={'name':name,'status':status,'passed':status==expected,'raw':raw}
 rows.append(record);Path(a.out).write_text(json.dumps(rows,indent=2)+'\n');return record
schema={'type':'object','properties':{'answer':{'type':'integer'}},'required':['answer'],'additionalProperties':False}
fmt={'type':'json_schema','json_schema':{'name':'answer','strict':True,'schema':schema}}
for stream in [False,True]:
 r=call('json-'+str(stream),{'response_format':fmt},stream=stream)
 if stream:
  events=[json.loads(s[5:]) for s in r['raw'].splitlines() if s.startswith('data:') and s[5:].strip()!='[DONE]']
  content=''.join(c.get('delta',{}).get('content','') for e in events for c in e.get('choices',[]))
 else:content=json.loads(r['raw'])['choices'][0]['message']['content']
 r['passed'] &= json.loads(content)=={'answer':7}
call('external-ref',{'response_format':{'type':'json_schema','json_schema':{'schema':{'$ref':'https://example.com/schema'}}}},400)
call('invalid-schema',{'response_format':{'type':'json_schema','json_schema':{'schema':{'type':'banana'}}}},400)
call('bad-budget',{'spark_reliability':{'thinking_budget':-1}},400)
call('budget-without-thinking',{'spark_reliability':{'thinking_budget':32,'answer_reserve':32}},400)
call('reserve-exceeds-limit',{'chat_template_kwargs':{'enable_thinking':True},'spark_reliability':{'thinking_budget':100,'answer_reserve':32}},400)
wrong={'type':'json_schema','json_schema':{'schema':{'type':'object','properties':{'answer':{'const':8}},'required':['answer']}}}
call('schema-mismatch',{'response_format':wrong},400)
r=call('schema-mismatch-stream',{'response_format':wrong},stream=True)
r['passed'] &= 'structured output validation failed' in r['raw'] and '"content"' not in r['raw']
call('truncated-json',{'response_format':fmt,'max_tokens':1},400)
# Ordinary code is not mistaken for a JSON contract.
r=call('plain-fence-preserved',{'messages':[{'role':'user','content':'Output exactly a markdown code fence containing print(7).'}]})
r['passed'] &= '```' in json.loads(r['raw'])['choices'][0]['message']['content']
Path(a.out).write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'passed':sum(x['passed'] for x in rows),'total':len(rows)}),flush=True)
assert all(x['passed'] for x in rows)
