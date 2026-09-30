import unittest
from reliability import contract,normalize,combined_stats
S={'type':'object','properties':{'answer':{'type':'integer'}},'required':['answer'],'additionalProperties':False}
class Contract(unittest.TestCase):
 def test_phase_metrics(self):
  a={'prefill_s':2.0,'decode_s':50.0,'rounds':100,'drafts':True,'cached':0,'min_rows':4}
  b={'prefill_s':1.0,'decode_s':.1,'rounds':1,'drafts':True,'cached':2000,'min_rows':2}
  c=combined_stats(a,b)
  self.assertEqual(c,dict(prefill_s=3.0,decode_s=50.1,rounds=101,drafts=True,cached=0,min_rows=2))
  self.assertEqual(a['rounds'],100)
 def test_nonobject_body(self):
  for value in [None,[],5]:
   with self.assertRaises(ValueError):contract(value)
 def test_complete_fence(self):
  self.assertEqual(normalize('```json\n{"answer":7}\n```',S,'stop'),('{"answer":7}',True))
 def test_valid_preserved(self):
  x=' { "answer": 7 } ';self.assertEqual(normalize(x,S,'stop'),(x,False))
 def test_fail_closed(self):
  for x in ['prefix\n```json\n{"answer":7}\n```','{"answer":"7"}','{"answer":7,"answer":8}','{"answer":NaN}','{"answer":1e999}','{"answer":7,"x":1}','{"answer":7','```python\n{"answer":7}\n```']:
   with self.subTest(x=x),self.assertRaises(Exception):normalize(x,S,'stop')
 def test_length_rejected(self):
  with self.assertRaises(ValueError):normalize('{"answer":7}',S,'length')
 def test_refs(self):
  with self.assertRaises(ValueError):contract({'response_format':{'type':'json_schema','json_schema':{'schema':{'$ref':'https://example.com'}}}})
 def test_bad_budget(self):
  for budget in [True,-1,1.2,'5']:
   with self.assertRaises(ValueError):contract({'spark_reliability':{'thinking_budget':budget}})
 def test_tools_not_silently_changed(self):
  with self.assertRaises(ValueError):contract({'tools':[{}],'response_format':{'type':'json_object'}})

class Continuation(unittest.TestCase):
 def test_reserved_continuation_and_usage(self):
  import sys,types,dataclasses
  from unittest.mock import patch
  from reliability import install
  @dataclasses.dataclass
  class Prepared:
   prompt:list
   max_tokens:int
   tools:list
   thinking:bool
  class Tokenizer:
   def encode(self,text,**kwargs):return types.SimpleNamespace(ids=[1001,1002])
   def decode(self,ids,**kwargs):return 'unfinished reasoning'
  class Error(Exception):pass
  class App:
   tok=Tokenizer()
   def prepare(self,body,chat):return Prepared([99],128,[],True)
   def run(self,body,chat,emit,*,prepared=None,cancelled=None):
    self.seen.append(prepared)
    if len(self.seen)==1:
     return {'finish':'length','content':'','reasoning':'unfinished reasoning','prompt_tokens':1,'completion_tokens':32,'stats':{'token_ids':list(range(32))},'final':{},'calls':None}
    return {'finish':'stop','content':'```json\n{"answer":7}\n```','reasoning':'','prompt_tokens':35,'completion_tokens':10,'stats':{'token_ids':[88]*10},'final':{},'calls':None}
  server=types.ModuleType('tensorfold.cuda.server');server.App=App;server.RequestError=Error;server.RequestCancelled=Error;server.token_sha=lambda ids:'hash'
  parent=types.ModuleType('tensorfold');cuda=types.ModuleType('tensorfold.cuda');parent.cuda=cuda;cuda.server=server
  with patch.dict(sys.modules,{'tensorfold':parent,'tensorfold.cuda':cuda,'tensorfold.cuda.server':server}):
   install();app=App();app.seen=[];emitted=[]
   result=app.run({'seed':61000,'spark_reliability':{'thinking_budget':32,'answer_reserve':32},'response_format':{'type':'json_schema','json_schema':{'schema':S}}},True,lambda d:emitted.append(d) or True)
  self.assertEqual(app.seen[1].prompt,[99]+list(range(32))+[1001,1002])
  self.assertEqual(app.seen[1].max_tokens,94)
  self.assertFalse(app.seen[1].thinking)
  self.assertEqual(result['completion_tokens'],42)
  self.assertEqual(result['prompt_tokens'],1)
  self.assertEqual(result['content'],'{"answer":7}')
  self.assertTrue(result['stats']['spark_reliability']['forced_exit'])
  self.assertTrue(result['stats']['spark_reliability']['normalized_fence'])
  self.assertNotIn('token_ids',result['stats'])
  self.assertEqual(emitted,[{'reasoning_content':'unfinished reasoning','content':'{"answer":7}'}])

if __name__=='__main__':unittest.main()
