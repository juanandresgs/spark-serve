"""Check admission and committed-token accounting without a GPU."""
import inspect
import concurrent.futures, threading, time, unittest
from unittest.mock import patch
import tensorfold.cuda.server as server
from telemetry import install

class TelemetryTests(unittest.TestCase):
 def test_admission_tokens_and_failure_cleanup(self):
  release=threading.Event()
  class Engine:
   def generate(self,prompt,max_tokens,sampling,on_tokens,draft=True,**kwargs):
    release.wait(5);on_tokens([10,11,12])
    if kwargs.get('fail'):raise RuntimeError('fixture')
    return {}
  def init(app):app.engine=Engine()
  with patch.object(server.App,'__init__',init),patch.object(server,'make_handler',server.make_handler):
   install();app=server.App()
   self.assertIn("draft",inspect.signature(app.engine.generate).parameters)
   with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    futures=[pool.submit(app.engine.generate,[1,2],3,None,lambda _:False,fail=i==5) for i in range(6)]
    end=time.monotonic()+3
    while app.spark_counts['waiting']!=2 and time.monotonic()<end:time.sleep(.01)
    self.assertEqual(app.spark_counts['running'],4);self.assertEqual(app.spark_counts['waiting'],2)
    release.set()
    for i,f in enumerate(futures):
     if i==5:
      with self.assertRaises(RuntimeError):f.result()
     else:f.result()
   self.assertEqual(app.spark_counts,dict(running=0,waiting=0,prompt=10,output=18,completed=5,errors=1,cancelled=0))
if __name__=='__main__':unittest.main()
