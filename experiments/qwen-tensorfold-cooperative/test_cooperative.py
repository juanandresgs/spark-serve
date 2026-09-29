import importlib.util
import sys
import socket
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass,field
from cooperative import CooperativeScheduler,_current,_http,ClientGone,peer_gone

# Only the GPU boundary is simulated; tests exercise the actual dispatcher.
@dataclass
class Stream:
    prompt:list
    count:int
    sampling:object=None
    draft:bool=True
    done:bool=False
    out:list=field(default_factory=list)
    def stats(self):return {'tokens':self.out}

class Decoder:
    def __init__(self):
        self.streams={};self.events=[];self.serial=threading.Lock();self.started=threading.Event()
    def live(self):return len(self.streams)
    def admit(self,s):
        job=_current.admission;job.state=types.SimpleNamespace(pos=0)
        for i in range(0,len(s.prompt),2048):
            with self.serial:
                time.sleep(.015);job.state.pos=min(i+2048,len(s.prompt));self.events.append(('prefill',len(s.prompt)))
            self.started.set();job.checkpoint()
        with self.serial:
            s.sid=id(s);s.out=[len(s.prompt)];self.streams[s.sid]=s;s.emit(s.out)
            s.done=s.count==1
    def round(self):
        with self.serial:
            for s in list(self.streams.values()):
                if not s.done:
                    s.out.append(len(s.prompt));s.emit([s.out[-1]]);s.done=len(s.out)>=s.count
            return [s for s in self.streams.values() if s.done]
    def finish(self,done):
        for s in done:self.streams.pop(s.sid)
    def drop(self):
        s=list(self.streams.values());self.streams.clear();return s

class Tests(unittest.TestCase):
    def setUp(self):
        mod=types.ModuleType('tensorfold.cuda.streams');mod.Stream=Stream
        self.old=sys.modules.get('tensorfold.cuda.streams');sys.modules['tensorfold.cuda.streams']=mod
    def tearDown(self):
        if self.old is None:sys.modules.pop('tensorfold.cuda.streams',None)
        else:sys.modules['tensorfold.cuda.streams']=self.old
    def test_short_finishes_before_two_long_prompts(self):
        d=Decoder();s=CooperativeScheduler(d,synchronize=lambda:None)
        with ThreadPoolExecutor(3) as pool:
            long1=pool.submit(s.submit,[1]*20000,5,None,True,lambda x:False)
            long2=pool.submit(s.submit,[2]*18000,5,None,True,lambda x:False)
            self.assertTrue(d.started.wait(1))
            short=pool.submit(s.submit,[3]*10,3,None,True,lambda x:False)
            self.assertEqual(short.result(2)['tokens'],[10]*3)
            self.assertFalse(long1.done());self.assertFalse(long2.done())
            self.assertEqual(long1.result(3)['tokens'],[20000]*5)
            self.assertEqual(long2.result(3)['tokens'],[18000]*5)
        self.assertFalse(s.jobs);self.assertFalse(s.replies);self.assertEqual(d.live(),0)
    def test_queue_drains_without_cross_request_output(self):
        d=Decoder();s=CooperativeScheduler(d,synchronize=lambda:None)
        with ThreadPoolExecutor(8) as pool:
            fs=[pool.submit(s.submit,[n]*(n*1000),4,None,True,lambda x:False) for n in range(1,9)]
            for n,f in enumerate(fs,1):self.assertEqual(f.result(5)['tokens'],[n*1000]*4)
        self.assertFalse(s.jobs);self.assertFalse(s.replies)
    def test_failed_admission_does_not_stop_other_requests(self):
        class Failing(Decoder):
            def admit(self,s):
                if len(s.prompt)==99:raise ValueError('synthetic admission failure')
                return super().admit(s)
        d=Failing();s=CooperativeScheduler(d,synchronize=lambda:None)
        with ThreadPoolExecutor(2) as pool:
            bad=pool.submit(s.submit,[0]*99,4,None,True,lambda x:False)
            good=pool.submit(s.submit,[1]*4096,4,None,True,lambda x:False)
            with self.assertRaises(ValueError):bad.result(2)
            self.assertEqual(good.result(2)['tokens'],[4096]*4)
        self.assertFalse(s.replies)
    def test_disconnect_during_prefill_releases_dispatcher(self):
        d=Decoder();s=CooperativeScheduler(d,synchronize=lambda:None)
        a,b=socket.socketpair()
        def request():
            _http.connection=a
            try:return s.submit([1]*100000,5,None,True,lambda x:False)
            finally:_http.connection=None
        try:
            with ThreadPoolExecutor(1) as pool:
                f=pool.submit(request);self.assertTrue(d.started.wait(1));b.close()
                with self.assertRaises(ClientGone):f.result(1)
            self.assertEqual(s.submit([2]*10,2,None,True,lambda x:False)['tokens'],[10]*2)
            self.assertFalse(s.replies)
        finally:a.close();b.close()
    def test_disconnect_probe_does_not_consume_bytes(self):
        a,b=socket.socketpair()
        try:
            b.sendall(b'x');self.assertFalse(peer_gone(a));self.assertEqual(a.recv(1),b'x')
            b.close();self.assertTrue(peer_gone(a))
        finally:a.close();b.close()
if __name__=='__main__':unittest.main()
