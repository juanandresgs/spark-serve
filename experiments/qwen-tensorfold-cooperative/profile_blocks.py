"""Diagnostic single-request, synchronization-heavy timing; not speed evidence.
CUDA elapsed includes host submission gaps, not only GPU busy time."""
import atexit,collections,json,os,resource,threading,time
from pathlib import Path

def install():
    import torch
    from tensorfold.families.qwen4_exp.cuda import forward,decode
    totals=collections.defaultdict(lambda:{'calls':0,'wall_s':0.0,'cuda_elapsed_ms':0.0})
    def timed(name,fn):
        def call(*args,**kwargs):
            b=args[{'hc_block':1,'moe_block':2,'finish':2}.get(name,3)]
            if not getattr(b,'prefill',False):return fn(*args,**kwargs)
            torch.cuda.synchronize();start=time.perf_counter();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record()
            result=fn(*args,**kwargs)
            z.record();z.synchronize();row=totals[name];row['calls']+=1;row['wall_s']+=time.perf_counter()-start;row['cuda_elapsed_ms']+=a.elapsed_time(z)
            return result
        return call
    # These are mutually separate top-level model blocks, not nested matmul timers.
    for name in ('ple_block','moe_block','attn_block','gdn_block','hc_block','finish'):
        original=getattr(forward,name)
        setattr(forward,name,timed(name,original))
    original=decode.prefill
    def prefill(*args,**kwargs):
        begin=time.perf_counter();before=resource.getrusage(resource.RUSAGE_SELF)
        result=original(*args,**kwargs);after=resource.getrusage(resource.RUSAGE_SELF)
        record={'seconds':time.perf_counter()-begin,'blocks':dict(totals),'minor_faults':after.ru_minflt-before.ru_minflt,'major_faults':after.ru_majflt-before.ru_majflt}
        with Path('/cache/prefill-block-profile.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
        totals.clear();return result
    from tensorfold.families.qwen4_exp.cuda import multi
    multi.prefill=prefill
