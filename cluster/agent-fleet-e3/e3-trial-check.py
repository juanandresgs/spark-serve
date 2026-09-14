"""Isolated, paired current-E2 / curated-E3 MoE comparison. Not model token speed."""
import argparse,importlib.util,json,os,random,statistics,sys,time,types
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--small',action='store_true');p.add_argument('--out',required=True);a=p.parse_args()
os.environ.update(EXL3_FUSED_MOE='1',EXL3_FAT_KERNEL='1',EXL3_FAT_GROUPED='0',EXL3_MOE_ROW_TILE='0',EXL3_FAT_SORTED='0',EXL3_FAT_BATCHED='0',EXL3_TEMP_ROWS_FUSED='128',EXL3_FAT_EXPERT_LOG='0',MAX_NUM_BATCHED_TOKENS='2048')
import torch
sys.path.insert(0,'/build')
from vllm.model_executor.layers.quantization import exl3 as old
spec=importlib.util.spec_from_file_location('e3_trial_overlay','/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py');new=importlib.util.module_from_spec(spec);sys.modules[spec.name]=new;spec.loader.exec_module(new)
assert new.exl3_fat_moe_symbols(),'compiled E3 symbols missing'
H,I,E,K=(256,128,8,2) if a.small else (4096,1024,288,8)
CAP=64 if a.small else 128
rec={'small':a.small,'shape':{'hidden':H,'intermediate':I,'experts':E,'topk':K},'parity':[],'cases':[],'started':time.time(),'source_commit':'dc6936cea8fd7b2e7ee5b7a48a5aa193857ca489'}
def save():Path(a.out).write_text(json.dumps(rec,indent=2)+'\n')
def phase(name):print(json.dumps({'phase':name,'time':time.time()}),flush=True)
phase('construct-layer')
g=torch.Generator(device='cpu').manual_seed(909)
method=new.Exl3MoEMethod(types.SimpleNamespace(swiglu_limit=10.0),new.Exl3Config());layer=torch.nn.Module()
method.create_weights(layer,num_experts=E,hidden_size=H,intermediate_size_per_partition=I,params_dtype=torch.float16)
with torch.no_grad():
 for name in ('w13_trellis','w2_trellis'):
  t=getattr(layer,name);t.copy_(torch.randint(-30000,30000,tuple(t.shape),dtype=torch.int16,generator=g))
 for name in ('w13_suh','w13_svh','w2_suh','w2_svh'):
  t=getattr(layer,name);t.copy_((torch.randn(tuple(t.shape),generator=g)*.5).half())
 layer.w13_suh[:,1].copy_(layer.w13_suh[:,0]);layer.w13_mcg.fill_(new.MCG_MARKER_SIGNED_INT32);layer.w2_mcg.fill_(new.MCG_MARKER_SIGNED_INT32)
layer=layer.to('cuda');method.process_weights_after_loading(layer)
rec['device']=torch.cuda.get_device_name();rec['initial_allocated']=torch.cuda.memory_allocated()

def inputs(n,skew,invalid=False):
 gen=torch.Generator().manual_seed(1200+n)
 x=torch.randn(n,H,generator=gen).half().cuda()
 probs=1/torch.arange(1,E+1).float().pow(skew)
 ids=torch.multinomial(probs.expand(n,-1),K,replacement=False,generator=gen)
 if invalid:ids[::7,0]=-1;ids[::11,-1]=E+2
 w=torch.rand(n,K,generator=gen).softmax(-1).half()
 return x,ids.cuda(),w.cuda()

def select(tier,cap):
 os.environ['EXL3_TEMP_ROWS_FUSED']=str(cap);os.environ['EXL3_FAT_GROUPED']='1' if tier=='e3' else '0'
 new.build_exl3_fused_state(layer,layer._exl3_inners);new._record_exl3_fat_resolution(layer)
 if tier=='e3':assert layer._exl3_fat_effective_tier=='grouped'
 return old if tier=='current-e2' else new

def err(ref,y):
 d=ref.float()-y.float();scale=max(float(ref.float().pow(2).mean().sqrt()),1e-6)
 return {'finite':bool(torch.isfinite(y).all()),'maxabs':float(d.abs().max()),'nrmse':float(d.pow(2).mean().sqrt())/scale,'ref_max':float(ref.abs().max())}

phase('numerical-gates')
# Fixed before observing E3: no more than 1.5x current E2 error plus small
# rounding floors, and a coarse absolute-error guard. This is a layer gate.
for n,skew,invalid in ([(256,1.0,False),(257,1.0,True)] if a.small else [(512,1.0,False),(513,1.0,True),(2048,1.0,False)]):
 x,ids,w=inputs(n,skew,invalid)
 ref=old.apply_exl3_python_loop(x,ids,w,layer._exl3_inners,None,10.0).to(x.dtype)
 m=select('current-e2',CAP);base=m.apply_exl3_experts(x,ids,w,layer);e2=err(ref,base)
 assert e2['finite'],'non-finite current E2 reference'
 row={'tokens':n,'invalid_routes':invalid,'current_e2':e2,'variants':{}}
 for tier,cap in [('new-e2',CAP),('e3',CAP),('e3',64)]:
  m=select(tier,cap);y=m.apply_exl3_experts(x,ids,w,layer);e=err(ref,y);repeat=m.apply_exl3_experts(x,ids,w,layer)
  e['repeat']=err(y,repeat)
  e['ok']=e['finite'] and e['maxabs']<=1.5*e2['maxabs']+.001*e2['ref_max'] and e['nrmse']<=1.5*e2['nrmse']+.0001 and e['maxabs']<max(.15,.08*max(1,e2['ref_max']))
  row['variants'][tier+'-'+str(cap)]=e
 rec['parity'].append(row);save();print(json.dumps(row),flush=True)
 assert all(v['ok'] for v in row['variants'].values()),'numerical gate failed; no timing/promotion'
 del x,ids,w,ref,base,y,repeat
 torch.cuda.empty_cache()

phase('paired-timing')
for n in ([256] if a.small else [64,512,2048]):
 for skew in ([1.0] if a.small else [0.0,1.0,1.5]):
  x,ids,w=inputs(n,skew)
  valid=ids[(ids>=0)&(ids<E)];counts=torch.bincount(valid,minlength=E)
  configs=[('current-e2',CAP),('new-e2',CAP),('e3',CAP),('e3',64)]
  samples={str(c):[] for c in configs};calls={};scratch={}
  for tier,cap in configs:
   m=select(tier,cap)
   for _ in range(2):m.apply_exl3_experts(x,ids,w,layer)
  torch.cuda.synchronize();started=time.time()
  for rep in range(5 if a.small else 9):
   order=configs.copy();random.Random(989+rep).shuffle(order)
   for tier,cap in order:
    m=select(tier,cap);torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    call0=new.exl3_fat_diag()['grouped_calls'];t=time.perf_counter();s.record();m.apply_exl3_experts(x,ids,w,layer);e.record();torch.cuda.synchronize()
    samples[str((tier,cap))].append({'cuda_ms':s.elapsed_time(e),'wall_ms':1000*(time.perf_counter()-t)})
    calls[str((tier,cap))]=new.exl3_fat_diag()['grouped_calls']-call0
    scratch[str((tier,cap))]={'current_e2_fat':old.exl3_fat_diag()['fat_scratch_bytes'],'new_e2_fat':new.exl3_fat_diag()['fat_scratch_bytes'],'grouped':new.exl3_fat_diag()['grouped_scratch_bytes']}
  row={'tokens':n,'skew':skew,'max_expert_rows':int(counts.max()),'fat_experts_128':int((counts>128).sum()),'fat_experts_64':int((counts>64).sum()),'start':started,'end':time.time(),'samples':samples,'medians':{k:{f:statistics.median(q[f] for q in v) for f in ('cuda_ms','wall_ms')} for k,v in samples.items()},'grouped_calls_per_apply':calls,'scratch_bytes':scratch}
  if n<=64:assert all(v==0 for v in calls.values()),'grouped kernel entered decode-sized step'
  rec['cases'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ('samples','scratch_bytes')}),flush=True)
  del x,ids,w;torch.cuda.empty_cache()
rec['ok']=True;rec['ended']=time.time();rec['peak_allocated']=torch.cuda.max_memory_allocated();save();phase('complete')
