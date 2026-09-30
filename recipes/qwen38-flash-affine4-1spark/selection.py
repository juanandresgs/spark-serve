"""Independent sparse-block selection using ordered integer keys and bounded tiles.

Use the dependency's score computation unchanged. Encode the documented order
(score descending, block index ascending), select with PyTorch's integer top-k,
then restore block order. No model projection/attention arithmetic is changed.
"""
import torch

class Selector:
    def __init__(self, fallback=None):
        self.fallback = fallback

    def __getitem__(self, grid):
        def launch(scores, position, ids, counts, sparse, stride, *, RATIO, TOP, IDW, BLOCK, **kwargs):
            if self.fallback is not None and BLOCK <= 32768:
                return self.fallback[grid](scores, position, ids, counts, sparse, stride, RATIO=RATIO, TOP=TOP, IDW=IDW, BLOCK=BLOCK, **kwargs)
            rows=grid[0]
            available=min(stride,BLOCK)
            blocks=torch.arange(available,device=scores.device,dtype=torch.int64)
            within=torch.arange(RATIO,device=scores.device,dtype=torch.int32)
            # Bound temporary keys independently of prefill length.
            for first in range(0,rows,128):
                last=min(rows,first+128)
                end=position.to(torch.int64)+torch.arange(first+1,last+1,device=scores.device,dtype=torch.int64)
                full=end//RATIO
                value=scores.reshape(-1,stride)[first:last,:available].contiguous()
                bits=value.view(torch.int32).to(torch.int64)&0xffffffff
                ordered=torch.where((bits&0x80000000)!=0,(~bits)&0xffffffff,bits|0x80000000)
                keys=(ordered-0x80000000)*0x100000000+(0xffffffff-blocks)
                keys.masked_fill_(blocks[None,:]>=full[:,None],torch.iinfo(torch.int64).min)
                chosen=torch.topk(keys,min(TOP,available),dim=1,sorted=False).indices
                chosen=torch.sort(chosen,dim=1).values.to(torch.int32)
                expanded=(chosen[:,:,None]*RATIO+within).reshape(last-first,-1)
                dest=ids.reshape(-1,IDW)[first:last]
                dest[:,:expanded.shape[1]].copy_(expanded)
                tail=(full[:,None]*RATIO+within).to(torch.int32)
                dest[:,TOP*RATIO:TOP*RATIO+RATIO].copy_(tail)
                is_sparse=full>TOP
                counts[first:last].copy_(torch.where(is_sparse,TOP*RATIO+end-full*RATIO,end).to(counts.dtype))
                sparse[first:last].copy_(is_sparse.to(sparse.dtype))
        return launch

def install():
    from tensorfold.families.qwen4_exp.cuda import attention
    attention._select=Selector(attention._select)
