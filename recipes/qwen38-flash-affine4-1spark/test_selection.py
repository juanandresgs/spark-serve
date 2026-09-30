"""Actual GPU equality of independent sparse selection against pinned dependency."""
import torch,triton
from selection import Selector
from tensorfold.families.qwen4_exp.cuda.attention import _select
select=Selector()
g=torch.Generator(device='cuda');g.manual_seed(147)
for position in [0,2047,8190,131000,253900]:
 for tied in [False,True]:
  rows=17;ratio=4;top=512;idw=top*ratio+ratio;stride=65536
  scores=torch.rand((rows,stride),device='cuda',generator=g)
  if tied:scores=(scores*5).floor()
  pos=torch.tensor(position,device='cuda',dtype=torch.int32)
  output=[]
  for kernel in [_select,select]:
   ids=torch.zeros((rows,idw),device='cuda',dtype=torch.int32);n=torch.zeros(rows,device='cuda',dtype=torch.int32);s=n.clone()
   kernel[(rows,)](scores,pos,ids,n,s,stride,RATIO=ratio,TOP=top,IDW=idw,BLOCK=triton.next_power_of_2((position+rows+ratio-1)//ratio),num_warps=16)
   output.append((ids.cpu(),n.cpu(),s.cpu()))
  torch.testing.assert_close(output[0][1],output[1][1],rtol=0,atol=0);torch.testing.assert_close(output[0][2],output[1][2],rtol=0,atol=0)
  for i in range(rows):
   if output[0][2][i]:
    count=output[0][1][i]
    torch.testing.assert_close(output[0][0][i,:count],output[1][0][i,:count],rtol=0,atol=0)
print('GPU sparse-selection equality passed: ten random/tied/boundary cases')
