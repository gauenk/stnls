"""

  Compute the destination indices for NonLocalScatter
  to remove the race condition of center
  and mitigate the race condition of the patch

 flows_k MUST be output from a grid search of size (2*wt+1,ws,ws)
 offset from flows. This is not valid for aribtrary values of flows_k

"""


# -- imports --
import torch as th
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

def run(tensor,flows_k,labels,stride0,stride1,H,W,invalid=th.inf):

    # -- unpack shapes --
    B,HD,T,nH1,nW1,K = tensor.shape[:6]
    _Q1 = T*nH1*nW1
    # print(_Q1,T,nH1,nW1)
    S = labels.max().long().item()+1
    tensor = tensor.reshape(B,HD,_Q1,K,-1)
    M = tensor.shape[-1]
    nH0 = (H-1)//stride0+1
    nW0 = (W-1)//stride0+1
    nH1 = (H-1)//stride1+1
    nW1 = (W-1)//stride1+1
    Q0,Q1 = T*nH0*nW0,T*nH1*nW1
    assert Q1 == _Q1,"Matching num queries."

    # -- change type if needed --
    dtype = tensor.dtype
    if dtype in [th.int32,th.int64]:
        tensor = tensor.float()

    # -- prepare --
    shape = (B,HD,Q0,S,M)
    gather_tensor = invalid*th.ones(shape,device=labels.device,dtype=tensor.dtype)
    stnls_cuda.gather_tensor_forward(gather_tensor,tensor,labels,flows_k,
                                     stride0,stride1,H,W)

    # -- adjust output type --
    if dtype in [th.int32,th.int64]:
        gather_tensor = gather_tensor.type(dtype)

    # -- squeeze single M --
    if M == 1:
        gather_tensor = gather_tensor[...,0]

    exit()
    return gather_tensor

