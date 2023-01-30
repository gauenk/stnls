"""

Get temporal inds from spatial inds

"""

import torch as th
import dnls_cuda

def run(inds,wt,fflow,bflow):

    # -- shaping --
    if inds.ndim == 4: inds = inds.unsqueeze(1)
    inds = inds.contiguous()
    B,HD,Q,K,_ = inds.shape
    B,T,_,H,W = fflow.shape
    st = 2*wt
    nT = min(st,T-1)

    # -- allocate --
    inds_t = th.ones((B,HD,Q,K,nT,3),device=inds.device,dtype=inds.dtype)
    inds_t[...] = -1

    # -- run --
    dnls_cuda.temporal_inds(inds,fflow,bflow,inds_t)
    th.cuda.synchronize()

    return inds_t
