"""

Get temporal inds from spatial inds

"""

import torch as th
import dnls_cuda

def run(inds,flows,wt):


    # -- shaping --
    B,Q,K,_ = inds.shape
    T = flows.fflow.shape[-4]
    st = 2*wt
    nT = min(st,T-1)

    # -- allocate --
    inds_t = -th.ones((B,Q,K,nT,3),device=inds.device,dtype=inds.dtype)

    # -- run --
    fflow = flows.fflow
    bflow = flows.bflow
    dnls_cuda.temporal_inds(inds,fflow,bflow,inds_t)

    return inds_t
