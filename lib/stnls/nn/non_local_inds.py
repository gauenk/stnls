"""

Get temporal inds from spatial inds

"""

import torch as th
import stnls_cuda

def run(fflow,bflow,ws,wt,stride0,stride1,full_ws=True):

    # -- unpack --
    B,T,_,H,W = fflow.shape
    nH = (H-1)//stride0 + 1
    nW = (W-1)//stride0 + 1
    Q = T*nH*nW
    St = min(2*wt+1,T)
    stride1 = float(stride1)

    # -- allocate --
    inds = -th.ones((B,Q,St,ws,ws,3),device=fflow.device,dtype=th.float32)

    # -- run --
    stnls_cuda.non_local_inds(inds,fflow,bflow,ws,wt,
                              stride0,stride1,full_ws)

    return inds
