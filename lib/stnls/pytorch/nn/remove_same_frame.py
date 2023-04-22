"""
Remove all indices that match the index 0 frame.

"""

import torch as th
import stnls_cuda
from .dim3_utils import dimN_dim3,dim3_dimN
from einops import rearrange

def run(dists,inds):

    # -- view --
    dists,inds,dshape,ishape = dimN_dim3(dists,inds)
    dshape = list(dshape)
    ishape = list(ishape)
    dshape[-1] = -1
    ishape[-2] = -1

    # -- view --
    B = dists.shape[0]
    dists = rearrange(dists,'b q k -> (b q) k')
    inds = rearrange(inds,'b q k tr -> (b q) k tr')
    dists0 = dists[:,:1]
    inds0 = inds[:,:1]
    # print("dists.shape,inds.shape: ",dists.shape,inds.shape)
    args = th.where(inds[:,:1,0] != inds[:,1:,0])
    # print("args.shape: ",[a.shape for a in args])
    # exit(0)

    # -- inexing --
    dists1 = dists[:,1:]
    inds1 = inds[:,1:]
    K = inds1.shape[1]
    BQ = inds.shape[0]

    # -- not same frame --
    _inds = []
    for i in range(3):
        inds1_i = inds1[...,i][args]
        inds1_i = rearrange(inds1_i,'(bq k) -> bq k',bq=BQ)
        _inds.append(inds1_i)
    inds = th.cat([inds0,th.stack(_inds,-1)],1)
    # print("inds.shape: ",inds.shape)
    dists = rearrange(dists1[args],'(bq k) -> bq k',bq=BQ)
    dists = th.cat([dists0,dists],1)
    # print("dists.shape: ",dists.shape)

    # th.cuda.synchronize()
    # print(dists[100,10])
    # print(inds[100,10])

    # -- view --
    dists = rearrange(dists,'(b q) k -> b q k',b=B)
    inds = rearrange(inds,'(b q) k tr -> b q k tr',b=B)
    # print("dists.shape,inds.shape: ",dists.shape,inds.shape)

    dists,inds = dim3_dimN(dists,inds,dshape,ishape)
    return dists,inds
