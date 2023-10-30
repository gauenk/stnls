# -- python --
import torch as th
import numpy as np
from einops import rearrange
import stnls_cuda
from ..shared import run_unfold

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Rasterize Indices
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def vid2patches(vid,nheads,stride,ps,dilation,reflect_bounds):
    # -- num search --
    B,T,C = vid.shape[:3]
    vid = rearrange(vid,'b t c h w -> (b t) c h w')
    patches = run_unfold(vid,ps,stride,dilation,reflect_bounds)
    shape_str = '(b t) (HD c ph pw) q -> (b HD) (t q) (c ph pw)'
    patches = rearrange(patches,shape_str,HD=nheads,t=T,c=C,ph=ps)
    return patches

def raster_indices(inds,iH,iW,stride):

    # -- num search --
    nH = (iH-1)//stride+1
    nW = (iW-1)//stride+1
    nHW = nH * nW
    # print("nH,nW: ",nH,nW)

    # -- rasterized --
    tI = inds[...,0].type(th.int64)
    hI = th.div(inds[...,1],stride,rounding_mode="floor").type(th.int64)
    wI = th.div(inds[...,2],stride,rounding_mode="floor").type(th.int64)
    rI = tI * nH * nW + hI * nW + wI

    # -- reshape --
    rI = rI.type(th.int64)

    return rI

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Indexing MatMult
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def matmult_fwd(x,y,I):
    # n3net uses "y" as ref and "x" as search
    b = y.shape[0]
    m = y.shape[1]
    n = x.shape[1]
    o = I.shape[2]
    e = x.shape[2]
    out = th.tensor(np.zeros(b*m*o), dtype=th.float).reshape(b,m,o).cuda()
    # out = th.tensor(np.zeros(b*n*o), dtype=th.float).reshape(b,n,o).cuda()
    # print("out.shape: ",out.shape)
    stnls_cuda.n3net_matmul1_fwd(x,y,I,out,n,m,e,o,b)
    return out

def matmult_bwd(x,y,I,grad):
    # n3net uses "x" as search and "y" as ref
    b = y.shape[0]
    m = y.shape[1]
    n = x.shape[1]
    o = I.shape[2]
    e = x.shape[2]
    grad_x = th.tensor(np.zeros(x.numel()), dtype=th.float).\
        reshape(x.shape[0],x.shape[1],x.shape[2]).cuda()
    grad_y = th.tensor(np.zeros(y.numel()), dtype=th.float).\
        reshape(y.shape[0],y.shape[1],y.shape[2]).cuda()
    stnls_cuda.n3net_matmul1_bwd(grad,x,y,I,grad_x,grad_y, m, n, e, o, b)
    return grad_x, grad_y

