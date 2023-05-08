# -- python --
import torch as th
import numpy as np
from einops import rearrange
import stnls
import stnls_cuda

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Rasterize Indices
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def vid2patches(vid,nheads,stride,ps,pt,dilation,reflect_bounds):

    # -- num search --
    border = "reflect" if reflect_bounds else "zero"
    unfold = stnls.iUnfold(ps, pt=pt, stride=stride,
                           dilation=dilation, border=border)
    patches = unfold(vid)
    shape_str = 'b q 1 pt (c HD) ph pw -> (b HD) q (pt c ph pw)'
    patches = rearrange(patches,shape_str,HD=nheads)

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
    b = y.shape[0]
    m = y.shape[1]
    n = x.shape[1]
    o = I.shape[2]
    e = x.shape[2]
    out = th.tensor(np.zeros(b*m*o), dtype=th.float).reshape(b,m,o).cuda()
    stnls_cuda.n3net_matmul1_fwd(x,y,I,out,n,m,e,o,b)
    return out

def matmult_bwd(x,y,I,grad):
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
    
# class IndexedMatmul1Efficient(th.autograd.Function):
#     """

#     Exec N3Net Matmult.
    
#     """

#     @staticmethod
#     def forward(ctx, x, y, I):
#         ctx.save_for_backward(x, y, I)
#         return out

#     @staticmethod
#     def backward(ctx, grad):
#         x, y, I = ctx.saved_tensors
#         b = y.shape[0]
#         m = y.shape[1]
#         n = x.shape[1]
#         o = I.shape[2]
#         e = x.shape[2]
#         grad_x = th.tensor(np.zeros(x.numel()), dtype=th.float).\
#             reshape(x.shape[0],x.shape[1],x.shape[2]).cuda()
#         grad_y = th.tensor(np.zeros(y.numel()), dtype=th.float).\
#             reshape(y.shape[0],y.shape[1],y.shape[2]).cuda()
#         stnls_cuda.n3net_matmul1_bwd(grad,x,y,I,grad_x,grad_y, m, n, e, o, b)
#         return grad_x, grad_y, I

