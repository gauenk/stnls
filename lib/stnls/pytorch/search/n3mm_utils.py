# -- python --
import torch as th
import numpy as np
from einops import rearrange


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Creating Indices
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_search_indices(ws,wt,fflow,bflow):

    

    inds = stnls.nn.temporal_inds(inds,wt,fflow,bflow)
    return inds

index_neighbours_cache = {}
def index_neighbours(b, n1, n2, m1, m2, s, dev, exclude_self=True):
    r"""
    This function generates the indexing tensors that define neighborhoods for each query patch
    It selects a neighborhood of s x s patches around each patch.
    Index tensors get cached in order to speed up execution time. This might lead to
    memory problems, though.
    """
    o = s**2
    if exclude_self:
        o-=1
    n = n1*n2
    m = m1*m2

    assert(m==n)

    key = "{}_{}_{}_{}_{}_{}_{}".format(n1,n2,m1,m2,s,exclude_self,dev)
    if not key in index_neighbours_cache:
        I = th.empty(1,m1*m2,o, device=dev, dtype=th.int64)

        ih = th.tensor(range(s), device=dev, dtype=th.int64).view(1,1,s,1)
        iw = th.tensor(range(s), device=dev, dtype=th.int64).view(1,1,1,s)*n2

        i = th.tensor(range(m1), device=dev, dtype=th.int64).view(m1,1,1,1)
        j = th.tensor(range(m2), device=dev, dtype=th.int64).view(1,m2,1,1)

        ch = (i-s//2).clamp(0,n1-s)
        cw = (j-s//2).clamp(0,n2-s)

        cidx = ch*n2+cw
        midx = (i*m2+j).view(m1,m2,1)

        mI = cidx + ih + iw
        mI = mI.view(m1,m2,-1)
        mI = mI[mI!=midx].view(m1*m2,-1)
        I[0,:,:] = mI

        index_neighbours_cache[key] = I

    I = index_neighbours_cache[key]
    I = I.repeat(b,1,1)
    return Variable(I, requires_grad=False)


vid_index_neighbours_cache = {}
def vid_index_neighbours(b,t,n1,n2,m1,m2,s,dev,exclude_self=True):

    # -- create vars --
    o = s**2
    if exclude_self:
        o-=1
    n = n1*n2
    m = m1*m2
    assert(m==n)

    key = "{}_{}_{}_{}_{}_{}_{}_{}".format(t,n1,n2,m1,m2,s,exclude_self,dev)
    if not key in vid_index_neighbours_cache:
        I = []
        for ti in range(t):
            It = index_neighbours(1, n1, n2, m1, m2, s, dev, exclude_self=True)
            I.append(It*n*(ti+1))
        I = th.cat(I,1)
        index_neighbours_cache[key] = I
    I = index_neighbours_cache[key]
    I = I.repeat(b,1,1)
    return Variable(I, requires_grad=False)

def vid_to_raster_inds(inds,iH,iW,stride,dev):

    # -- num search --
    nH = (iH-1)//stride+1
    nW = (iW-1)//stride+1
    nHW = nH * nW
    # print("nH,nW: ",nH,nW)

    # -- rasterized --
    tI = inds[...,0]
    hI = th.div(inds[...,1],stride,rounding_mode="floor")
    wI = th.div(inds[...,2],stride,rounding_mode="floor")
    rI = tI * nH * nW + hI * nW + wI
    # print("inds.shape: ",inds.shape)
    # print(tI.shape,stride)
    # print(tI[0,:10],inds[0,:10,0])
    # print(hI[0,:10],inds[0,:10,1])
    # print(wI[0,:10],inds[0,:10,2])
    # print(rI[0,:10])
    # exit(0)


    # -- reshape --
    rI = rI[None,:].contiguous()
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

