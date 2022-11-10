

# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- softmax --
import torch.nn.functional as nnf

# -- cpp cuda kernel --
import dnls_cuda


def unique_topk(vals,k,dim=1):
    assert dim == 1
    vals = vals.contiguous()
    a,b = vals.shape
    device = vals.device
    args = th.zeros((a,k),device=device,dtype=th.int32)
    dnls_cuda.unique_topk(vals,args,k,dim)
    return args.type(th.int64)

class UniqueTopKFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vals, args, k, dim):
        pass

    @staticmethod
    def backward(ctx, vals, args, k, dim):
        pass


# class UniqueTopK(th.nn.Module):

#     def __init__(self, k, ps, pt, nheads,
#                  chnls=-1, dilation=1, stride0=1, stride1=1,
#                  use_k=True, use_adj=True, reflect_bounds=True,
#                  search_abs=False, nbwd=1, exact=False,
#                  h0_off=0, w0_off=0, h1_off=0, w1_off=0,
#                  remove_self=False, anchor_self=False, rbwd=True):
#         super().__init__()


