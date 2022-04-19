
# -- python --
import math
import torch as th
import torch.nn as nn

# -- cpp cuda kernel --
import dnls_cuda


class ScatterNlFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid2fill, patches, queryInds, ps, pt):
        outputs = dnls_cuda.scatter_forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        ctx.save_for_backward([ps,pt])
        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        ps,pt = ctx.saved_tensors
        outputs = dnls_cuda.scatter_backward(grad_h.contiguous(),
                                             grad_cell.contiguous(),
                                             ps,pt)
        return

class ScatterNL(nn.Module):
    # [video -> patches] @ queryInds

    def __init__(self, ps, pt):
        super(ScatterNl, self).__init__()
        self.ps = ps
        self.pt = pt

    def forward(self, vid2fill, patches, queryInds):
        return ScatterNlFunction.apply(vid2fill,patches,queryInds,self.ps,self.pt)

