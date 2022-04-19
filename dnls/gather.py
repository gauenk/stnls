
# -- python --
import math
import torch as th
import torch.nn as nn

# -- cpp cuda kernel --
import dnls_cuda


class GatherNlFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid2fill, patches, queryInds, ps, pt):
        outputs = dnls_cuda.gather_forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        ctx.save_for_backward([ps,pt])
        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        ps,pt = ctx.saved_tensors
        outputs = dnls_cuda.gather_backward(grad_h.contiguous(),
                                             grad_cell.contiguous(),
                                             ps,pt)
        return

class GatherNL(nn.Module):
    # [patches -> video] @ queryInds

    def __init__(self, ps, pt):
        super(GatherNl, self).__init__()
        self.ps = ps
        self.pt = pt

    def forward(self, vid2fill, patches, queryInds):
        return GatherNlFunction.apply(vid2fill,patches,queryInds,self.ps,self.pt)

