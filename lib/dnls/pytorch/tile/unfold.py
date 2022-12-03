"""

A gather _without_ a race condition with k == 1

"""


# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(nq,k,ps,pt,c,device):
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class unfold(th.autograd.Function):
    """
    [patches -> video] @ nlInds

    nlInds.shape = [NumQueries,K,3]
    patches.shape = [NumQueries,K,pt,c,ps,ps]
    """

    @staticmethod
    def forward(ctx, patches, vid, qStart, stride, dilation):

        # -- allocate --
        colors = vid.shape[-3]
        device = vid.device

        # -- forward --
        dnls_cuda.unfold_forward(vid, patches, qStart, stride, dilation)

        # -- store --
        ctx.qStart = qStart
        ctx.dilation = dilation
        ctx.vid_shape = vid.shape
        ctx.stride = stride

        return patches

    @staticmethod
    def backward(ctx, grad_patches):

        # -- fmt --
        grad_patches = grad_patches.contiguous()
        qStart = ctx.qStart
        stride = ctx.stride
        dilation = ctx.dilation
        vid_shape = ctx.vid_shape

        # -- allocate --
        grad_vid = allocate_vid(vid_shape,grad_patches.device)

        # -- forward --
        dnls_cuda.unfold_backward(grad_vid,grad_patches,qStart,stride,dilation)

        return None,grad_vid,None,None,None,None

class Unfold(th.nn.Module):
    # [patches -> video] @ nlInds [with k == 1]

    def __init__(self, ps, pt=1, stride=1, dilation=1, device="cuda:0"):
        super(Unfold, self).__init__()
        self.ps = ps
        self.pt = pt
        self.stride = stride
        self.dilation = dilation
        self.patches = th.empty(0).to(device)
        self.device = device

    # def update_patches(patches):
    #     self.patches = th.stack([self.patches,patches],0)

    def forward(self, vid, qStart, qNum):
        colors = vid.shape[1]
        patches = allocate_patches(qNum,1,self.ps,self.pt,colors,self.device)
        patches = unfold.apply(patches,vid,qStart,self.stride,self.dilation)
        return patches


