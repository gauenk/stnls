"""

A gather _without_ a race condition with k == 1

"""


# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


def allocate_patches(nq,k,ps,pt,c,device):
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class fold(th.autograd.Function):
    """
    [patches -> video] @ nlInds

    nlInds.shape = [NumQueries,K,3]
    patches.shape = [NumQueries,K,pt,c,ps,ps]
    """

    @staticmethod
    def forward(ctx, patches, vid, qStart, stride, dilation):
        dnls_cuda.fold_forward(vid, patches, qStart, stride, dilation)
        ctx.qStart = qStart
        ctx.stride = stride
        ctx.qNum = patches.shape[0]
        ctx.dilation = dilation
        ctx.pt = patches.shape[2]
        ctx.ps = patches.shape[5]
        return vid

    @staticmethod
    def backward(ctx, grad_vid):

        # -- unpack --
        grad_vid = grad_vid.contiguous()
        ps,pt = ctx.ps,ctx.pt
        dilation = ctx.dilation
        qStart = ctx.qStart
        stride = ctx.stride
        qNum = ctx.qNum

        # -- alloc --
        t,c,h,w  = grad_vid.shape
        npix = t*h*w
        colors = grad_vid.shape[1]
        device = grad_vid.device
        grad_patches = allocate_patches(qNum,1,ps,pt,colors,device)

        # -- backward --
        dnls_cuda.fold_backward(grad_vid,grad_patches,qStart,stride,dilation)

        return grad_patches,None,None,None,None

class Fold(th.nn.Module):
    # [patches -> video] @ nlInds [with k == 1]

    def __init__(self, vid_shape, stride=1, dilation=1, device="cuda:0"):
        super(Fold, self).__init__()
        self.device = device
        self.vid_shape = vid_shape
        self.vid = self.allocate_vid(vid_shape,device)
        self.stride = stride
        self.dilation = dilation

    def allocate_vid(self,vid_shape,device):
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        return vid

    def forward(self, patches, qStart):
        bpatches,qStart = patches,qStart
        vid = self.allocate_vid(self.vid_shape,self.device)
        vid = fold.apply(bpatches, vid, qStart,
                         self.stride,self.dilation)
        self.vid = self.vid + vid
        return self.vid

