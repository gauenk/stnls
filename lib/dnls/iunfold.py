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

class iUnfoldFunction(th.autograd.Function):
    """
    [patches -> video] @ nlInds

    nlInds.shape = [NumQueries,K,3]
    patches.shape = [NumQueries,K,pt,c,ps,ps]
    """

    @staticmethod
    def forward(ctx, patches, vid, start, coords, stride, dilation):

        # -- unpack --
        top,left = coords[:2]
        btm,right = coords[2:]

        # -- allocate --
        colors = vid.shape[1]
        device = vid.device

        # -- forward --
        dnls_cuda.iunfold_forward(vid, patches,
                                  top,left,btm,right,
                                  start, stride, dilation)

        # -- store --
        ctx.start = start
        ctx.coords = coords
        ctx.dilation = dilation
        ctx.vid_shape = vid.shape
        ctx.stride = stride

        return patches

    @staticmethod
    def backward(ctx, grad_patches):

        # -- fmt --
        grad_patches = grad_patches.contiguous()
        start = ctx.start
        stride = ctx.stride
        dilation = ctx.dilation
        vid_shape = ctx.vid_shape
        coords = ctx.coords

        # -- unpack --
        top,left = coords[:2]
        btm,right = coords[2:]

        # -- allocate --
        grad_vid = allocate_vid(vid_shape,grad_patches.device)

        # -- forward --
        dnls_cuda.iunfold_backward(grad_vid,grad_patches,
                                   top,left,btm,right,
                                   start,stride,dilation)

        return None,grad_vid,None,None,None,None,None

class iUnfold(th.nn.Module):
    # [patches -> video] @ nlInds [with k == 1]

    def __init__(self, ps, coords, pt=1, stride=1, dilation=1, device="cuda:0"):
        super(iUnfold, self).__init__()
        self.ps = ps
        self.pt = pt
        self.stride = stride
        self.dilation = dilation
        self.patches = th.empty(0).to(device)
        self.device = device
        self.coords = coords
        assert not(self.coords is None)

    # def update_patches(patches):
    #     self.patches = th.stack([self.patches,patches],0)

    def forward(self, vid, start, qNum):
        colors = vid.shape[1]
        patches = allocate_patches(qNum,1,self.ps,self.pt,colors,self.device)
        patches = iUnfoldFunction.apply(patches,vid,start,self.coords,
                                        self.stride,self.dilation)
        return patches


