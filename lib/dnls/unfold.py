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

class UnfoldFunction(th.autograd.Function):
    """
    [patches -> video] @ nlInds

    nlInds.shape = [NumQueries,K,3]
    patches.shape = [NumQueries,K,pt,c,ps,ps]
    """

    @staticmethod
    def forward(ctx, patches, vid, qStart, qStride, qCut, dilation):

        # -- allocate --
        colors = vid.shape[1]
        device = vid.device

        # -- forward --
        dnls_cuda.unfold_forward(vid, patches, qStart, qStride, dilation)

        # -- store --
        ctx.qStart = qStart
        ctx.dilation = dilation
        ctx.vid_shape = vid.shape
        ctx.qStride = qStride
        ctx.qCut = qCut

        return patches

    @staticmethod
    def backward(ctx, grad_patches):

        # -- fmt --
        grad_patches = grad_patches.contiguous()
        qStart = ctx.qStart
        qStride = ctx.qStride
        dilation = ctx.dilation
        vid_shape = ctx.vid_shape

        # -- allocate --
        grad_vid = allocate_vid(vid_shape,grad_patches.device)

        # -- forward --
        dnls_cuda.unfold_backward(grad_vid,grad_patches,qStart,qStride,dilation)

        return grad_vid,None,None,None,None

class Unfold(th.nn.Module):
    # [patches -> video] @ nlInds [with k == 1]

    def __init__(self, ps, pt=1, qStride=1, dilation=1, device="cuda:0"):
        super(Unfold, self).__init__()
        self.ps = ps
        self.pt = pt
        self.qStride = qStride
        self.dilation = dilation
        self.patches = th.empty(0).to(device)


    def update_buffer(self,patches):
        # -- skip no bacthing --
        if patches.shape[0] == self.npix: return

        # -- compute buffer size --
        bs = patches.shape[0]
        padf = self.dilation*(self.ps-1)
        t,c,h,w = self.vid.shape
        buf_size = w*padf

        # -- curr vidx --
        ti = self.curr_vidx // (h*w)
        hi = (self.curr_vidx // w) % h
        wi = self.curr_vidx % w
        if ti > 0 and hi == 0 and wi == 0:
            buf_size = 0

        # -- assign --
        if buf_size > 0:
            self.patch_batch_buffer = patches[-buf_size:].detach()
        else:
            self.patch_batch_buffer = None

    def batched_patches(self,patches,qStart):
        self.curr_vidx += patches.shape[0]
        if patches.shape[0] == self.npix:
            return patches,qStart,0
        elif self.patch_batch_buffer is None:
            self.update_buffer(patches)
            return patches,qStart,0
        else:
            b_patches = th.cat([self.patch_batch_buffer,patches])
            qCut = len(self.patch_batch_buffer)
            qStart -= qCut
            assert qStart >= 0
            self.update_buffer(patches)
            return b_patches,qStart,qCut

    def update_patches(patches):
        self.patches = th.stack([self.patches,patches],0)

    def forward(self, vid, qStart, qNum):

        patches = allocate_patches(qNum,1,ps,pt,colors,device)
        qCut = 0
        UnfoldFunction.apply(patches,vid,qStart,self.qStride,
                             qCut,self.dilation)
        self.update_patches(patches)
        return patches


