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

class FoldFunction(th.autograd.Function):
    """
    [patches -> video] @ nlInds

    nlInds.shape = [NumQueries,K,3]
    patches.shape = [NumQueries,K,pt,c,ps,ps]
    """

    @staticmethod
    def forward(ctx, patches, vid, qStart, qCut, qStride, dilation):
        dnls_cuda.fold_forward(vid, patches, qStart, qStride, dilation)
        ctx.qStart = qStart
        ctx.qStride = qStride
        ctx.qNum = patches.shape[0]
        ctx.dilation = dilation
        ctx.pt = patches.shape[2]
        ctx.ps = patches.shape[5]
        ctx.qCut = qCut
        return vid

    @staticmethod
    def backward(ctx, grad_vid):

        # -- unpack --
        grad_vid = grad_vid.contiguous()
        ps,pt = ctx.ps,ctx.pt
        dilation = ctx.dilation
        qStart = ctx.qStart
        qStride = ctx.qStride
        qNum = ctx.qNum
        qCut = ctx.qCut

        # -- alloc --
        t,c,h,w  = grad_vid.shape
        npix = t*h*w
        colors = grad_vid.shape[1]
        device = grad_vid.device
        grad_patches = allocate_patches(qNum,1,ps,pt,colors,device)

        # -- clear at qcut --
        # grad_patches[:qCut] = 0.

        # -- backward --
        dnls_cuda.fold_backward(grad_vid,grad_patches,qStart,qStride,dilation)

        return grad_patches,None,None,None,None,None

class Fold(th.nn.Module):
    # [patches -> video] @ nlInds [with k == 1]

    def __init__(self, vid_shape, ps, qStride=1, dilation=1, device="cuda:0"):
        super(Fold, self).__init__()
        self.device = device
        self.vid_shape = vid_shape
        self.vid = self.allocate_vid(vid_shape,device)
        self.patch_batch_buffer = None
        self.qStride = qStride
        self.dilation = dilation

        self.curr_vidx = 0 # starting video pixel at top of "forward"
        self.ps = ps
        t,c,h,w = self.vid.shape
        self.npix = t*h*w

    def allocate_vid(self,vid_shape,device):
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        return vid

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

    def forward(self, patches, qStart):
        # bpatches,qStart,qCut = self.batched_patches(patches,qStart)
        bpatches,qStart,qCut = patches,qStart,0
        vid = self.allocate_vid(self.vid_shape,self.device)
        vid = FoldFunction.apply(bpatches, vid, qStart, qCut,
                                 self.qStride,self.dilation)
        self.vid = self.vid + vid
        return self.vid

