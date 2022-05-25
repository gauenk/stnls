
# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


class GatherNlFunction(th.autograd.Function):
    """
    [patches -> video] @ nlInds

    nlInds.shape = [NumQueries,K,3]
    """

    @staticmethod
    def forward(ctx, vid, wvid, patches, nlDists, nlInds, lam, dilation):
        dnls_cuda.gather_forward(vid, wvid, patches, nlDists, nlInds, lam, dilation)
        ctx.save_for_backward([nlInds])
        return vid,wvid

    @staticmethod
    def backward(ctx, grad_vid):
        nlInds = ctx.saved_tensors
        grad_vid = grad_vid.contiguous()
        patches = allocate_patches(nlInds,ps,pt)
        dnls_cuda.gather_backward(grad_vid,patches,nlInds)
        return patches

    @staticmethod
    def allocate_patches(nlInds,ps,pt):
        device = nlInds.device
        nq,k,c = nlInds.shape[:2],vid.shape[1]
        patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
        return patches

class GatherNL(th.nn.Module):
    # [patches -> video] @ nlInds

    def __init__(self, vid_shape, lam=1., dilation=1, device="cuda:0"):
        super(GatherNL, self).__init__()
        self.vid,self.wvid = self.allocate_vid(vid_shape,device)
        self.lam = lam
        self.dilation = dilation

    def allocate_vid(self,vid_shape,device):
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        wvid = th.zeros(vid_shape,device=device,dtype=th.float32)
        return vid,wvid

    def forward(self, patches, nlDists, nlInds, shape=None):
        return GatherNlFunction.apply(self.vid,self.wvid,
                                      patches,nlDists,nlInds,
                                      self.lam,self.dilation)

