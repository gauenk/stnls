
# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


class ScatterNlFunction(th.autograd.Function):
    # [video -> patches] @ nlInds

    @staticmethod
    def forward(ctx, vid, nlInds, ps, pt=1):
        """
        vid = [T,C,H,W]
        nlInds = [NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        """
        patches = allocate_patches(vid,nlInds,ps,pt)
        dnls_cuda.scatter_forward(vid, nlInds, patches)
        ctx.save_for_backward([nlInds,ps,pt])
        return patches

    @staticmethod
    def backward(ctx, grad_patches):
        nlInds,ps,pt = ctx.saved_tensors
        grad_vid = allocate_vid(vid_shape,grad_patches.device)
        grad_patches = grad_patches.contiguous()
        dnls_cuda.scatter_backward(grad_patches,vid,nlInds,ps,pt)
        return grad_vid

    @staticmethod
    def allocate_vid(vid_shape,device):
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        return vid

    @staticmethod
    def allocate_patches(vid,nlInds,ps,pt):
        device = vid.device
        nq,k,c = nlInds.shape[:2],vid.shape[1]
        assert c in [1,3],"Must be the color channel."
        patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
        return patches

class ScatterNL(th.nn.Module):
    # [video -> patches] @ nlInds

    def __init__(self, ps, pt=1):
        super(ScatterNl, self).__init__()
        self.ps = ps
        self.pt = pt

    def forward(self, vid, nlInds):
        return ScatterNlFunction.apply(vid,nlInds,self.ps,self.pt)

