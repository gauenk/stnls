
# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


class GatherNlFunction(th.autograd.Function):
    # [patches -> video] @ nlInds

    @staticmethod
    def forward(ctx, vid, patches, nlInds):
        dnls_cuda.gather_forward(vid, patches, nlInds)
        ctx.save_for_backward([nlInds])
        return vid

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
        assert c in [1,3],"Must be the color channel."
        patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
        return patches

class GatherNL(th.nn.Module):
    # [patches -> video] @ nlInds

    def __init__(self, vid_shape):
        super(GatherNl, self).__init__()
        self.vid = self.allocate_vid(vid_shape,device)

    def allocate_vid(vid_shape,device):
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        return vid

    def forward(self, patches, nlInds):
        return GatherNlFunction.apply(self.vid,patches,nlInds)

