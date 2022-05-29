
# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda



def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(nlInds,ps,pt,c):
    device = nlInds.device
    nq,k = nlInds.shape[:2]
    assert c in [1,3],"Must be the color channel."
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches


class ScatterNlFunction(th.autograd.Function):
    # [video -> patches] @ nlInds

    @staticmethod
    def forward(ctx, vid, nlInds, ps, pt=1, dilation=1, exact=False):
        """
        vid = [T,C,H,W]
        nlInds = [NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        """
        patches = allocate_patches(nlInds,ps,pt,vid.shape[1])
        dnls_cuda.scatter_forward(vid, patches, nlInds, dilation)
        ctx.save_for_backward(nlInds)
        ctx.ps,ctx.pt = ps,pt
        ctx.vid_shape = vid.shape
        ctx.dilation = dilation
        ctx.exact = exact
        return patches

    @staticmethod
    def backward(ctx, grad_patches):
        nlInds = ctx.saved_tensors[0]
        ones = th.ones_like(nlInds[:,:,0]).type(th.float32)
        ps,pt = ctx.ps,ctx.pt
        vid_shape = ctx.vid_shape
        dilation = ctx.dilation
        exact = ctx.exact
        grad_vid = allocate_vid(vid_shape,grad_patches.device)
        grad_patches = grad_patches.contiguous()
        dnls_cuda.scatter_backward(grad_patches,grad_vid,ones,nlInds,
                                   dilation,0.,exact)
        return grad_vid,None,None,None,None,None,None,None

class ScatterNl(th.nn.Module):
    # [video -> patches] @ nlInds

    def __init__(self, ps, pt=1, dilation=1, exact=False):
        super(ScatterNl, self).__init__()
        self.ps = ps
        self.pt = pt
        self.dilation = dilation
        self.exact = exact

    def forward(self, vid, nlInds):
        return ScatterNlFunction.apply(vid,nlInds,self.ps,self.pt,
                                       self.dilation,self.exact)

