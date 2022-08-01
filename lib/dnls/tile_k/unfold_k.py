
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
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches


class unfold_k(th.autograd.Function):
    # [video -> patches] @ nlInds

    @staticmethod
    def forward(ctx, vid, nlInds, ps, pt=1, dilation=1, btype="default", exact=False,
                adj = 0, reflect_bounds=True):
        """
        vid = [T,C,H,W]
        nlInds = [NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        """
        patches = allocate_patches(nlInds,ps,pt,vid.shape[1])
        dnls_cuda.scatter_forward(vid, patches, nlInds, dilation, adj, reflect_bounds)
        # print("nlInds.shape: ",nlInds.shape)
        ctx.save_for_backward(nlInds)
        ctx.ps,ctx.pt = ps,pt
        ctx.vid_shape = vid.shape
        ctx.dilation = dilation
        ctx.exact = exact
        ctx.btype = btype
        ctx.adj = adj
        ctx.reflect_bounds = reflect_bounds
        return patches

    @staticmethod
    def backward(ctx, grad_patches):
        nlInds = ctx.saved_tensors[0]
        ps,pt = ctx.ps,ctx.pt
        vid_shape = ctx.vid_shape
        dilation = ctx.dilation
        exact = ctx.exact
        btype = ctx.btype
        adj = ctx.adj
        reflect_bounds = ctx.reflect_bounds
        grad_vid = allocate_vid(vid_shape,grad_patches.device)
        grad_patches = grad_patches.contiguous()
        if btype in "default" or btype in "simple":
            dnls_cuda.scatter_backward(grad_vid,grad_patches,nlInds,
                                              dilation,exact,adj,reflect_bounds)
        elif btype in "efficient":
            dnls_cuda.scatter_backward_eff(grad_vid,grad_patches,nlInds,
                                           dilation,exact,adj,reflect_bounds)
        else:
            raise ValueError(f"Uknown backward type for scatter [{btype}]")
        return grad_vid,None,None,None,None,None,None,None,None

class UnfoldK(th.nn.Module):
    # [video -> patches] @ nlInds

    def __init__(self, ps, pt=1, dilation=1, btype="default", exact=False,
                 adj=0, reflect_bounds = True, device="cuda:0"):
        super(UnfoldK, self).__init__()
        self.ps = ps
        self.pt = pt
        self.dilation = dilation
        self.exact = exact
        self.btype = btype
        self.adj = adj
        self.reflect_bounds = reflect_bounds
        self.device = device

    def forward(self, vid, nlInds):
        return unfold_k.apply(vid,nlInds,self.ps,self.pt,
                              self.dilation,self.btype,self.exact,
                              self.adj, self.reflect_bounds)

