
# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(inds,ps,pt,c):
    device = inds.device
    nq,k = inds.shape[0],1
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches


class WpSumFunction(th.autograd.Function):
    # [video -> patches] @ inds

    @staticmethod
    def forward(ctx, vid, dists, inds, ps, pt=1, dilation=1,
                adj = 0, reflect_bounds=True, exact=False):
        """
        vid = [T,C,H,W]
        inds = [NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        """
        patches = allocate_patches(inds,ps,pt,vid.shape[1])
        print("vid.shape: ",vid.shape)
        print("patches.shape: ",patches.shape)
        print("dists.shape: ",dists.shape)
        print("inds.shape: ",inds.shape)
        print("dilation,adj,reflect_bounds: ",dilation,adj,reflect_bounds)
        dnls_cuda.wpsum_forward(vid, patches, dists, inds, dilation, adj, reflect_bounds)
        ctx.save_for_backward(dists,inds)
        ctx.ps,ctx.pt = ps,pt
        ctx.vid_shape = vid.shape
        ctx.dilation = dilation
        ctx.exact = exact
        ctx.adj = adj
        ctx.reflect_bounds = reflect_bounds
        return patches

    @staticmethod
    def backward(ctx, grad_patches):
        dists,inds = ctx.saved_tensors
        ps,pt = ctx.ps,ctx.pt
        vid_shape = ctx.vid_shape
        dilation = ctx.dilation
        exact = ctx.exact
        adj = ctx.adj
        reflect_bounds = ctx.reflect_bounds
        grad_vid = allocate_vid(vid_shape,grad_patches.device)
        grad_patches = grad_patches#.contiguous()
        print(grad_vid.shape,grad_patches.shape)
        dnls_cuda.wpsum_backward(grad_vid,grad_patches,dists,inds,
                                 dilation,adj,reflect_bounds,exact)
        return grad_vid,None,None,None,None,None,None,None,None

class WeightedPatchSum(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, pt=1, dilation=1, adj=0, reflect_bounds = True, exact=False):
        super(WeightedPatchSum, self).__init__()
        self.ps = ps
        self.pt = pt
        self.dilation = dilation
        self.exact = exact
        self.adj = adj
        self.reflect_bounds = reflect_bounds

    def forward(self, vid, dists, inds):
        patches = WpSumFunction.apply(vid,dists,inds,self.ps,self.pt,
                                   self.dilation, self.adj,
                                   self.reflect_bounds, self.exact)
        nq,_,_,c,ph,pw = patches.shape
        patches = patches.view(nq,c,ph,pw)
        return patches

