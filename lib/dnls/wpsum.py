
# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda

from dnls.utils.timer import ExpTimer


def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(inds,ps,pt,c):
    device = inds.device
    nq,k = inds.shape[0],1
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

def allocate_iunfold_patches(nq,k,ps,pt,c,device):
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class WpSumFunction(th.autograd.Function):
    # [video -> patches] @ inds

    # -- static video since it is the same --
    # vid = None

    @staticmethod
    def forward(ctx, vid, dists, inds, ps, pt=1,
                h_off=0,w_off=0,dilation=1,adj=0,reflect_bounds=True,exact=False):
        """
        vid = [T,C,H,W]
        inds = [NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        """
        # if WpSumFunction.vid is None: WpSumFunction.vid = vid
        patches = allocate_patches(inds,ps,pt,vid.shape[1])
        dnls_cuda.wpsum_forward(vid, patches, dists, inds,
                                h_off,w_off,dilation,adj,reflect_bounds)
        # print("dists._version: ",dists._version)
        # print("inds._version: ",inds._version)
        ctx.save_for_backward(dists,inds,vid)
        ctx.ps,ctx.pt = ps,pt
        ctx.vid_shape = vid.shape
        ctx.dilation = dilation
        ctx.h_off = h_off
        ctx.w_off = w_off
        ctx.adj = adj
        ctx.reflect_bounds = reflect_bounds
        ctx.exact = exact

        return patches

    @staticmethod
    def backward(ctx, grad_patches):
        dists,inds,vid = ctx.saved_tensors
        # vid = WpSumFunction.vid
        ps,pt = ctx.ps,ctx.pt
        vid_shape = ctx.vid_shape

        h_off = ctx.h_off
        w_off = ctx.w_off
        dilation = ctx.dilation
        adj = ctx.adj
        reflect_bounds = ctx.reflect_bounds
        exact = ctx.exact

        # -- start timer --
        timer = ExpTimer()
        timer.start("wpsum_bwd")

        # -- gradient for video --
        grad_vid = allocate_vid(vid_shape,grad_patches.device)
        dnls_cuda.wpsum_backward_vid(grad_vid,grad_patches,dists,inds,
                                     h_off,w_off,dilation,adj,reflect_bounds,exact)

        # -- gradient for dists --
        grad_dists = th.zeros_like(dists)
        dnls_cuda.wpsum_backward_dists(grad_dists,grad_patches,vid,inds,
                                       h_off,w_off,dilation,adj,reflect_bounds,exact)

        # -- stop timer --
        th.cuda.synchronize()
        timer.stop("wpsum_bwd")
        # print(timer)

        return grad_vid,grad_dists,None,None,None,None,None,None,None,None,None

class WeightedPatchSum(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, pt=1, h_off=0, w_off=0, dilation=1,
                 adj=0, reflect_bounds = True, exact=False):
        super(WeightedPatchSum, self).__init__()
        self.ps = ps
        self.pt = pt

        self.h_off = h_off
        self.w_off = w_off

        self.dilation = dilation
        self.adj = adj
        self.reflect_bounds = reflect_bounds
        self.exact = exact


    def forward(self, vid, dists, inds):
        patches = WpSumFunction.apply(vid,dists,inds,self.ps,self.pt,
                                      self.h_off,self.w_off,
                                      self.dilation,self.adj,
                                      self.reflect_bounds, self.exact)
        nq,_,_,c,ph,pw = patches.shape
        patches = patches.view(nq,c,ph,pw)
        return patches

