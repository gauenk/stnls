
# -- python --
import torch as th
from einops import rearrange

# -- cpp cuda kernel --
import dnls_cuda

# -- misc --
from ...utils.timer import ExpTimer


def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(nq,k,ps,pt,c,device):
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class WpSumHeadsFunction(th.autograd.Function):
    # [video -> patches] @ inds

    # -- static video since it is the same --
    # vid = None

    @staticmethod
    def forward(ctx, vid, dists, inds, ps, pt=1,
                h_off=0,w_off=0,dilation=1,adj=0,reflect_bounds=True,exact=False):
        """
        vid = [nHeads or 1,T,C,H,W]
        dists = [nHeads,NumQueries,K]
        inds = [nHeads or 1,NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        """
        # -- add head dim if 1 --
        nheads = dists.shape[0]
        if vid.ndim == 4:
            vid = rearrange(vid,'t (H c) h w -> H t c h w',H=nheads)
        if inds.ndim == 3: inds = inds[None,:]

        # if WpSumFunction.vid is None: WpSumFunction.vid = vid
        device = dists.device
        nheads,nq,k = dists.shape
        patches = allocate_patches(nq,nheads,ps,pt,vid.shape[2],device)
        vid = vid.contiguous()

        # print(vid.shape)
        # print(patches.shape)
        # print(dists.shape)
        # print(inds.shape)
        # print("-"*10)

        # void cuda_wpsum_heads_forward(
        #     torch::Tensor vid, torch::Tensor patches,
        #     torch::Tensor dists, torch::Tensor inds,
        #     int h_off, int w_off, int dilation, int adj, bool reflect_bounds){
        # print(h_off,w_off,dilation,adj,reflect_bounds)
        dnls_cuda.wpsum_heads_forward(vid, patches, dists, inds,
                                      h_off,w_off,dilation,adj,
                                      reflect_bounds)
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
        # print("wpsum_heads: bwd.")

        # -- start timer --
        # timer = ExpTimer()
        # timer.start("wpsum_heads_bwd")
        # print(grad_patches.shape)

        # -- gradient for video --
        # print(vid_shape,inds.shape,dists.shape,vid.shape)
        grad_vid = allocate_vid(vid_shape,grad_patches.device)
        dnls_cuda.wpsum_heads_backward_vid(grad_vid,grad_patches,
                                           dists,inds,
                                           h_off,w_off,dilation,adj,
                                           reflect_bounds,exact)

        # -- gradient for dists --
        grad_dists = th.zeros_like(dists)
        dnls_cuda.wpsum_heads_backward_dists(grad_dists,grad_patches,
                                             vid,inds,
                                             h_off,w_off,dilation,adj,
                                             reflect_bounds,exact)

        # -- final shaping --
        grad_vid = rearrange(grad_vid,'H t c h w -> t (H c) h w')

        # -- stop timer --
        # th.cuda.synchronize()
        # timer.stop("wpsum_bwd")
        # print(timer)

        return grad_vid,grad_dists,None,None,None,None,None,None,None,None,None

class WeightedPatchSumHeads(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, pt=1, h_off=0, w_off=0, dilation=1,
                 adj=0, reflect_bounds = True, exact=False):
        super(WeightedPatchSumHeads, self).__init__()
        self.ps = ps
        self.pt = pt

        self.h_off = h_off
        self.w_off = w_off

        self.dilation = int(dilation)
        self.adj = int(adj)
        self.reflect_bounds = reflect_bounds
        self.exact = exact

    def forward(self, vid, dists, inds):
        patches = WpSumHeadsFunction.apply(vid,dists,inds,self.ps,self.pt,
                                           self.h_off,self.w_off,
                                           self.dilation,self.adj,
                                           self.reflect_bounds, self.exact)
        nheads = dists.shape[0]
        nq,_,_,c,ph,pw = patches.shape
        patches = patches.view(nq,nheads,c,ph,pw)
        return patches

