
# -- python --
import torch as th
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

# -- misc --
from ...utils.timer import ExpTimer


def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(b,nq,nhead,pt,c,ps,device):
    patches = th.zeros((b,nq,nhead,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class WpSumHeadsFunction(th.autograd.Function):
    # [video -> patches] @ inds

    # -- static video since it is the same --
    # vid = None

    @staticmethod
    def forward(ctx, vid, dists, inds, ps, pt=1,
                h_off=0,w_off=0,dilation=1,adj=0,
                reflect_bounds=True,rbwd=False,nbwd=1,exact=False):
        """
        vid = [BatchSize,nHeads or 1,T,C,H,W]
        dists = [BatchSize,nHeads,NumQueries,K]
        inds = [BatchSize,nHeads or 1,NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        """
        # -- add head dim if 1 --
        vid_in_dim = vid.ndim
        total_color = vid.shape[-3]
        bsize,nheads = dists.shape[:2]
        # print("vid.shape: ",vid.shape,nheads,bsize)
        # assert vid.ndim == 5,"must be 5 dims"
        if vid.ndim == 5:
            if (total_color % nheads) == 0:
                vid = rearrange(vid,'b t (H c) h w -> b H t c h w',H=nheads)
            else:
                vid = rearrange(vid,'b t c h w -> b 1 t c h w')
        if inds.ndim == 4: inds = inds[:,None] # add heads dim
        # print("dists.shape,inds.shape: " ,dists.shape,inds.shape)

        # if WpSumFunction.vid is None: WpSumFunction.vid = vid
        device = dists.device
        bsize,nheads,nq,k = dists.shape
        # print("vid.shape: ",vid.shape)
        # print("bsize,nheads,nq,pt,vid.shape[3],ps: ",bsize,nheads,nq,pt,vid.shape[3],ps)
        patches = allocate_patches(bsize,nheads,nq,pt,vid.shape[-3],ps,device)
        vid = vid.contiguous()

        # print(vid.shape)
        # print(patches.shape)
        # print(dists.shape)
        # print(inds.shape)
        # print("-"*10)
        # print("adj: ",adj)
        # print("reflect_bounds: ",reflect_bounds)
        # print("inds.shape: ",inds.shape)

        # void cuda_wpsum_heads_forward(
        #     torch::Tensor vid, torch::Tensor patches,
        #     torch::Tensor dists, torch::Tensor inds,
        #     int h_off, int w_off, int dilation, int adj, bool reflect_bounds){
        # print(h_off,w_off,dilation,adj,reflect_bounds)
        stnls_cuda.wpsum_heads_forward(vid, patches, dists, inds,
                                      h_off,w_off,dilation,adj,
                                      reflect_bounds)
        # print("dists._version: ",dists._version)
        # print("inds._version: ",inds._version)
        ctx.save_for_backward(dists,inds,vid)
        ctx.vid_in_dim = vid_in_dim
        ctx.ps,ctx.pt = ps,pt
        ctx.vid_shape = vid.shape
        ctx.dilation = dilation
        ctx.h_off = h_off
        ctx.w_off = w_off
        ctx.adj = adj
        ctx.reflect_bounds = reflect_bounds
        ctx.exact = exact
        ctx.rbwd = rbwd
        ctx.nbwd = nbwd

        # -- viz --
        # print("fwd.")
        # print("patches.shape: ",patches.shape)

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
        rbwd = ctx.rbwd
        nbwd = ctx.nbwd
        # print("wpsum_heads: bwd.")
        grad_patches = grad_patches.contiguous()

        # -- viz --
        # print("bwd.")
        # print("vid.shape: ",vid.shape)
        # print("dists.shape: ",dists.shape)
        # print("grad_patches.shape: ",grad_patches.shape)

        # -- start timer --
        # timer = ExpTimer()
        # timer.start("wpsum_heads_bwd")
        # print(grad_patches.shape)
        # if nbwd > 1:
        #     print("Warning: nbwd not implemented for wpsum.")

        # -- gradient for video --
        # print(vid_shape,inds.shape,dists.shape,vid.shape)
        # print(h_off,w_off)
        # print(vid_shape)
        _,nheads,_,_ = dists.shape
        _b,_H,_t,_c,_h,_w = vid_shape
        modded_h = False
        vid_shape_og = list(vid_shape)
        vid_shape = list(vid_shape)
        if _H == 1 and _H != nheads:
            vid_shape[1] = nheads
            modded_h = True

        # -- ave multiple evals --
        if nbwd > 1:
            grad_vid = allocate_vid(vid_shape_og,grad_patches.device)
            for i in range(nbwd):
                grad_vid_i = allocate_vid(vid_shape,grad_patches.device)
                # stnls_cuda.wpsum_heads_backward_vid(grad_vid_i,grad_patches,
                #                                    dists,inds,
                #                                    h_off,w_off,dilation,adj,
                #                                    reflect_bounds,rbwd,exact)
                if modded_h:
                    grad_vid_i = grad_vid_i.sum(1,keepdim=True)
                grad_vid += grad_vid_i/nbwd
        else:
            grad_vid = allocate_vid(vid_shape,grad_patches.device)
            # print("grad_vid.shape: ",grad_vid.shape)
            # stnls_cuda.wpsum_heads_backward_vid(grad_vid,grad_patches,
            #                                    dists,inds,
            #                                    h_off,w_off,dilation,adj,
            #                                    reflect_bounds,rbwd,exact)
            # print("grad_vid.shape: ",grad_vid.shape)
            if modded_h:
                grad_vid = grad_vid.sum(1,keepdim=True)
            # print("grad_vid.shape: ",grad_vid.shape)
        # -- end output --

        # -- gradient for dists --
        grad_dists = th.zeros_like(dists)
        stnls_cuda.wpsum_heads_backward_dists(grad_dists,grad_patches,
                                             vid,inds,
                                             h_off,w_off,dilation,adj,
                                             reflect_bounds,exact)
        # th.cuda.synchronize()
        # print("2.")

        # -- final shaping --
        vid_in_dim = ctx.vid_in_dim
        if vid_in_dim == 5:
            grad_vid = rearrange(grad_vid,'b H t c h w -> b t (H c) h w')

        # -- stop timer --
        # th.cuda.synchronize()
        # timer.stop("wpsum_bwd")
        # print(timer)

        return grad_vid,grad_dists,None,None,None,\
            None,None,None,None,None,None,None,None

class WeightedPatchSumHeads(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, pt=1, h_off=0, w_off=0, dilation=1,
                 adj=0, reflect_bounds = True, rbwd=False, nbwd=1, exact=False):
        super(WeightedPatchSumHeads, self).__init__()
        self.ps = ps
        self.pt = pt

        self.h_off = h_off
        self.w_off = w_off

        self.dilation = int(dilation)
        self.adj = int(adj)
        self.reflect_bounds = reflect_bounds
        self.rbwd = rbwd
        self.nbwd = nbwd
        self.exact = exact

    def forward(self, vid, dists, inds):
        patches = WpSumHeadsFunction.apply(vid,dists,inds,self.ps,self.pt,
                                           self.h_off,self.w_off,
                                           self.dilation,self.adj,
                                           self.reflect_bounds,
                                           self.rbwd,self.nbwd,self.exact)
        nheads = dists.shape[1]
        b,nheads,nq,_,c,ph,pw = patches.shape
        patches = patches.view(b,nheads,nq,c,ph,pw)
        return patches

    def flops(self, nrefs, chnls_per_head, nheads, k):

        # -- init --
        flops = 0

        # -- unpack --
        chnls = chnls_per_head
        ps,pt = self.ps,self.pt

        # -- compute weighted patch sum --
        flops_per_patch = 2 * (chnls * ps * ps * pt) # multi weight & add to accumulate
        flops_per_ref = flops_per_patch * k # accumulate over "k" patches
        flops = flops_per_ref * nrefs * nheads# do for each reference

        return flops