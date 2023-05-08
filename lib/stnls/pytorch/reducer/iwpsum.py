
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

class InplaceWeightedPatchSumFunction(th.autograd.Function):
    # [video -> patches] @ inds

    # -- static video since it is the same --
    # vid = None

    @staticmethod
    def forward(ctx, vid, dists, inds, ps, pt=1,
                h_off=0,w_off=0,dilation=1,adj=0,
                reflect_bounds=True,rbwd=False,nbwd=1,
                exact=False,use_atomic=False):
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
        ctx.use_atomic = use_atomic
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
        use_atomic = ctx.use_atomic

        h_off = ctx.h_off
        w_off = ctx.w_off
        dilation = ctx.dilation
        adj = ctx.adj
        reflect_bounds = ctx.reflect_bounds
        exact = ctx.exact
        rbwd = ctx.rbwd
        nbwd = ctx.nbwd
        # print("wpsum_heads: bwd.")
        grad_vid = grad_vid.contiguous()

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
            grad_vid = allocate_vid(vid_shape_og,grad_vid.device)
            for i in range(nbwd):
                grad_vid_i = allocate_vid(vid_shape,grad_vid.device)
                # stnls_cuda.wpsum_heads_backward_vid(grad_vid_i,grad_vid,
                #                                    dists,inds,
                #                                    h_off,w_off,dilation,adj,
                #                                    reflect_bounds,rbwd,exact)
                if modded_h:
                    grad_vid_i = grad_vid_i.sum(1,keepdim=True)
                grad_vid += grad_vid_i/nbwd
        else:
            grad_vid = allocate_vid(vid_shape,grad_vid.device)
            # print("grad_vid.shape: ",grad_vid.shape)
            stnls_cuda.iwpsum_heads_backward_vid(grad_vid,grad_vid,
                                                dists,inds,
                                                h_off,w_off,dilation,adj,
                                                reflect_bounds,rbwd,
                                                exact,use_atomic)
            # print("grad_vid.shape: ",grad_vid.shape)
            if modded_h:
                grad_vid = grad_vid.sum(1,keepdim=True)
            # print("grad_vid.shape: ",grad_vid.shape)
        # -- end output --

        # -- gradient for dists --
        grad_dists = th.zeros_like(dists)
        stnls_cuda.iwpsum_heads_backward_dists(grad_dists,grad_vid,
                                              vid,inds,
                                              h_off,w_off,dilation,adj,
                                              reflect_bounds,exact,use_atomic)

        # -- final shaping --
        vid_in_dim = ctx.vid_in_dim
        if vid_in_dim == 5:
            grad_vid = rearrange(grad_vid,'b H t c h w -> b t (H c) h w')

        # -- stop timer --
        # th.cuda.synchronize()
        # timer.stop("wpsum_bwd")
        # print(timer)

        return grad_vid,grad_dists,None,None,None,\
            None,None,None,None,None,None,None,None,None

class InplaceWeightedPatchSum(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, batchsize=-1, pt=1, dilation=1, 
                 reflect_bounds=True, use_adj=True, off_H=0, off_W=0,
                 rbwd=False, nbwd=1, exact=False, use_atomic=True):
        super().__init__()

        self.ps = ps
        self.batchsize = batchsize
        self.pt = pt
        self.dilation = int(dilation)
        self.use_adj = use_adj
        self.reflect_bounds = reflect_bounds
        self.off_H = off_H
        self.off_W = off_W
        self.rbwd = rbwd
        self.nbwd = nbwd
        self.exact = exact
        self.use_atomic = use_atomic

    def forward(self, vid, dists, inds):
        fxn = InplaceWeightedPatchSumFunction
        vidz = fxn.apply(vid,dists,inds,self.ps,
                            self.batchsize,self.pt,
                            self.dilation,self.reflect_bounds,
                            self.use_adj,self.off_H,self.off_W,
                            self.rbwd,self.nbwd,
                            self.exact,self.use_atomic)
        return vidz

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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#   [Direct API]  stnls.reducer.iwpsum(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid, dists, inds, ps, batchsize=-1,
           pt=1, dilation=1,reflect_bounds=True,
           use_adj=True, off_H0=0, off_W0=0, off_H1=0, off_W1=0,
           rbwd=True, nbwd=1, exact=False, use_atomic=False):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = InplaceWeightedPatchSumFunction.apply
    return fxn(vid,dists,inds,ps,batchsize,
               pt,dilation,reflect_bounds,use_adj,
               off_H0,off_W0,off_H1,off_W1,
               rbwd,nbwd,exact,use_atomic)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#   [Python Dict API] stnls.reducer.iwpsum(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg):
    pairs = {"ps":7,"batchsize":-1,"pt":1,"dilation":1,
             "reflect_bounds":True, "use_adj":True,
             "off_H0":0,"off_W0":0,"rbwd":True, "nbwd":1,
             "exact":False, "use_atomic": False}
    return extract_pairs(pairs,cfg)

def init(cfg):
    cfg = extract_config(cfg)
    reducer = InplaceWeightedPatchSum(
        cfg.ps, batchsize=cfg.batchsize,
        pt=cfg.pt, dilation=cfg.dilation,
        reflect_bounds=cfg.reflect_bounds,
        adj=cfg.use_adj,off_H=cfg.off_H0,off_W=cfg.off_W0,
        rbwd=cfg.rbwd, nbwd=cfg.nbwd,
        exact=cfg.exact, use_atomic=cfg.use_atomic)
    return reducer



