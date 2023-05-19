"""

WeightedPatchSum

input: video
output: patches

"""

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

class WeightedPatchSumFunction(th.autograd.Function):
    # [video -> patches] @ inds

    # -- static video since it is the same --
    # vid = None

    @staticmethod
    def forward(ctx, vid, dists, inds, ps, pt=1,
                dilation=1,reflect_bounds=True,use_adj=False,
                off_H=0,off_W=0,rbwd=False,nbwd=1,
                exact=False,use_atomic=True):
        """
        vid = [BatchSize,nHeads or 1,T,C,H,W]
        dists = [BatchSize,nHeads,NumQueries,K]
        inds = [BatchSize,nHeads or 1,NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        """

        # -- reshape inputs --
        vid_in_dim = vid.ndim
        total_color = vid.shape[-3]
        if dists.ndim == 3: dists = dists[:,None] # add heads dim
        bsize,nheads = dists.shape[:2]
        if vid.ndim == 5:
            if (total_color % nheads) == 0:
                vid = rearrange(vid,'b t (H c) h w -> b H t c h w',H=nheads)
            else:
                vid = rearrange(vid,'b t c h w -> b 1 t c h w')
        if inds.ndim == 4: inds = inds[:,None] # add heads dim

        # -- alloc --
        device = dists.device
        bsize,nheads,nq,k = dists.shape
        patches = allocate_patches(bsize,nheads,nq,pt,vid.shape[-3],ps,device)

        # -- forward --
        vid = vid.contiguous()
        dists = dists.contiguous()
        inds = inds.contiguous()
        stnls_cuda.wpsum_forward(vid, patches, dists, inds,
                                 off_H,off_W,dilation,use_adj,
                                 reflect_bounds)

        # -- ctx save --
        ctx.save_for_backward(dists,inds,vid)
        ctx.vid_in_dim = vid_in_dim
        ctx.ps,ctx.pt = ps,pt
        ctx.vid_shape = vid.shape
        ctx.dilation = dilation
        ctx.off_H = off_H
        ctx.off_W = off_W
        ctx.use_adj = use_adj
        ctx.reflect_bounds = reflect_bounds
        ctx.use_atomic = use_atomic
        ctx.exact = exact
        ctx.rbwd = rbwd
        ctx.nbwd = nbwd

        return patches

    @staticmethod
    def backward(ctx, grad_patches):

        # -- unpack --
        dists,inds,vid = ctx.saved_tensors
        ps,pt = ctx.ps,ctx.pt
        vid_shape = ctx.vid_shape
        use_atomic = ctx.use_atomic
        off_H = ctx.off_H
        off_W = ctx.off_W
        dilation = ctx.dilation
        use_adj = ctx.use_adj
        reflect_bounds = ctx.reflect_bounds
        exact = ctx.exact
        rbwd = ctx.rbwd
        nbwd = ctx.nbwd
        grad_patches = grad_patches.contiguous()

        # -- reshaping --
        _,nheads,_,_ = dists.shape
        _b,_H,_t,_c,_h,_w = vid_shape
        modded_h = False
        vid_shape_og = list(vid_shape)
        vid_shape = list(vid_shape)
        if _H == 1 and _H != nheads:
            vid_shape[1] = nheads
            modded_h = True

        # -- video backward --
        if nbwd > 1:
            grad_vid = allocate_vid(vid_shape_og,grad_patches.device)
            for i in range(nbwd):
                grad_vid_i = allocate_vid(vid_shape,grad_patches.device)
                stnls_cuda.wpsum_backward_vid(grad_vid_i,grad_patches,
                                              dists,inds,
                                              off_H,off_W,dilation,use_adj,
                                              reflect_bounds,rbwd,exact,
                                              use_atomic)
                if modded_h:
                    grad_vid_i = grad_vid_i.sum(1,keepdim=True)
                grad_vid += grad_vid_i/nbwd
        else:
            grad_vid = allocate_vid(vid_shape,grad_patches.device)
            stnls_cuda.wpsum_backward_vid(grad_vid,grad_patches,
                                          dists,inds,
                                          off_H,off_W,dilation,use_adj,
                                          reflect_bounds,rbwd,exact,use_atomic)
            if modded_h:
                grad_vid = grad_vid.sum(1,keepdim=True)

        # -- distances backward --
        grad_dists = th.zeros_like(dists)
        stnls_cuda.wpsum_backward_dists(grad_dists,grad_patches,
                                              vid,inds,
                                              off_H,off_W,dilation,use_adj,
                                              reflect_bounds,exact,use_atomic)
        # -- final shaping --
        vid_in_dim = ctx.vid_in_dim
        if vid_in_dim == 5:
            grad_vid = rearrange(grad_vid,'b H t c h w -> b t (H c) h w')

        return grad_vid,grad_dists,None,None,None,None,\
            None,None,None,None,None,None,None,None,None

class WeightedPatchSum(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, pt=1, dilation=1,
                 reflect_bounds=True, use_adj=False,
                 off_H=0, off_W=0,
                 rbwd=False, nbwd=1, exact=False, use_atomic=True):
        super().__init__()

        self.ps = ps
        self.pt = pt

        self.off_H = off_H
        self.off_W = off_W

        self.dilation = int(dilation)
        self.use_adj = use_adj
        self.reflect_bounds = reflect_bounds
        self.rbwd = rbwd
        self.nbwd = nbwd
        self.exact = exact
        self.use_atomic = use_atomic

    def forward(self, vid, dists, inds):
        fxn = WeightedPatchSumFunction
        patches = fxn.apply(vid,dists,inds,self.ps,self.pt,
                            self.dilation,self.reflect_bounds,
                            self.use_adj,self.off_H,self.off_W,
                            self.rbwd,self.nbwd,
                            self.exact,self.use_atomic)
        # b,nheads,nq,_,c,ph,pw = patches.shape
        # patches = patches.view(b,nheads,nq,c,ph,pw)
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


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#   [Direct API]  stnls.reducer.wpsum(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid, dists, inds, ps, pt=1,
           dilation=1,reflect_bounds=True,
           use_adj=False, off_H0=0, off_W0=0,
           rbwd=True, nbwd=1, exact=False, use_atomic=True):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = WeightedPatchSumFunction.apply
    return fxn(vid,dists,inds,ps,pt,dilation,
               reflect_bounds,use_adj,off_H0,off_W0,
               rbwd,nbwd,exact,use_atomic)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#   [Python Dict API] stnls.reducer.wpsum(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg):
    pairs = {"ps":7,"pt":1,"dilation":1,
             "reflect_bounds":True, "use_adj":False,
             "off_H0":0,"off_W0":0, "rbwd":False, 
             "nbwd":1, "exact":False, "use_atomic": True}
    return extract_pairs(pairs,cfg)

def init(cfg):
    cfg = extract_config(cfg)
    reducer = WeightedPatchSum(cfg.ps,pt=cfg.pt, dilation=cfg.dilation,
                               adj=cfg.use_adj,reflect_bounds=cfg.reflect_bounds,
                               off_H=cfg.off_H0,off_W=cfg.off_W0,
                               rbwd=cfg.rbwd, nbwd=cfg.nbwd,
                               exact=cfg.exact, use_atomic=cfg.use_atomic)
    return reducer


