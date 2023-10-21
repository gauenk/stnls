"""

Usage: see scripts/example_attn.py

"""

# -- python --
import torch as th
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

# -- api --
from stnls.utils import extract_pairs

def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(b,nq,nhead,pt,c,ps,device):
    patches = th.zeros((b,nq,nhead,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class WeightedPatchSumFunction(th.autograd.Function):
    # [video -> patches] @ inds

    # vidz = fxn.apply(vid,dists,inds,self.ps,
    #                  self.pt,self.dilation,
    #                  self.reflect_bounds,self.use_adj,
    #                  self.use_atomic)

    @staticmethod
    def forward(ctx, vid, dists, inds, ps,
                pt=1, dilation=1, reflect_bounds=True, use_adj=False):
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
        if vid.ndim == 5:
            if (total_color % nheads) == 0:
                vid = rearrange(vid,'b t (H c) h w -> b H t c h w',H=nheads)
            else:
                vid = rearrange(vid,'b t c h w -> b 1 t c h w')
        if inds.ndim == 4: inds = inds[:,None] # add heads dim
        # print("dists.shape,inds.shape: " ,dists.shape,inds.shape)

        # if WpSumFunction.vid is None: WpSumFunction.vid = vid
        device = dists.device
        B,HD,T,nH,nW,k = dists.shape
        vid = vid.contiguous()
        inds = inds.contiguous()
        out_vid = th.zeros_like(vid)
        out_vidz = th.zeros_like(vid[:,:,:1,:1,:,:]).type(th.int)
        patch_offset = 0 if use_adj else -(ps//2)

        if inds.dtype == th.int:
            fwd_fxn = stnls_cuda.wpsum_int_forward
        else:
            fwd_fxn = stnls_cuda.wpsum_bilin2d_forward
        fwd_fxn(vid, out_vid, out_vidz, dists, inds,
                ps, pt, dilation, reflect_bounds, patch_offset)
        out_vid = out_vid / out_vidz
        ctx.save_for_backward(dists,inds,vid)
        ctx.vid_in_dim = vid_in_dim
        ctx.ps,ctx.pt = ps,pt
        ctx.vid_shape = vid.shape
        ctx.dilation = dilation
        ctx.use_adj = use_adj
        ctx.reflect_bounds = reflect_bounds
        ctx.nheads = nheads

        # -- viz --
        # print("fwd.")
        # print("patches.shape: ",patches.shape)

        # -- reshape --
        out_vid = rearrange(out_vid,'b H t c h w -> b t (H c) h w')

        return out_vid

    @staticmethod
    def backward(ctx, grad_in):

        # -- unpack --
        dists,inds,vid = ctx.saved_tensors
        ps,pt = ctx.ps,ctx.pt
        vid_shape = ctx.vid_shape
        dilation = ctx.dilation
        use_adj = ctx.use_adj
        reflect_bounds = ctx.reflect_bounds
        HD = ctx.nheads
        grad_in = grad_in.contiguous()

        # -- reshape --
        grad_in = rearrange(grad_in,'b t (hd c) h w -> b hd t c h w',H=HD)
        grad_in = grad_in.contiguous()

        # -- video backward --
        th.cuda.synchronize()
        grad_out = th.zeros_like(grad_in)
        stnls_cuda.wpsum_backward_vid(grad_out,grad_in,
                                       dists,inds,ps,pt,
                                       dilation,reflect_bounds,use_adj)
        th.cuda.synchronize()

        # -- distances backward --
        grad_dists = th.zeros_like(dists)
        stnls_cuda.wpsum_backward_dists(grad_dists,grad_in,
                                         vid,inds,ps,pt,
                                         dilation,reflect_bounds,use_adj)
        th.cuda.synchronize()

        # -- final shaping --
        vid_in_dim = ctx.vid_in_dim
        if vid_in_dim == 5:
            grad_out = rearrange(grad_out,'b hd t c h w -> b t (hd c) h w')
            grad_out = grad_out.contiguous()
            # print("grad_out.shape: ",grad_out.shape)

        return grad_out,grad_dists,None,None,None,\
            None,None,None,None,None,None,None,None,None

class WeightedPatchSum(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, pt=1, dilation=1,
                 reflect_bounds=True, use_adj=False):
        super().__init__()

        self.ps = ps
        self.pt = pt
        self.dilation = int(dilation)
        self.use_adj = use_adj
        self.reflect_bounds = reflect_bounds

    def forward(self, vid, dists, inds):
        fxn = WeightedPatchSumFunction
        vidz = fxn.apply(vid,dists,inds,self.ps,
                         self.pt,self.dilation,
                         self.reflect_bounds,self.use_adj)
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
#   [Direct API]  stnls.reducer.wpsum(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid, dists, inds, ps,
           pt=1, dilation=1,reflect_bounds=True, use_adj=False):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = WeightedPatchSumFunction.apply
    return fxn(vid,dists,inds,ps,
               pt,dilation,reflect_bounds,use_adj)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#   [Python Dict API] stnls.reducer.wpsum(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg):
    pairs = {"ps":7,"pt":1,"dilation":1,
             "reflect_bounds":True, "use_adj":False}
    return extract_pairs(pairs,cfg)

def init(cfg):
    cfg = extract_config(cfg)
    reducer = WeightedPatchSum(
        cfg.ps, pt=cfg.pt, dilation=cfg.dilation,
        reflect_bounds=cfg.reflect_bounds,use_adj=cfg.use_adj)
    return reducer



