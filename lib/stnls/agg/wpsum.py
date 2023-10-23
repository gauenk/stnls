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

    @staticmethod
    def forward(ctx, vid, weights, inds, ps, stride0,
                pt=1, dilation=1, reflect_bounds=True, use_adj=False, itype="int"):
        """
        vid = [BatchSize,nHeads or 1,T,C,H,W]
        weights = [BatchSize,nHeads,NumQueries,K]
        inds = [BatchSize,nHeads or 1,NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        """

        # -- add head dim if 1 --
        vid_in_dim = vid.ndim
        total_color = vid.shape[-3]
        bsize,nheads = weights.shape[:2]
        if vid.ndim == 5:
            if (total_color % nheads) == 0:
                vid = rearrange(vid,'b t (H c) h w -> b H t c h w',H=nheads)
            else:
                vid = rearrange(vid,'b t c h w -> b 1 t c h w')
        if inds.ndim == 4: inds = inds[:,None] # add heads dim
        # print("weights.shape,inds.shape: " ,weights.shape,inds.shape)

        # -- allocate --
        device = weights.device
        B,HD,T,nH,nW,K = weights.shape
        vid = vid.contiguous()
        inds = inds.contiguous()
        out_vid = th.zeros_like(vid)
        counts = th.zeros_like(vid[:1,:1,:1,:1,:,:]).type(th.int)
        patch_offset = 0 if use_adj else -(ps//2)

        # -- view --
        Q = T*nH*nW
        weights = weights.view(B,HD,Q,K)
        inds = inds.view(B,HD,Q,K,3)

        # -- exec --
        if inds.dtype == th.int:
            fwd_fxn = stnls_cuda.wpsum_int_forward
        else:
            fwd_fxn = stnls_cuda.wpsum_bilin2d_forward
        fwd_fxn(out_vid, counts, vid, weights, inds,
                ps, stride0, pt, dilation, reflect_bounds, patch_offset)
        eps = 1e-10
        out_vid = out_vid / (counts+eps)
        assert th.all(counts>1e-3)

        # -- backward --
        ctx.save_for_backward(weights,inds,vid)
        ctx.vid_in_dim = vid_in_dim
        ctx.ps,ctx.pt = ps,pt
        ctx.vid_shape = vid.shape
        ctx.dilation = dilation
        ctx.use_adj = use_adj
        ctx.reflect_bounds = reflect_bounds
        ctx.nheads = nheads

        return out_vid

    @staticmethod
    def backward(ctx, grad_out_vid):

        # -- unpack --
        weights,inds,vid = ctx.saved_tensors
        ps,pt = ctx.ps,ctx.pt
        vid_shape = ctx.vid_shape
        dilation = ctx.dilation
        use_adj = ctx.use_adj
        reflect_bounds = ctx.reflect_bounds
        HD = ctx.nheads
        patch_offset = 0 if use_adj else -(ps//2)

        # -- reshape --
        grad_out_vid = rearrange(grad_out_vid,'b t (hd c) h w -> b hd t c h w',H=HD)
        grad_out_vid = grad_out_vid.contiguous()

        # -- video backward --
        grad_in_vid = th.zeros_like(grad_out_vid)
        stnls_cuda.wpsum_int_backward(grad_in_vid,grad_weights,
                                      grad_out_vid,weights,inds,ps,stride0,pt,
                                      dilation,reflect_bounds,patch_offset)

        # -- final shaping --
        vid_in_dim = ctx.vid_in_dim
        if vid_in_dim == 5:
            grad_in_vid = rearrange(grad_in_vid,'b hd t c h w -> b t (hd c) h w')
            grad_in_vid = grad_in_vid.contiguous()
            # print("grad_in_vid.shape: ",grad_in_vid.shape)

        return grad_in_vid,grad_weights,None,None,None,None,\
            None,None,None,None,None,None,None,None,None

class WeightedPatchSum(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, stride0, pt=1, dilation=1,
                 reflect_bounds=True, use_adj=False, itype="float"):
        super().__init__()
        _vars = ["ps","stride0","pt","reflect_bounds","dilation","use_adj","itype"]
        self._vars = _vars
        for var in _vars:
            setattr(self,var,eval(var))

    def forward(self, vid, weights, inds):
        inputs = [getattr(self,var) for var in self._vars]
        vid_out = WeightedPatchSumFunction.apply(vid,weights,inds,*inputs)
        return vid_out

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
#   [Direct API]  stnls.agg.wpsum(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid, weights, inds, ps, stride0,
           pt=1, dilation=1,reflect_bounds=True, use_adj=False):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    fxn = WeightedPatchSumFunction.apply
    return fxn(vid,weights,inds,ps,stride0,
               pt,dilation,reflect_bounds,use_adj)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#   [Python Dict API] stnls.agg.wpsum(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg):
    pairs = {"ps":7,"stride0":1,"pt":1,"dilation":1,
             "reflect_bounds":True, "use_adj":False}
    return extract_pairs(pairs,cfg)

def init(cfg):
    cfg = extract_config(cfg)
    reducer = WeightedPatchSum(
        cfg.ps, cfg.stride0, pt=cfg.pt, dilation=cfg.dilation,
        reflect_bounds=cfg.reflect_bounds,use_adj=cfg.use_adj)
    return reducer



