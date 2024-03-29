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

def get_inds(inds,itype):
    inds = inds.contiguous()
    if itype == "int" and th.is_floating_point(inds):
        return inds.round().int()
    elif itype in ["float","2d","3d"] and not(th.is_floating_point(inds)):
        return inds.float()
    else:
        return inds

class PooledPatchSumFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid, weights, flows, ps, stride0,
                pt=1, dilation=1, reflect_bounds=True, use_adj=False, itype="float"):
        """
        vid = [BatchSize,nHeads or 1,T,C,H,W]
        weights = [BatchSize,nHeads,NumQueries,K]
        flows = [BatchSize,nHeads or 1,NumQueries,K,3]
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
        if flows.ndim == 4: flows = flows[:,None] # add heads dim

        # -- unpack --
        device = weights.device
        B,HD,T,nH,nW,K = weights.shape
        wshape = weights.shape
        vid = vid.contiguous()
        flows = get_inds(flows,itype)

        # -- shape output --
        inF,inH,inW = vid.shape[-3:]
        psHalf = (ps-1)//2+1
        outH = ps*nH
        outW = ps*nW
        # print("ps,nH,nW,outH,outW: ",ps,nH,nW,outH,outW)
        out_shape = (B,HD,T,inF,outH,outW)

        # -- allocate --
        dtype = vid.dtype
        out_vid = th.zeros(out_shape,device=device,dtype=dtype)
        counts = th.zeros_like(out_vid[0,0,0,0,:,:]).type(th.int)
        patch_offset = 0 if use_adj else -(ps//2)
        # print(patch_offset)

        # -- view --
        Q = T*nH*nW
        weights = weights.view(B,HD,Q,K)
        flows = flows.view(B,HD,Q,K,3)

        # -- exec --
        fwd_fxn = stnls_cuda.pool_int_forward
        # if flows.dtype == th.int:
        #     fwd_fxn = stnls_cuda.pool_int_forward
        # else:
        #     # flows[...,1:] = flows[...,1:].int()+1
        #     fwd_fxn = stnls_cuda.pool_bilin2d_forward
        ps = ps + (1 - ps % 2)
        fwd_fxn(out_vid, counts, vid, weights, flows,
                ps, stride0, pt, dilation, reflect_bounds, patch_offset)
        eps = 1e-10
        # print(out_vid.shape,vid.shape)
        # print(out_vid.sum((-2,-1)))
        # # print(out_vid[0,0,0,0,:,:].sum((-2)))
        # # print(out_vid[0,0,0,0,:,:].sum((-1)))
        # print(out_vid[0,0,0,0,:5,:5])
        # print(th.where(out_vid==1))
        # print(out_vid[0,0,0,0,:7,:7])
        # print(out_vid[0,0,-1,0,-7:,-7:])
        # print(out_vid[0,0,0,:,6,233])
        # print(out_vid[0,0,0,:,7,229])

        # -- normalize --
        H,W = vid.shape[-2:]
        # # print(counts.sum(-1))
        # # print(counts.sum(-2))
        # print(vid[0,0,0,0,3:,3:])
        # print(counts[3:,3:])
        # print(th.where(counts==0))
        # exit()
        # print("counts [min,max]: ",counts.min().item(),counts.max().item())
        # print("[pre] out_vid [min,max]: ",out_vid.min().item(),out_vid.max().item())
        counts = counts.view((1,1,1,1,outH,outW))
        # from dev_basics.utils import vid_io
        # vid_io.save_video(counts[:,0],"outputs/debug","counts")
        # vid_io.save_video(out_vid[:,0,:,:3]/out_vid.max(),"outputs/debug","vout")
        out_vid = out_vid / (counts+eps)
        # print("out_vid [min,max]: ",out_vid.min().item(),out_vid.max().item())
        assert th.all(counts>1e-3)
        # exit()

        # -- backward --
        ctx.save_for_backward(weights,flows,vid,counts)
        ctx.vid_in_dim = vid_in_dim
        ctx.itype = itype
        ctx.ps,ctx.pt = ps,pt
        ctx.stride0 = stride0
        ctx.vid_shape = vid.shape
        ctx.wshape = wshape
        ctx.dilation = dilation
        ctx.use_adj = use_adj
        ctx.reflect_bounds = reflect_bounds
        ctx.nheads = nheads

        return out_vid

    @staticmethod
    def backward(ctx, grad_out_vid):

        # -- unpack --
        weights,flows,vid,counts = ctx.saved_tensors
        ps,pt = ctx.ps,ctx.pt
        stride0 = ctx.stride0
        vid_shape = ctx.vid_shape
        dilation = ctx.dilation
        use_adj = ctx.use_adj
        reflect_bounds = ctx.reflect_bounds
        HD = ctx.nheads
        itype = ctx.itype
        patch_offset = 0 if use_adj else -(ps//2)

        # -- normalize --
        H,W = counts.shape[-2:]
        grad_out_vid = grad_out_vid / counts.view(1,1,1,H,W)

        # -- alloc --
        grad_weights = th.zeros_like(weights)
        grad_flows = th.zeros_like(flows) if itype == "float" else None
        grad_in_vid = th.zeros_like(grad_out_vid)

        # -- info --
        # print("ps,stride0,pt,dilation,reflect_bounds,patch_offset: ",
        #       ps,stride0,pt,dilation,reflect_bounds,patch_offset)

        # th.cuda.synchronize()
        # print(grad_out_vid[th.where(grad_out_vid>0)])
        # print(grad_out_vid.sum())
        # print(grad_weights[0,0])

        # -- video backward --
        if itype == "int":
            bwd_fxn = stnls_cuda.pool_int_backward
            bwd_fxn(grad_in_vid,grad_weights,
                    grad_out_vid,vid,weights,flows,
                    ps,stride0,pt,dilation,
                    reflect_bounds,patch_offset)
        # elif not(flows.requires_grad):
        #     bwd_fxn = stnls_cuda.wpsum_bilin2d_backward
        #     bwd_fxn(grad_in_vid,grad_weights,
        #             grad_out_vid,vid,weights,flows,
        #             ps,stride0,pt,dilation,
        #             reflect_bounds,patch_offset)
        else:
            bwd_fxn = stnls_cuda.pool_bilin2d_backward
            bwd_fxn(grad_in_vid,grad_weights,grad_flows,
                    grad_out_vid,vid,weights,flows,
                    ps,stride0,pt,dilation,
                    reflect_bounds,patch_offset)

        # print(th.where(grad_weights[0,0].abs()>0))
        # print(grad_weights[th.where(grad_weights.abs()>0)])
        # print(grad_out_vid.sum(),grad_weights.sum())

        # -- shaping vid --
        vid_in_dim = ctx.vid_in_dim
        if vid_in_dim == 5:
            grad_in_vid = rearrange(grad_in_vid,'b hd t c h w -> b t (hd c) h w')
            grad_in_vid = grad_in_vid.contiguous()

        # -- shaping weight,flows --
        grad_weights = grad_weights.reshape(ctx.wshape)
        if ctx.itype == "float":
            grad_flows = grad_flows.reshape(ctx.wshape+(3,))
        else:
            grad_flows = None

        return grad_in_vid,grad_weights,grad_flows,None,None,None,\
            None,None,None,None,None,None,None,None,None

class PooledPatchSum(th.nn.Module):
    # [video -> patches] @ flows

    def __init__(self, ps, stride0, pt=1, dilation=1,
                 reflect_bounds=True, use_adj=False, itype="float"):
        super().__init__()
        _vars = ["ps","stride0","pt","dilation","reflect_bounds","use_adj","itype"]
        self._vars = _vars
        for var in _vars:
            setattr(self,var,eval(var))

    def forward(self, vid, weights, flows):
        inputs = [getattr(self,var) for var in self._vars]
        vid_out = PooledPatchSumFunction.apply(vid,weights,flows,*inputs)
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

def _apply(vid, weights, flows, ps, stride0,
           pt=1, dilation=1,reflect_bounds=True, use_adj=False):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    fxn = PooledPatchSumFunction.apply
    return fxn(vid,weights,flows,ps,stride0,
               pt,dilation,reflect_bounds,use_adj)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#   [Python Dict API] stnls.agg.wpsum(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ps":3,"stride0":1,"pt":1,"dilation":1,
             "reflect_bounds":True, "use_adj":False, "itype":"float"}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    cfg = extract_config(cfg,False)
    reducer = PooledPatchSum(
        cfg.ps, cfg.stride0, pt=cfg.pt, dilation=cfg.dilation,
        reflect_bounds=cfg.reflect_bounds,use_adj=cfg.use_adj,itype=cfg.itype)
    return reducer



