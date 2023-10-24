"""

Stack Non-Local Patches

Usage: see scripts/example_attn.py


Example:

    vid # [B HD T F H W] or [B T F' H W] with F' = (HD F) and HD = inds.shape[1]
    weights.shape # B,HD,Q,K
    inds.shape # B,HD,Q,K,3
    stacking = NonLocalStack(ps=1)
    stack = stacking(vid,weights,inds)
    stack # [B HD K T F H W]

"""


# -- python --
import torch as th
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

def ensure_ndim6(vid,nheads):
    if vid.ndim == 5:
        B,T,HD_F,H,W = vid.shape
        vid = rearrange(vid,'b t (hd f) h w -> b hd t f h w',hd=nheads)
    assert vid.ndim == 6
    return vid

def revert_ndim(grad_vid,ndim):
    if ndim == 5:
        B,HD,T,F,H,W = grad_vid.shape
        grad_vid = rearrange(grad_vid,'b hd t f h w -> b t (hd f) h w')
    return grad_vid

def get_counts(vid,stride0,ps,pt,dilation,use_adj,reflect_bounds):
    vid0 = vid[:,0]
    B,T,F,H,W = vid0.shape
    device = vid0.device
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1
    Q = T*nH*nW
    patches = th.zeros((B,Q,pt,F,ps,ps),device=device,dtype=th.float32)
    _vid = th.zeros(vid0.shape,device=device,dtype=th.float32)
    counts = th.zeros(vid0.shape,device=device,dtype=th.float32)
    stnls_cuda.nlfold_forward(_vid, counts, patches, stride0, dilation,
                              use_adj,  reflect_bounds)
    counts = counts[0,0,0].type(th.int32)
    return counts

def get_inds(inds,itype):
    inds = inds.contiguous()
    if itype == "int" and th.is_floating_point(inds):
        return inds.round().int()
    elif itype in ["float","2d","3d"] and not(th.is_floating_point(inds)):
        return inds.float()
    else:
        return inds

def get_imode(itype):
    if itype == "int": return 0
    elif itype in ["float","2d"]: return 1
    elif itype in ["3d"]: return 2

class non_local_stack(th.autograd.Function):
    """

    Stack the non-local patches according to inds
    across the "ki"^th channel

    """

    @staticmethod
    def forward(ctx, vid, weights, inds,
                ps=7,stride0=4,pt=1,reflect_bounds=True,
                dilation=1,use_adj=False, itype="int"):

        # -- init --
        HD = inds.shape[1]
        K = inds.shape[-2]
        q_start=0
        ndim = vid.ndim
        vid = ensure_ndim6(vid,HD)
        B,HD_v,T,F,H,W = vid.shape
        HD_i = inds.shape[1]
        HD = max(HD_v,HD_i)
        wshape = weights.shape
        stack = th.zeros((B,HD,K,T,F,H,W),device=vid.device,dtype=th.float32)
        counts = th.zeros((B,HD,H,W),device=vid.device,dtype=th.int32)
        # print("B,HD,K,T,F,H,W: ",B,HD,K,T,F,H,W)
        # print("stack [weights.shape,inds.shape]: ",weights.shape,inds.shape)
        patch_offset = 0 if use_adj else -(ps//2)

        # -- reshape --
        # nH = (H-1)//stride0+1
        # nW = (W-1)//stride0+1
        weights = weights.view(B,HD,-1,K).clone()
        inds = inds.view(B,HD,-1,K,3).clone()

        # -- non-local stacking --
        vid = vid.contiguous().clone()
        weights = weights.contiguous().clone()
        inds = get_inds(inds,itype).clone()
        imode = get_imode(itype)
        assert inds.shape[-1] == 3

        # -- all same num of heads --
        assert vid.shape[1] == inds.shape[1]
        assert inds.shape[1] == weights.shape[1]
        # print(vid.shape)
        # print(inds[0,0,9])
        # inds_n = rearrange(inds,'b HD (H W) k two -> b (HD k) two H W',H=H,W=W)
        # print(inds_n[0,0])
        # for i in range(inds_n.shape[1]):
        #     print(inds_n[0,i,:,:3,:3])
        # print(stride0,use_adj,ps)
        assert not th.any(th.isnan(weights)).item()
        # print(vid.shape)

        # print(vid.shape,weights.shape,inds.shape,stack.shape,counts.shape)
        # print(weights[0,0,0,0])
        # print(inds,imode)
        if inds.dtype == th.int:
            fwd_fxn = stnls_cuda.non_local_stack_int_forward
            fwd_fxn(vid, weights, inds, stack, counts,
                    ps, pt, dilation, stride0,
                    reflect_bounds, patch_offset)
        else:
            fwd_fxn = stnls_cuda.non_local_stack_bilin2d_forward
            fwd_fxn(vid, weights, inds, stack, counts,
                    ps, pt, dilation, stride0,
                    reflect_bounds, patch_offset)
        assert th.all(counts > 0).item()
        # print(counts[0,0])
        # print(counts[1,0])
        # print(counts[2,0])

        eps = 1e-10
        stack /= (counts.view((B,HD,1,1,1,H,W))+eps)
        assert not th.any(th.isnan(stack)).item()

        # -- save for back-prop --
        ctx.save_for_backward(vid,stack,weights,inds,counts)
        ctx.stride0 = stride0
        ctx.dilation = dilation
        ctx.use_adj = use_adj
        ctx.reflect_bounds = reflect_bounds
        ctx.ps = ps
        ctx.pt = pt
        ctx.ndim = ndim
        ctx.itype = itype
        ctx.wshape = wshape

        return stack

    @staticmethod
    def backward(ctx, grad_stack):

        # -- unpack ctx --
        vid,stack,weights,inds,counts = ctx.saved_tensors
        ps,pt = ctx.ps,ctx.pt
        dilation = ctx.dilation
        stride0 = ctx.stride0
        use_adj = ctx.use_adj
        reflect_bounds = ctx.reflect_bounds
        itype = ctx.itype
        ndim = ctx.ndim
        imode = get_imode(itype)
        patch_offset = 0 if use_adj else -(ps//2)

        # -- alloc --
        grad_vid = th.zeros_like(vid)
        grad_weights = th.zeros_like(weights)
        grad_stack = grad_stack.contiguous()
        if itype != "int": grad_inds = th.zeros_like(inds)
        else: grad_inds = th.zeros((1,)*5).to(inds.device).int()

        # print(grad_stack[0,0,0,0,:,:2,:2])
        # print(th.all(grad_stack==0))

        # -- view --
        # print("grad_vid.shape: ",grad_vid.shape)
        # print("grad_stack.shape: ",grad_stack.shape)
        # print("grad_weights.shape: ",grad_weights.shape)
        # print("vid.shape: ",vid.shape)
        # print("weights.shape: ",weights.shape)
        # print("inds.shape: ",inds.shape)
        # print("stack.shape: ",stack.shape)
        # print("counts.shape: ",counts.shape)
        # print("ps,pt,dilation,stride0: ",ps,pt,dilation,stride0)

        # -- exec --
        # print("counts.")
        # import stnls
        # H,W = counts.shape
        # counts_og = counts
        # counts = get_counts(vid,stride0,ps,pt,
        #                     dilation,use_adj,reflect_bounds)
        # print(counts_og[:5,:5])
        # print(counts[:5,:5])
        # print(counts_og[-5:,-5:])
        # print(counts[-5:,-5:])


        # print(th.mean(1.*(counts_og-counts)**2))
        # counts_s = counts / counts.max()
        # counts_s = counts_s.view(1,1,1,H,W)
        # stnls.utils.vid_io.save_video(counts_s,"./output/tests/tile/","counts_nlfold")
        # counts_s = counts_og / counts_og.max()
        # counts_s = counts_s.view(1,1,1,H,W)
        # stnls.utils.vid_io.save_video(counts_s,"./output/tests/tile/","counts_nlstack")

        # -- backward --
        # print("non_local_stack")
        # print(counts[30:34,30:34])
        # H,W = counts.shape
        # print(grad_stack.abs().mean())
        # print(grad_stack[0,0,0,0,0,30:34,30:34])
        B,HD,T,C,H,W = grad_vid.shape
        eps = 1e-10
        grad_stack = grad_stack / (counts+eps)
        if ctx.itype == "int":
            fwd_fxn = stnls_cuda.non_local_stack_int_backward
            fwd_fxn(grad_vid,grad_weights,grad_stack,
                    vid,weights,inds,stack,counts,
                    ps,pt,dilation,stride0,reflect_bounds,patch_offset)
        else:
            fwd_fxn = stnls_cuda.non_local_stack_bilin2d_backward
            fwd_fxn(grad_vid,grad_weights,grad_inds,grad_stack,
                    vid,weights,inds,stack,counts,
                    ps,pt,dilation,stride0,reflect_bounds,patch_offset)

        # -- info --
        # print("grad weights.")
        # print(grad_weights.shape)
        # print(th.all(grad_weights==0))

        # -- ensure original ndim --
        grad_vid = revert_ndim(grad_vid,ndim)

        # -- don't propogate "int" --
        if itype == "int": grad_inds = None
        else:
            grad_inds = grad_inds.reshape(ctx.wshape+(3,))

        # -- reshape --
        grad_weights = grad_weights.reshape(ctx.wshape)

        return grad_vid,grad_weights,grad_inds,None,None,\
            None,None,None,None,None,None,None,None,None,None

class NonLocalStack(th.nn.Module):

    def __init__(self,ps,stride0,pt=1,dilation=1,
                 reflect_bounds=True,use_adj=False,itype="float"):
        super().__init__()
        _vars = ["ps","stride0","pt","reflect_bounds","dilation","use_adj","itype"]
        self._vars = _vars
        for var in _vars:
            setattr(self,var,eval(var))

    def forward(self, vid, weights, inds):
        inputs = [getattr(self,var) for var in self._vars]
        stack = non_local_stack.apply(vid, weights, inds, *inputs)
        return stack

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#            [Functional API]  stnls.agg.nlstack(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid, weights, flows, ps=1, stride0=1, pt=1,
           reflect_bounds=True, dilation=1, use_adj=False, itype="float"):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = NonLocalSearchFunction.apply
    return fxn(vid,weights,flows,ps,stride0,pt,reflect_bounds,dilation,use_adj,itype)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Python Dict API] stnls.tile.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ps":7,"stride0":4,"pt":1,"reflect_bounds":True,
             "dilation":1, "use_adj":False,"itype":"int"}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    cfg = extract_config(cfg)
    search = NonLocalStack(cfg.ps,cfg.stride0,cfg.pt,cfg.reflect_bounds,
                           cfg.dilation,cfg.use_adj,cfg.itype)
    return search


