"""

Stack Non-Local Patches


vid # [B HD T F H W] or [B T F' H W] with F' = (HD F) and HD = inds.shape[1]
inds.shape # B,HD,Q,K

stack = stnls.non_local_stack(vid,inds,ps=ps,stride0=stride0)

stack # [B HD T F H W]


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
                dilation=1,use_adj=False,off_H0=0,off_W0=0,off_H1=0,off_W1=0,
                itype_fwd="int",itype_bwd="int"):

        # -- init --
        HD = inds.shape[1]
        K = inds.shape[-2]
        q_start=0
        ndim = vid.ndim
        vid = ensure_ndim6(vid,HD)
        B,HD_v,T,F,H,W = vid.shape
        HD_i = inds.shape[1]
        HD = max(HD_v,HD_i)
        stack = th.zeros((B,HD,K,T,F,H,W),device=vid.device,dtype=th.float32)
        counts = th.zeros((B,HD,H,W),device=vid.device,dtype=th.int32)
        # print("B,HD,K,T,F,H,W: ",B,HD,K,T,F,H,W)
        # print("stack [weights.shape,inds.shape]: ",weights.shape,inds.shape)

        # -- reshape --
        # nH = (H-1)//stride0+1
        # nW = (W-1)//stride0+1
        weights = weights.reshape(B,HD,-1,K)
        inds = inds.reshape(B,HD,-1,K,3)

        # -- non-local stacking --
        vid = vid.contiguous()
        weights = weights.contiguous()
        inds = get_inds(inds,itype_fwd)
        imode = get_imode(itype_fwd)
        assert inds.shape[-1] == 3
        # print(vid.shape)
        # print(inds[0,0,9])
        # inds_n = rearrange(inds,'b HD (H W) k two -> b (HD k) two H W',H=H,W=W)
        # print(inds_n[0,0])
        # for i in range(inds_n.shape[1]):
        #     print(inds_n[0,i,:,:3,:3])
        # print(stride0,use_adj,ps)
        assert not th.any(th.isnan(weights)).item()

        # print(inds,imode)
        stnls_cuda.non_local_stack_forward(vid, weights, inds,
                                           stack, counts,
                                           ps, pt, dilation, stride0,
                                           use_adj, reflect_bounds, q_start,
                                           off_H0, off_W0, off_H1, off_W1, imode)
        # print(counts[30:34,30:34])
        # counts = get_counts(vid,stride0,ps,pt,
        #                     dilation,use_adj,reflect_bounds)
        # print(stack[0][0][0])
        # print(counts)
        # exit()
        # print(counts)
        # assert th.all(counts == vid.shape[0]).item()
        # counts = counts/(1.*vid.shape[0])
        # print(K)
        # print(inds[0,0,12*W+23-1])
        # print(inds[0,0,12*W+23])
        # print(inds[0,0,12*W+23+1])
        # print(counts)
        # print(th.where(counts==0))
        assert th.all(counts > 0).item()
        eps = 1e-10
        # counts = counts.view((B,HD,1,1,1,H,W))
        stack /= (counts.view((B,HD,1,1,1,H,W))+eps)
        # stack /= (counts+eps)
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
        ctx.off_H0,ctx.off_W0 = off_H0,off_W0
        ctx.off_H1,ctx.off_W1 = off_H1,off_W1
        ctx.itype_bwd = itype_bwd

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
        off_H0,off_W0 = ctx.off_H0,ctx.off_W0
        off_H1,off_W1 = ctx.off_H1,ctx.off_W1
        itype_bwd = ctx.itype_bwd
        ndim = ctx.ndim
        imode = get_imode(itype_bwd)

        # -- alloc --
        grad_vid = th.zeros_like(vid)
        grad_weights = th.zeros_like(weights)
        grad_stack = grad_stack.contiguous()
        if itype_bwd != "int": grad_inds = th.zeros_like(inds)
        else: grad_inds = th.zeros((1,)*5).to(inds.device).int()

        # -- view --
        # print("grad_vid.shape: ",grad_vid.shape)
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
        # grad_stack /= (counts.view((B,HD,1,1,1,H,W))+eps)
        # grad_stack = grad_stack / (counts+eps)
        # if imode == 0:
        #     inds = inds.int()

        # -- view --
        print(grad_vid.ndim,grad_weights.ndim,grad_inds.ndim,grad_stack.ndim)
        print(vid.ndim,weights.ndim,inds.ndim,stack.ndim,counts.ndim)

        stnls_cuda.non_local_stack_backward(
            grad_vid,grad_weights,grad_inds,grad_stack,
            vid,weights,inds,stack,counts,
            ps,pt,dilation,stride0,use_adj,reflect_bounds,
            off_H0,off_W0,off_H1,off_W1, imode)

        # -- info --
        # print("grad weights.")
        # print(grad_weights)

        # -- ensure original ndim --
        grad_vid = revert_ndim(grad_vid,ndim)

        # -- don't propogate "int" --
        if itype_bwd == "int": grad_inds = None
        # print(grad_stack.abs().mean(),grad_vid.abs().mean(),grad_weights.abs().mean())

        # print("stack [grad_weights.shape,grad_inds.shape]: ",grad_weights.shape)
        return grad_vid,grad_weights,grad_inds,None,None,\
            None,None,None,None,None,None,None,None,None,None

class NonLocalStack(th.nn.Module):

    def __init__(self,ps=7,stride0=4,pt=1,reflect_bounds=True,
                 dilation=1,use_adj=False,off_H0=0,off_W0=0,off_H1=0,off_W1=0,
                 itype_fwd="int",itype_bwd="int"):
        super().__init__()
        _vars = ["ps","stride0","pt","reflect_bounds","dilation","use_adj",
                 "off_H0","off_W0","off_H1","off_W1","itype_fwd","itype_bwd"]
        self._vars = _vars
        for var in _vars:
            setattr(self,var,eval(var))

    def forward(self, vid, weights, inds):
        inputs = [getattr(self,var) for var in self._vars]
        stack = non_local_stack.apply(vid, weights, inds, *inputs)
        return stack

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Python Dict API] stnls.tile.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ps":7,"stride0":4,"pt":1,"reflect_bounds":True,
             "dilation":1, "use_adj":False,
             "off_H0":0,"off_W0":0,"off_H1":0,"off_W1":0,
             "itype_fwd":"int","itype_bwd":"int"}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    cfg = extract_config(cfg)
    search = NonLocalStack(cfg.ps,cfg.stride0,cfg.pt,cfg.reflect_bounds,
                           cfg.dilation,cfg.use_adj,
                           cfg.off_H0,cfg.off_W0,cfg.off_H1,cfg.off_W1,
                           cfg.itype_fwd,cfg.itype_bwd)
    return search


