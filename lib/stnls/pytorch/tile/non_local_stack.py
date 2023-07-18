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
    print(counts.shape)
    return counts

class non_local_stack(th.autograd.Function):
    """

    Stack the non-local patches according to inds
    across the "ki"^th channel

    """

    @staticmethod
    def forward(ctx, vid, weights, inds,
                ps=7,stride0=4,pt=1,reflect_bounds=True,
                dilation=1,use_adj=False,off_H0=0,off_W0=0,off_H1=0,off_W1=0):

        # -- init --
        HD = inds.shape[1]
        K = inds.shape[3]
        q_start=0
        ndim = vid.ndim
        vid = ensure_ndim6(vid,HD)
        B,HD,T,F,H,W = vid.shape
        stack = th.zeros((B,HD,K,T,F,H,W),device=vid.device,dtype=th.float32)
        counts = th.zeros((H,W),device=vid.device,dtype=th.int32)

        # -- non-local stacking --
        vid = vid.contiguous()
        weights = weights.contiguous()
        inds = inds.contiguous()
        stnls_cuda.non_local_stack_forward(vid, weights, inds,
                                           stack, counts,
                                           ps, pt, dilation, stride0,
                                           use_adj, reflect_bounds, q_start,
                                           off_H0, off_W0, off_H1, off_W1)
        # print(counts[30:34,30:34])
        # counts = get_counts(vid,stride0,ps,pt,
        #                     dilation,use_adj,reflect_bounds)
        stack /= counts.view((1,1,1,1,1,H,W))

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
        ndim = ctx.ndim

        # -- alloc --
        grad_vid = th.zeros_like(vid)
        grad_weights = th.zeros_like(weights)
        grad_stack = grad_stack.contiguous()

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
        H,W = counts.shape
        # print(grad_stack[0,0,0,0,0,30:34,30:34])
        grad_stack = grad_stack / counts.view((1,1,1,1,1,H,W))
        # print(grad_stack[0,0,0,0,0,30:34,30:34])
        stnls_cuda.non_local_stack_backward(
            grad_vid,grad_weights,grad_stack,
            vid,weights,inds,stack,counts,
            ps,pt,dilation,stride0,use_adj,reflect_bounds,
            off_H0,off_W0,off_H1,off_W1)

        # -- ensure original ndim --
        grad_vid = revert_ndim(grad_vid,ndim)

        return grad_vid,grad_weights,None,None,None,\
            None,None,None,None,None,None,None,None

class NonLocalStack(th.nn.Module):

    def __init__(self,ps=7,stride0=4,pt=1,reflect_bounds=True,dilation=1,use_adj=False,
                 off_H0=0,off_W0=0,off_H1=0,off_W1=0):
        super().__init__()
        _vars = ["ps","stride0","pt","reflect_bounds","dilation","use_adj",
                 "off_H0","off_W0","off_H1","off_W1"]
        self._vars = _vars
        for var in _vars:
            setattr(self,var,eval(var))

    def forward(self, vid, weights, inds):
        inputs = [getattr(self,var) for var in self._vars]
        stack = non_local_stack.apply(vid, weights, inds, *inputs)
        return stack
