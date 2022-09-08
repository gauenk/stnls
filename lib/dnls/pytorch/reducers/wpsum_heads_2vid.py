"""

Directly accumulate in video

"""

# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda

from dnls.utils.timer import ExpTimer
from dnls.utils.inds import get_nums_hw


def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(inds,ps,pt,c,nheads):
    device = inds.device
    nq,k = inds.shape[0],nheads
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

def allocate_iunfold_patches(nq,k,ps,pt,c,device):
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class WpSumHeadsFunction2Vid(th.autograd.Function):
    # [video -> patches] @ inds

    # -- static video since it is the same --
    # vid = None

    @staticmethod
    def forward(ctx, vid, dists, inds, qstart, ps, pt=1,
                h_off=0,w_off=0,stride=1,dilation=1,adj=0,
                reflect_bounds=True,only_full=True,exact=False):
        """
        vid = [T,C,H,W]
        inds = [NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        """
        # if WpSumFunction.vid is None: WpSumFunction.vid = vid
        nheads = dists.shape[-1]
        t,c,h,w = vid.shape
        n_h,n_w = get_nums_hw(vid.shape,stride,ps,dilation,"pad_same")
        vid_fill = th.zeros((t,nheads,c,h,w),device=vid.device,dtype=vid.dtype)
        dnls_cuda.wpsum_heads_2vid_forward(vid, vid_fill, dists, inds,
                                           h_off,w_off,qstart,n_h,n_w,
                                           ps,pt,stride,dilation,adj,
                                           reflect_bounds,only_full)
        # print("dists._version: ",dists._version)
        # print("inds._version: ",inds._version)
        ctx.save_for_backward(dists,inds,vid)
        ctx.qstart = qstart
        ctx.ps,ctx.pt = ps,pt
        ctx.vid_shape = vid.shape
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.h_off = h_off
        ctx.w_off = w_off
        ctx.adj = adj
        ctx.reflect_bounds = reflect_bounds
        ctx.exact = exact

        return vid_fill

    @staticmethod
    def backward(ctx, grad_vid_fill):
        dists,inds,vid = ctx.saved_tensors
        # vid = WpSumFunction.vid
        qstart = ctx.qstart
        ps,pt = ctx.ps,ctx.pt
        vid_shape = ctx.vid_shape

        h_off = ctx.h_off
        w_off = ctx.w_off
        stride = ctx.stride
        dilation = ctx.dilation
        adj = ctx.adj
        reflect_bounds = ctx.reflect_bounds
        exact = ctx.exact
        n_h,n_w = get_nums_hw(vid.shape,stride,ps,dilation,"pad_same")

        # -- start timer --
        # timer = ExpTimer()
        # timer.start("wpsum_heads_bwd")

        # -- gradient for video --
        grad_vid = allocate_vid(vid_shape,grad_patches.device)
        dnls_cuda.wpsum_heads_2vid_backward_vid(grad_vid,grad_vid_fill,
                                                dists,inds,h_off,w_off,
                                                qstart,n_h,n_w,
                                                ps,pt,stride,dilation,adj,
                                                reflect_bounds,exact)

        # -- gradient for dists --
        grad_dists = th.zeros_like(dists)
        dnls_cuda.wpsum_heads_2vid_backward_dists(grad_dists,grad_patches,
                                                  vid,inds,h_off,w_off,
                                                  qstart,n_h,n_w,
                                                  ps,pt,stride,dilation,adj,
                                                  reflect_bounds,exact)

        # -- stop timer --
        # th.cuda.synchronize()
        # timer.stop("wpsum_bwd")
        # print(timer)

        return grad_vid,grad_dists,None,None,None,None,None,None,None,None,None

class WeightedPatchSumHeads2Vid(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, pt=1, h_off=0, w_off=0,
                 stride=1, dilation=1, adj=0,
                 reflect_bounds = True, only_full=True, exact=False):
        super(WeightedPatchSumHeads2Vid, self).__init__()
        self.ps = ps
        self.pt = pt
        self.stride = int(stride)
        self.dilation = int(dilation)

        self.h_off = h_off
        self.w_off = w_off

        self.adj = int(adj)
        self.only_full = only_full
        self.reflect_bounds = reflect_bounds
        self.exact = exact

    def forward(self, vid, dists, inds, qstart):

        # -- exec --
        fvid = WpSumHeadsFunction2Vid.apply(vid,dists,inds,qstart,
                                            self.ps,self.pt,self.h_off,self.w_off,
                                            self.stride,self.dilation,self.adj,
                                            self.reflect_bounds, self.only_full,
                                            self.exact)

        # -- final shape --
        _,_,nheads = dists.shape
        t,c,h,w = vid.shape
        fvid = fvid.view(t,nheads,c,h,w)

        return fvid

