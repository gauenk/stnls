
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- softmax/unfold --
import torch.nn.functional as nnf
from torch.nn.functional import unfold,pad

# -- cpp cuda kernel --
import dnls_cuda

# -- local --
from .search_utils import *

@cache.lur
def compute_search_inds(B,Q,H,ws,full_ws):
    inds = th.zeros((B,Q,H,ws*ws,3))
    return inds


class ProdSearchPatchesWithHeadsFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, patches0, patches1,
                k, ws_h, ws_w, nheads, chnls,
                stride0, stride1, dilation=1,
                h0_off=0,w0_off=0,h1_off=0,w1_off=0,
                use_k=True, use_adj=True, search_abs=False,
                full_ws=False, anchor_self=False, remove_self=False):
        """
        patches0 = [B,Q,HD,C,H,W]
        ws = search Window Spatial (ws)
        """
        # -- reshape with heads --
        dtype = patches0.dtype
        device = patches0.device
        assert patches0.ndim in [6], "Must be 6 dims."
        assert patches0.shape[2] == nheads
        assert patches1.shape[2] == nheads
        B,Q,H,c,h,w = patches0.shape

        # -- allocs --
        BQH = B*Q*H
        dists_exh,inds_exh = allocate_exh_prod(BQH,1,ws,ws,device,dtype)
        dists_exh = dists_exh.view(B,Q,H,-1,ws,ws)
        inds_exh = inds_exh.view(B,Q,H,-1,ws,ws,3)

        # -- pre-computed search offsets --
        access_inds = compute_search_inds(inds, stride0, stride1, use_adj,
                                          h0_off,w0_off,h1_off,w1_off)

        # -- viz --
        # print("patches0.shape: " ,patches0.shape)
        # print("patches1.shape: " ,patches1.shape)
        # print("dists_exh.shape: " ,dists_exh.shape)
        # print("inds_exh.shape: " ,inds_exh.shape)

        # -- forward --
        gpuid = th.cuda.current_device()
        th.cuda.set_device(device)
        dnls_cuda.prod_search_patches_with_heads_forward(patches0, patches1,
                                                         dists_exh, inds_exh,
                                                         access_inds,
                                                         chnls, dilation,
                                                         anchor_self)
        # th.cuda.synchronize()

        # -- shape for next step --
        B,H,Q = dists_exh.shape[:3]
        dists_exh=dists_exh.view(B*H,Q,-1)#.contiguous()
        inds_exh=inds_exh.view(B*H,Q,-1,3)#.contiguous()

        # -- remove self --
        if remove_self:
            dists_exh,inds_exh = run_remove_self_cuda(dists_exh,inds_exh,
                                                      stride0,n_h0,n_w0)

        # -- shape for next step --
        dists_exh = dists_exh.view(B*H*Q,-1)#.contiguous()
        inds_exh = inds_exh.view(B*H*Q,-1,3)#.contiguous()
        self_dists = self_dists.view(B*H*Q)
        # dists_exh=dists_exh.view(B,H,Q,-1)#.contiguous()
        # inds_exh=inds_exh.view(B,H,Q,-1,3)#.contiguous()

        # -- topk --
        if use_k:
            dists,inds = allocate_rtn(B*H*Q,k,device,dtype)
            # get_topk_prod(dists_exh,inds_exh,dists,inds)
            topk_with_anchor(dists_exh,inds_exh,dists,inds,self_dists,anchor_self)
        else:
            dists,inds = dists_exh,inds_exh

        # -- fill nans --
        args = th.where(th.isnan(dists))
        dists[args] = -th.inf # fix nan

        # -- fill if anchored --
        # if anchor_self:
        #     raise ValueError("Still unknown how to fix the 'self' position.")
        #     # args = th.where(dists == th.inf)
        #     # dists[args] = 0. # not the inner product value

        # -- final shape with heads -
        dists = dists.view(B,H,Q,-1)
        inds = inds.view(B,H,Q,-1,3)

        # -- for backward --
        ctx.save_for_backward(inds,patches0,patches1)
        ctx.patches_shape = patches0.shape
        ctx.nheads = nheads
        ctx.stride0 = stride0
        ctx.ps,ctx.pt,ctx.dil = ps,pt,dilation
        ctx.use_adj = use_adj
        ctx.n_h0,ctx.n_w0 = n_h0,n_w0
        ctx.h0_off,ctx.w0_off = h0_off, w0_off
        ctx.h1_off,ctx.w1_off = h1_off, w1_off
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds_is_none):
        inds,patches0,patches1 = ctx.saved_tensors
        nheads = ctx.nheads
        patches_shape = ctx.patches_shape
        stride1 = ctx.stride1
        ps,pt,dil = ctx.ps,ctx.pt,ctx.dil
        use_adj = ctx.use_adj
        n_h0,n_w0 = ctx.n_h0,ctx.n_w0
        h0_off, w0_off = ctx.h0_off,ctx.w0_off
        h1_off, w1_off = ctx.h1_off,ctx.w1_off
        grad_patches0 = allocate_patches(patches_shape,grad_dists.device)
        grad_patches1 = allocate_patches(patches_shape,grad_dists.device)

        # -- allow for repeated exec --
        dnls_cuda.prod_search_with_heads_backward(grad_patches0,grad_patches1,
                                                  patches0,patches1,
                                                  grad_dists,inds,
                                                  nheads,stride1,n_h0,n_w0,
                                                  h0_off, w0_off, h1_off, w1_off,
                                                  ps,pt,dil, use_adj)
        # -- finalize shape --
        grad_patches0 = rearrange(grad_patches0,'B H t c h w -> B t (H c) h w')
        grad_patches1 = rearrange(grad_patches1,'B H t c h w -> B t (H c) h w')

        # -- print stats --
        # print("grad_dists[min,max]: ",grad_dists.min().item(),grad_dists.max().item())
        # print("grad_patches0[min,max]: ",grad_patches0.min().item(),grad_patches0.max().item())
        # print("grad_patches1[min,max]: ",grad_patches1.min().item(),grad_patches1.max().item())

        return grad_patches0,grad_patches1,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None

class ProdSearchPatchesWithHeads(th.nn.Module):

    def __init__(self, k, ps, pt, ws, nheads,
                 chnls=-1, dilation=1, stride0=1, stride1=1,
                 use_k=True, use_adj=True, reflect_bounds=True,
                 search_abs=False, full_ws = False,
                 h0_off=0,w0_off=0,h1_off=0,w1_off=0,remove_self=False,
                 anchor_self=False,rbwd=True):
        super().__init__()
        self.k = k
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.nheads = nheads
        self.chnls = chnls
        self.dilation = dilation
        self.stride0 = stride0
        self.stride1 = stride1
        self.h0_off = h0_off
        self.w0_off = w0_off
        self.h1_off = h1_off
        self.w1_off = w1_off
        self.use_adj = use_adj
        self.use_k = use_k
        self.reflect_bounds = reflect_bounds
        self.search_abs = search_abs
        self.full_ws = full_ws
        self.anchor_self = anchor_self
        self.remove_self = remove_self
        self.nbwd = nbwd

    def query_batch_info(self,vshape,only_full=True,use_pad=True):
        n_h,n_w = get_num_img(vshape,self.stride0,self.ps,self.dilation,
                              only_full,use_pad)
        return n_h,n_w

    def _get_args(self,vshape):
        # -- unpack --
        ws,k,chnls = self.ws,self.k,self.chnls
        assert len(vshape) == 4,"must be 4."

        # -- compute number of searchable patches --
        n_h,n_w = get_num_img(vshape,self.stride1,self.ps,self.dilation)
        ws_h,ws_w = ws,ws
        pad_ws = True
        if ws == -1:
            ws_h,ws_w = n_h,n_w
            pad_ws = False
        if k == -1: k = ws**2
        if chnls <= 0:
            if ndim == 5: chnls = c//self.nheads
            else: chnls = c
        if ndim == 5:
            assert c % self.nheads == 0,"must be multiple of each other."
        return ws,k,chnls,pad_ws

    def unfold(self,vid,ps,ws_h,ws_w,stride,dil,pad_ws,rbounds):
        if self.full_ws:
            vid_pad = vid
        else:
            pads = 2*[pad_ws*(ws_h//2)+ps//2,pad_ws*(ws_w//2)+ps//2]
            mode = "reflect" if rbounds else "zero"
            vid_pad = pad(vid,pads,mode=mode)
        patches = unfold(vid_pad,(ps,ps),stride=stride,dilation=dil)
        return patches

    def forward(self, vid0, vid1=None):
        if vid1 is None: vid1 = vid0
        ws_h,ws_w,k,chnls,pad_ws = self._get_args(vid0.shape)
        patches0 = unfold(vid0,self.ps,ws_h,ws_w,self.stride0,
                          self.dilation,pad_ws,self.reflect_bounds)
        patches1 = unfold(vid1,self.ps,ws_h,ws_w,self.stride1,
                          self.dilation,pad_ws,self.reflect_bounds)
        return ProdSearchPatchesWithHeadsFunction.apply(patches0,patches1,
                                                        k,ws_h,ws_w,
                                                        self.nheads,chnls,
                                                        self.stride0,self.stride1,
                                                        self.dilation,
                                                        self.h0_off,self.w0_off,
                                                        self.h1_off,self.w1_off,
                                                        self.use_k,self.use_adj,
                                                        self.search_abs,
                                                        self.full_ws,self.anchor_self,
                                                        self.remove_self)

    def flops(self,T,C,H,W):

        # -- init --
        flops = 0

        # -- unpack --
        vshape = (1,T,C,H,W)
        ws_h,ws_w,k,chnls,pad_ws = self._get_args(vid0.shape)
        nheads = self.nheads
        ps,pt = self.ps,self.pt

        # -- compute search --
        nrefs_hw = ((H-1)//self.stride0+1) * ((W-1)//self.stride0+1)
        nrefs = T * nheads * nrefs_hw
        nsearch = ws_h * ws_w
        flops_per_search = chnls * ps * ps * pt
        search_flops = nrefs * nsearch * flops_per_search
        flops += search_flops

        # -- compute top-k --
        if self.use_k:
            sort_flops = nrefs * (nsearch * np.log(nsearch))
            flops += sort_flops

        return flops
