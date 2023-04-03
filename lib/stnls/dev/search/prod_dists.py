
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- softmax --
import torch.nn.functional as nnf

# -- cpp cuda kernel --
import stnls_cuda

# -- local --
from .search_utils import *

class ProdSearchWithHeadsFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, inds_exh, qstart, stride0,
                h0_off, w0_off, h1_off, w1_off,
                k, ps, pt, nheads, chnls,
                dilation=1,stride1=1,use_k=True,use_adj=True,
                reflect_bounds=True,search_abs=False,
                anchor_self=False,remove_self=False,
                nbwd=1,rbwd=True,exact=False):
        """
        vid0 = [B,T,C,H,W] or [B,H,T,C,H,W]
        inds = [B,H,Q,K_1,3]
        """

        # -- chw to hwc --
        # vid0 = rearrange(vid0,'b t c h w -> b t h w c')
        # vid1 = rearrange(vid1,'b t c h w -> b t h w c')

        # -- reshape with heads --
        dtype = vid0.dtype
        device = vid0.device
        assert vid0.ndim in [5], "Must be 5 dims."
        if vid0.ndim == 5:
            # c = vid0.shape[-1]
            c = vid0.shape[2]
            assert c % nheads == 0,"must be multiple of each other."
            # shape_str = 'b t h w (H c) -> b H t h w c'
            shape_str = 'b t (H c) h w -> b H t c h w'
            vid0 = rearrange(vid0,shape_str,H=nheads).contiguous()
            vid1 = rearrange(vid1,shape_str,H=nheads).contiguous()
        assert vid0.shape[1] == nheads
        assert vid1.shape[1] == nheads
        # vid0 = vid0.contiguous()
        # vid1 = vid1.contiguous()
        # B,H,t,h,w,c = vid0.shape
        B,H,t,c,h,w = vid0.shape
        vshape = (t,c,h,w)
        n_h0,n_w0 = get_num_img(vshape,stride0,ps,dilation)

        # -- allocs --
        B,H,Q,k0,_ = inds_exh.shape
        BHQ = B*H*Q
        dists_exh = -th.inf * th.ones((B,H,Q,k0),device=device,dtype=dtype)
        assert inds_exh.shape[0] == B
        assert inds_exh.shape[1] == H
        assert inds_exh.shape[2] == Q

        # -- allocates self --
        if anchor_self:
            self_dists = -th.inf * th.ones((B,H,Q),device=device,dtype=dtype)
        else:
            self_dists = -th.inf * th.ones((1,1,1),device=device,dtype=dtype)

        # -- forward --
        th.cuda.set_device(device)
        stnls_cuda.prod_dists(vid0, vid1, dists_exh, inds_exh, self_dists,
                             qstart, stride0, n_h0, n_w0,
                             h0_off, w0_off, h1_off, w1_off,
                             ps, pt, chnls, dilation, stride1, use_adj,
                             reflect_bounds, search_abs, anchor_self)

        # -- shape for next step --
        B,H,Q = dists_exh.shape[:3]
        dists_exh = dists_exh.view(B*H,Q,-1)#.contiguous()
        inds_exh = inds_exh.view(B*H,Q,-1,3)#.contiguous()
        # print(dists_exh)

        # -- remove self --
        if remove_self:
            dists_exh,inds_exh = run_remove_self_cuda(dists_exh,inds_exh,qstart,
                                                      stride0,n_h0,n_w0)

        # -- shape for next step --
        dists_exh = dists_exh.view(B*H*Q,-1)#.contiguous()
        inds_exh = inds_exh.view(B*H*Q,-1,3)#.contiguous()
        # dists_exh=dists_exh.view(B,H,Q,-1)#.contiguous()
        # inds_exh=inds_exh.view(B,H,Q,-1,3)#.contiguous()
        # print("dists_exh.shape:" ,dists_exh.shape)

        # -- topk --
        if use_k:
            dists,inds = allocate_rtn(B*H*Q,k,device,dtype)
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
        ctx.save_for_backward(inds,vid0,vid1)
        ctx.vid_shape = vid0.shape
        ctx.nheads = nheads
        ctx.qstart,ctx.stride0 = qstart,stride0
        ctx.ps,ctx.pt,ctx.dil = ps,pt,dilation
        ctx.reflect_bounds = reflect_bounds
        ctx.rbwd,ctx.exact = rbwd,exact
        ctx.use_adj,ctx.nbwd = use_adj,nbwd
        ctx.n_h0,ctx.n_w0 = n_h0,n_w0
        ctx.h0_off,ctx.w0_off = h0_off, w0_off
        ctx.h1_off,ctx.w1_off = h1_off, w1_off
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists,inds_no_grad):
        inds,vid0,vid1 = ctx.saved_tensors
        nheads = ctx.nheads
        vid_shape,nbwd = ctx.vid_shape,ctx.nbwd
        qstart,stride0 = ctx.qstart,ctx.stride0
        ps,pt,dil = ctx.ps,ctx.pt,ctx.dil
        rbwd = ctx.rbwd
        exact,use_adj = ctx.exact,ctx.use_adj
        reflect_bounds = ctx.reflect_bounds
        n_h0,n_w0 = ctx.n_h0,ctx.n_w0
        h0_off, w0_off = ctx.h0_off,ctx.w0_off
        h1_off, w1_off = ctx.h1_off,ctx.w1_off
        grad_vid0 = allocate_vid(vid_shape,grad_dists.device)
        grad_vid1 = allocate_vid(vid_shape,grad_dists.device)

        # -- allow for repeated exec --
        if nbwd == 1:
            stnls_cuda.prod_search_with_heads_backward(grad_vid0,grad_vid1,
                                                      vid0,vid1,
                                                      grad_dists,inds,
                                                      qstart,nheads,stride0,n_h0,n_w0,
                                                      h0_off, w0_off, h1_off, w1_off,
                                                      ps,pt,dil, use_adj,
                                                      reflect_bounds,rbwd,exact)
        else:
            for _ in range(nbwd):
                grad_vid0_i = allocate_vid(vid_shape,grad_dists.device)
                grad_vid1_i = allocate_vid(vid_shape,grad_dists.device)
                stnls_cuda.prod_search_with_heads_backward(grad_vid0_i,grad_vid1_i,
                                                          vid0,vid1,
                                                          grad_dists,inds,
                                                          qstart,nheads,stride0,
                                                          n_h0,n_w0,
                                                          h0_off, w0_off,
                                                          h1_off, w1_off,
                                                          ps,pt,dil,use_adj,
                                                          reflect_bounds,rbwd,exact)
                grad_vid0 += grad_vid0_i
                grad_vid1 += grad_vid1_i
            grad_vid0 /= nbwd
            grad_vid1 /= nbwd

        # -- finalize shape --
        grad_vid0 = rearrange(grad_vid0,'B H t c h w -> B t (H c) h w')
        grad_vid1 = rearrange(grad_vid1,'B H t c h w -> B t (H c) h w')

        # -- print stats --
        # print("grad_dists[min,max]: ",grad_dists.min().item(),grad_dists.max().item())
        # print("grad_vid0[min,max]: ",grad_vid0.min().item(),grad_vid0.max().item())
        # print("grad_vid1[min,max]: ",grad_vid1.min().item(),grad_vid1.max().item())

        return grad_vid0,grad_vid1,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None

class ProdDistsWithHeads(th.nn.Module):

    def __init__(self, k, ps, pt, nheads,
                 chnls=-1, dilation=1, stride0=1, stride1=1,
                 use_k=True, use_adj=True, reflect_bounds=True,
                 search_abs=False, nbwd=1, exact=False,
                 h0_off=0, w0_off=0, h1_off=0, w1_off=0,
                 remove_self=False, anchor_self=False, rbwd=True):
        super().__init__()
        self.k = k
        self.ps = ps
        self.pt = pt
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
        self.anchor_self = anchor_self
        self.remove_self = remove_self
        self.nbwd = nbwd
        self.exact = exact
        self.rbwd = rbwd

    def forward(self, vid0, inds, qstart=0, vid1=None):
        if vid1 is None: vid1 = vid0
        chnls = vid0.shape[-3]//self.nheads if self.chnls <= 0 else self.chnls
        k = inds.shape[-1] if self.k <= 0 else self.k
        return ProdSearchWithHeadsFunction.apply(vid0,vid1,
                                         inds,qstart,self.stride0,
                                         self.h0_off,self.w0_off,
                                         self.h1_off,self.w1_off,
                                         k,self.ps,self.pt,self.nheads,chnls,
                                         self.dilation,self.stride1,
                                         self.use_k,self.use_adj,
                                         self.reflect_bounds,self.search_abs,
                                         self.anchor_self,
                                         self.remove_self,
                                         self.nbwd,self.rbwd,self.exact)

    def flops(self,nsearch,T,C,H,W,inds_k):

        # -- init --
        flops = 0

        # -- unpack --
        vshape = (1,T,C,H,W)
        chnls = C//self.nheads if self.chnls <= 0 else self.chnls
        k = inds_k if self.k <= 0 else self.k
        nheads = self.nheads
        ps,pt = self.ps,self.pt

        # -- compute search --
        nrefs_hw = ((H-1)//self.stride0+1) * ((W-1)//self.stride0+1)
        nrefs = T * nheads * nrefs_hw
        flops_per_search = chnls * ps * ps * pt
        search_flops = nrefs * nsearch * flops_per_search
        flops += search_flops

        # -- compute top-k --
        if self.use_k:
            sort_flops = nrefs * (nsearch * np.log(nsearch))
            flops += sort_flops

        return flops
