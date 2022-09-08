
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- softmax --
import torch.nn.functional as nnf

# -- padding --
from dnls.utils.pads import comp_pads

# -- cpp cuda kernel --
import dnls_cuda

# -- local --
from .search_utils import *

class ProdSearchWithHeadsFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, fflow, bflow,
                qstart, nqueries, stride0,
                h0_off, w0_off, h1_off, w1_off,
                k, ps, pt, ws_h, ws_w, wt, nheads, chnls,
                dilation=1,stride1=1,use_k=True,use_adj=True,
                reflect_bounds=True,search_abs=False,
                full_ws=False,anchor_self=False,remove_self=False,
                nbwd=1,rbwd=True,exact=False):
        """
        vid0 = [T,C,H,W]
        qinds = [NumQueries,K,3]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """

        # -- reshape with heads --
        device = vid0.device
        if vid0.ndim == 4:
            c = vid0.shape[1]
            assert c % nheads == 0,"must be multiple of each other."
            vid0 = rearrange(vid0,'t (H c) h w -> H t c h w',H=nheads).contiguous()
            vid1 = rearrange(vid1,'t (H c) h w -> H t c h w',H=nheads).contiguous()
        assert vid0.shape[0] == nheads
        assert vid1.shape[0] == nheads
        H,t,c,h,w = vid0.shape
        n_h0,n_w0 = get_num_img(vid0.shape[1:],stride0,ps,dilation)

        # -- allocs --
        B = nqueries*nheads
        dists_exh,inds_exh = allocate_exh_prod(B,wt,ws_h,ws_w,device)
        dists_exh = dists_exh.view(nheads,nqueries,-1,ws_h,ws_w)
        inds_exh = inds_exh.view(nheads,nqueries,-1,ws_h,ws_w,3)

        # -- pre-computed search offsets --
        tranges,n_tranges,min_tranges = create_frame_range(t,wt,wt,pt,device)

        # -- forward --
        gpuid = th.cuda.current_device()
        fflow = fflow.to(device)
        bflow = bflow.to(device)
        th.cuda.set_device(device)
        dnls_cuda.prod_search_with_heads_forward(vid0, vid1, fflow, bflow,
                                                 dists_exh, inds_exh,
                                                 qstart, nqueries, nheads, stride0,
                                                 n_h0, n_w0,
                                                 h0_off, w0_off, h1_off, w1_off,
                                                 ps, pt, ws_h, ws_w,
                                                 wt, chnls, dilation, stride1, use_adj,
                                                 reflect_bounds, search_abs, full_ws,
                                                 anchor_self, tranges,
                                                 n_tranges, min_tranges)
        th.cuda.synchronize()

        # -- shape for output --
        H,b = dists_exh.shape[:2]
        dists_exh=dists_exh.view(H*b,-1)#.contiguous()
        inds_exh=inds_exh.view(H*b,-1,3)#.contiguous()

        # -- remove self --
        if remove_self:
            dists_exh,inds_exh = run_remove_self_cuda(dists_exh,inds_exh,qstart,
                                                      stride0,n_h0,n_w0)

        # -- topk --
        if use_k:
            HB = dists_exh.shape[0]
            dists,inds = allocate_rtn(HB,k,device)
            get_topk_prod(dists_exh,inds_exh,dists,inds)
        else:
            args = th.where(th.isnan(dists_exh))
            dists_exh[args] = -th.inf # fix nan
            dists,inds = dists_exh,inds_exh

        # -- fill if anchored --
        if anchor_self:
            raise ValueError("Still unknown how to fix the 'self' position.")
            # args = th.where(dists == th.inf)
            # dists[args] = 0. # not the inner product value

        # -- shape with heads -
        dists = dists.view(H,b,-1)
        inds = inds.view(H,b,-1,3)

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
    def backward(ctx, grad_dists, grad_inds_is_none):
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
            dnls_cuda.prod_search_with_heads_backward(grad_vid0,grad_vid1,
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
                dnls_cuda.prod_search_with_heads_backward(grad_vid0_i,grad_vid1_i,
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
        grad_vid0 = rearrange(grad_vid0,'H t c h w -> t (H c) h w')
        grad_vid1 = rearrange(grad_vid1,'H t c h w -> t (H c) h w')

        return grad_vid0,grad_vid1,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None

class ProdSearchWithHeads(th.nn.Module):

    def __init__(self, fflow, bflow, k, ps, pt, ws, wt, nheads,
                 chnls=-1, dilation=1, stride0=1, stride1=1,
                 use_k=True, use_adj=True, reflect_bounds=True,
                 search_abs=False, full_ws = False, nbwd=1, exact=False,
                 h0_off=0,w0_off=0,h1_off=0,w1_off=0,remove_self=False,
                 anchor_self=False,rbwd=True):
        super().__init__()
        self.k = k
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.wt = wt
        self.nheads = nheads
        self.fflow = fflow
        self.bflow = bflow
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
        self.exact = exact
        self.rbwd = rbwd

    def query_batch_info(self,vshape,only_full=True,use_pad=True):
        n_h,n_w = get_num_img(vshape,self.stride0,self.ps,self.dilation,
                              only_full,use_pad)
        return n_h,n_w

    def window_attn_mod(self,dists,rel_pos,mask,vshape):
        t,c,h,w = vshape
        wsize = 8#self.ws
        # print(self.stride0,self.ps,self.ws,self.dilation,vshape)
        # exit(0)
        n_h,n_w = get_num_img(vshape,self.stride0,self.ps,self.dilation,False,False)
        nh_r = n_h//wsize
        shape_str = 'H (t h w) d2 -> H t h w d2'
        dists = rearrange(dists,shape_str,h=n_h,t=t)
        shape_str = 'H t (nh rh) (nw rw) d2 -> (t nh nw) H (rh rw) d2'
        dists = rearrange(dists,shape_str,rh=wsize,rw=wsize)
        N,H,R,D = dists.shape

        if not(rel_pos is None):
            ratio = dists.shape[-1] // rel_pos.shape[-1]
            # print("dists.shape: ",dists.shape)
            # print("rel_pos.shape: ",rel_pos.shape)
            rel_pos = repeat(rel_pos,'a b c -> a b (c r)',r=ratio)
            dists = dists + rel_pos.unsqueeze(0)

        if not(mask is None):
            ratio = dists.shape[-1] // mask.shape[-1]
            # print(t,n_h,nh_r,self.stride0,self.ps,self.ws)
            dists = rearrange(dists,'(t n) H d1 d2 -> t n H d1 d2',t=t)
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            mshape = mask.shape
            mask = mask.unsqueeze(1).unsqueeze(0)
            # print("mask: ",dists.shape,mshape,mask.shape)#,ratio)
            # print("dists.shape: ",dists.shape)
            # print("masks.shape: ",mask.shape)
            dists = dists + mask
            dists = rearrange(dists,'t n H d1 d2 -> (t n) H d1 d2')

        dists = nnf.softmax(dists,-1)
        shape_str = '(t nh nw) H (rh rw) d2 -> H t (nh rh) (nw rw) d2'
        dists = rearrange(dists,shape_str,rh=wsize,rw=wsize,nh=nh_r,t=t)
        dists = rearrange(dists,'H t h w d2 -> H (t h w) d2')

        return dists

    def _get_args(self,vshape):
        # -- unpack --
        ws,wt,k,chnls = self.ws,self.wt,self.k,self.chnls
        vshape = vshape[-4:] # (t,c,h,w) NOT (H,t,c,h,w)
        t,c,h,w = vshape

        # -- compute number of searchable patches --
        n_h,n_w = get_num_img(vshape,self.stride1,self.ps,self.dilation)
        ws_h,ws_w = ws,ws
        if ws == -1: ws_h,ws_w = n_h,n_w
        if k == -1: k = ws**2 * (2*wt + 1)
        if chnls <= 0: chnls = c//self.nheads
        assert c % self.nheads == 0,"must be multiple of each other."
        return ws_h,ws_w,wt,k,chnls

    def _update_flow(self,vshape,device):
        vshape = vshape[-4:] # (t,c,h,w) NOT (H,t,c,h,w)
        t,c,h,w = vshape
        t,c,h,w = vshape
        zflow = th.zeros((t,2,h,w),device=device)
        if self.fflow is None: self.fflow = zflow
        if self.bflow is None: self.bflow = zflow
        for i in [0,2,3]:
            assert self.fflow.shape[i] == vshape[i],"Must be equal size: %d" % i
            assert self.bflow.shape[i] == vshape[i],"Must be equal size: %d" % i

    def forward(self, vid0, qstart, nqueries, vid1=None):
        if vid1 is None: vid1 = vid0
        self._update_flow(vid0.shape,vid0.device)
        ws_h,ws_w,wt,k,chnls = self._get_args(vid0.shape)
        return ProdSearchWithHeadsFunction.apply(vid0,vid1,
                                         self.fflow,self.bflow,
                                         qstart,nqueries,self.stride0,
                                         self.h0_off,self.w0_off,
                                         self.h1_off,self.w1_off,
                                         k,self.ps,self.pt,ws_h,ws_w,wt,
                                         self.nheads,chnls,
                                         self.dilation,self.stride1,
                                         self.use_k,self.use_adj,
                                         self.reflect_bounds,self.search_abs,
                                         self.full_ws,self.anchor_self,
                                         self.remove_self,
                                         self.nbwd,self.rbwd,self.exact)
