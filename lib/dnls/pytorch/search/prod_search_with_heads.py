
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- softmax --
import torch.nn.functional as nnf

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
                full_ws=False,anchor_self=False,
                use_self=False,remove_self=False,
                nbwd=1,rbwd=True,exact=False):
        """
        vid0 = [B,T,C,H,W] or [B,H,T,C,H,W]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
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
        nqueries = t*n_h0*n_w0 if nqueries <= 0 else nqueries
        Q = nqueries

        # -- allocs --
        BHQ = B*H*Q
        st = min(2*wt+1,t)
        dists_exh,inds_exh = allocate_exh_prod(BHQ,st,ws_h,ws_w,device,dtype)
        dists_exh = dists_exh.view(B,H,Q,-1,ws_h,ws_w)
        inds_exh = inds_exh.view(B,H,Q,-1,ws_h,ws_w,3)

        # -- allocates self --
        assert use_self == anchor_self
        if anchor_self:
            self_dists = th.zeros((B,H,Q),device=device,dtype=dtype)
        else:
            self_dists = -th.inf * th.ones((1,1,1),device=device,dtype=dtype)

        # -- pre-computed search offsets --
        tranges,n_tranges,min_tranges = create_frame_range(t,wt,wt,pt,device)

        # -- viz --
        # print("prod_search_with_heads.")
        # print("vid0.shape: " ,vid0.shape)
        # print("vid1.shape: " ,vid1.shape)
        # print("fflow.shape: " ,fflow.shape)
        # print("bflow.shape: " ,bflow.shape)
        # print("dists_exh.shape: " ,dists_exh.shape)
        # print("inds_exh.shape: " ,inds_exh.shape)

        # -- setup flows --
        gpuid = th.cuda.current_device()
        fflow = fflow.to(device).type(dtype)
        bflow = bflow.to(device).type(dtype)
        # fflow = rearrange(fflow,'b t c h w -> b t h w c').contiguous()
        # bflow = rearrange(bflow,'b t c h w -> b t h w c').contiguous()

        # -- forward --
        th.cuda.set_device(device)
        dnls_cuda.prod_search_with_heads_forward(vid0, vid1, fflow, bflow,
                                                 dists_exh, inds_exh, self_dists,
                                                 qstart, stride0, n_h0, n_w0,
                                                 h0_off, w0_off, h1_off, w1_off,
                                                 ps, pt, ws_h, ws_w,
                                                 wt, chnls, dilation, stride1,
                                                 use_adj,
                                                 reflect_bounds, search_abs, full_ws,
                                                 anchor_self, use_self, tranges,
                                                 n_tranges, min_tranges)
        # th.cuda.synchronize()

        # -- shape for next step --
        B,H,Q = dists_exh.shape[:3]
        dists_exh=dists_exh.view(B*H,Q,-1)#.contiguous()
        inds_exh=inds_exh.view(B*H,Q,-1,3)#.contiguous()

        # -- remove self --
        if remove_self:
            dists_exh,inds_exh = run_remove_self_cuda(dists_exh,inds_exh,qstart,
                                                      stride0,n_h0,n_w0)

        # -- shape for next step --
        dists_exh = dists_exh.view(B*H*Q,-1)#.contiguous()
        inds_exh = inds_exh.view(B*H*Q,-1,3)#.contiguous()

        # -- topk --
        if use_k:
            print("[prod.] running top-k.")
            dists,inds = allocate_rtn(B*H*Q,k,device,dtype)
            topk_with_anchor(dists_exh,inds_exh,dists,inds,self_dists,anchor_self)
            # get_topk_prod(dists_exh,inds_exh,dists,inds)
        else:
            dists,inds = dists_exh,inds_exh

        # -- fill nans --
        args = th.where(th.isnan(dists))
        dists[args] = -th.inf # fix nan

        # -- fill if anchored --
        # if anchor_self:
        #     raise ValueError("Still unknown how to fix the 'self' position.")
            # args = th.where(dists == th.inf)
            # dists[args] = 0. # not the inner product value

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

        # -- ensure contiguous --
        grad_dists = grad_dists.contiguous()
        inds = inds.contiguous()

        # -- allow for repeated exec --
        bwd_fxn = dnls_cuda.prod_search_with_heads_backward
        if nbwd == 1:
            bwd_fxn(grad_vid0,grad_vid1,
                    vid0,vid1,
                    grad_dists,inds,
                    qstart,nheads,stride0,
                    n_h0,n_w0,
                    h0_off, w0_off, h1_off, w1_off,
                    ps,pt,dil, use_adj,
                    reflect_bounds,rbwd,exact)
        else:
            for _ in range(nbwd):
                grad_vid0_i = allocate_vid(vid_shape,grad_dists.device)
                grad_vid1_i = allocate_vid(vid_shape,grad_dists.device)
                bwd_fxn(grad_vid0_i,grad_vid1_i,
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
            None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None

class ProdSearchWithHeads(th.nn.Module):

    def __init__(self, fflow, bflow, k, ps, pt, ws, wt, nheads,
                 chnls=-1, dilation=1, stride0=1, stride1=1,
                 use_k=True, use_adj=True, reflect_bounds=True,
                 search_abs=False, full_ws = False, nbwd=1, exact=False,
                 h0_off=0,w0_off=0,h1_off=0,w1_off=0,remove_self=False,
                 anchor_self=False,use_self=False,rbwd=True):
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
        self.use_self = use_self
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
        ndim = len(vshape)
        vshape = vshape[-4:] # (t,c,h,w) NOT (B,H,t,c,h,w)
        t,c,h,w = vshape
        assert ndim in [5,6],"Must be 5 or 6 dim."

        # -- compute number of searchable patches --
        n_h,n_w = get_num_img(vshape,self.stride1,self.ps,self.dilation)
        ws_h,ws_w = ws,ws
        if ws == -1: ws_h,ws_w = n_h,n_w
        if k == -1: k = ws**2 * (2*wt + 1)
        if chnls <= 0:
            if ndim == 5: chnls = c//self.nheads
            else: chnls = c
        if ndim == 5:
            assert c % self.nheads == 0,"must be multiple of each other."
        return ws_h,ws_w,wt,k,chnls

    def update_flow(self,vshape,device,flows=None):
        b,t,c,h,w = vshape
        zflow = th.zeros((b,t,2,h,w),device=device)
        noflow = flows is None
        # print("noflow: ",noflow)
        self.fflow = zflow if noflow else flows.fflow
        self.bflow = zflow if noflow else flows.bflow

    def _update_flow(self,vshape,device):
        vshape = vshape # (t,c,h,w) NOT (H,t,c,h,w)
        assert len(vshape) in [5]
        # if len(vshape) == 5: b,t,c,h,w = vshape
        # else: b,h,t,c,h,w = vshape
        b,t,c,h,w = vshape
        zflow = th.zeros((b,t,2,h,w),device=device)
        if self.fflow is None: self.fflow = zflow
        if self.bflow is None: self.bflow = zflow
        # print("vshape: ",vshape)
        # print("self.fflow.shape: ",self.fflow.shape)
        for i in [0,1,3,4]:
            assert self.fflow.shape[i] == vshape[i],"Must be equal size: %d" % i
            assert self.bflow.shape[i] == vshape[i],"Must be equal size: %d" % i

    def set_flows(self,flows,vid):
        (b,t,c,h,w),device = vid.shape,vid.device
        zflow = th.zeros((b,t,2,h,w),device=device,dtype=th.int32)
        zflow = th.zeros((b,t,2,h,w),device=device,dtype=th.int32)
        noflow = flows is None
        self.fflow = zflow if noflow else flows.fflow
        self.bflow = zflow if noflow else flows.bflow

    def forward(self, vid0, vid1, qstart=0, nqueries=-1):
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
                                         self.use_self, self.remove_self,
                                         self.nbwd,self.rbwd,self.exact)

    def flops(self,T,C,H,W):

        # -- unpack --
        vshape = (1,T,C,H,W)
        ws_h,ws_w,wt,k,chnls = self._get_args(vshape)
        nheads = self.nheads
        ps,pt = self.ps,self.pt
        # print("C,c,nheads: ",C,chnls,nheads)

        # -- compute search --
        nrefs_hw = ((H-1)//self.stride0+1) * ((W-1)//self.stride0+1)
        nrefs = T * nheads * nrefs_hw
        nsearch = ws_h * ws_w * (2*wt+1)
        flops_per_search = 2 * chnls * ps * ps * pt
        search_flops = nrefs * nsearch * flops_per_search
        flops = search_flops

        # -- compute top-k --
        if self.use_k:
            sort_flops = nrefs * (nsearch * np.log(nsearch))
            flops += sort_flops

        return flops
