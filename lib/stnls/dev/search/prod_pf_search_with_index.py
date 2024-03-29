
# -- python --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- utils --
from ...utils.timer import ExpTimer

# -- cpp cuda kernel --
import stnls_cuda

# -- local --
from .search_utils import *

class ProductPfSearchFunction_with_index(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, fflow, bflow, qstart, nqueries,
                k, ps, pt, ws_h, ws_w, wt, chnls,
                stride0, stride1, dilation,lam,
                search_abs, reflect_bounds, use_adj, use_k,
                h0_off, w0_off, h1_off, w1_off,
                anchor_self, use_self, remove_self, full_ws, nbwd,
                rbwd, exact):
        """
        vid = [T,C,H,W]
        ws = xsearch Window Spatial (ws)
        wt = xsearch Window Time (wt)
        """

        # -- unpack --
        dtype = vid0.dtype
        device = vid0.device
        bsize,t,c,h,w = vid0.shape
        n_h0,n_w0 = get_num_img(vid0[0].shape,stride0,ps,dilation)
        nqueries = t*n_h0*n_w0 if nqueries <= 0 else nqueries
        nq = nqueries
        Q = nqueries
        # print("k, ps, pt, ws_h, ws_w, wt: ",k, ps, pt, ws_h, ws_w, wt)
        # print("chnls, stride0, stride1, dilation,lam: ",
        #       chnls, stride0, stride1, dilation,lam)
        # print("search_abs, reflect_bounds, use_adj: ",
        #       search_abs, reflect_bounds, use_adj)
        # print("use_k, h0_off, w0_off, h1_off, w1_off: ",
        #       use_k, h0_off, w0_off, h1_off, w1_off)
        # print("remove_self, full_ws, nbwd, rbwd, exact: ",
        #       remove_self, full_ws, nbwd, rbwd, exact)

        # -- allocs --
        # bufs = allocate_bufs(nq,t,ws_h,ws_w,wt,device)
        B,Q = bsize,nqueries
        BQ = B*Q
        nframes = vid0.shape[1]
        st = min(2*wt+1,nframes)
        dists_exh,inds_exh = allocate_exh_prod(BQ,st,ws_h,ws_w,device,dtype)
        dists_exh = dists_exh.view(B,Q,-1,ws_h,ws_w)
        inds_exh = inds_exh.view(B,Q,-1,ws_h,ws_w,3)

        # -- alloc self --
        assert use_self == anchor_self
        if use_self:
            self_dists = -th.inf * th.ones((B,Q),device=device,dtype=dtype)
        else:
            self_dists = -th.inf * th.ones((1,1),device=device,dtype=dtype)

        # -- pre-computed xsearch offsets --
        tranges,n_tranges,min_tranges = create_frame_range(t,wt,wt,pt,device)
        # print(tranges)

        # -- testing [delete me] --
        # print("prod_pf_search_with_index.")
        # print("vid0.shape: ",vid0.shape)
        # print("vid1.shape: ",vid1.shape)
        # print("fflow.shape: ",fflow.shape)
        # print("bflow.shape: ",bflow.shape)
        # print("dists_exh.shape: ",dists_exh.shape)
        # print("inds_exh.shape: ",inds_exh.shape)
        # print("self_dists.shape: ",self_dists.shape)
        # print(qstart,stride0,n_h0,n_w0,ps,pt,ws_h,ws_w,wt,chnls)

        # -- forward --
        th.cuda.set_device(device)
        stnls_cuda.search_prod_pf_with_index_forward(
            vid0, vid1, fflow, bflow,
            dists_exh, inds_exh, self_dists,
            qstart, stride0, n_h0, n_w0,
            ps, pt, ws_h, ws_w, wt, chnls, stride1, dilation,
            search_abs, reflect_bounds, use_adj, full_ws,
            anchor_self, use_self, h0_off, w0_off, h1_off, w1_off,
            tranges,n_tranges)#,min_tranges)


        # -- remove self --
        if remove_self:
            dists_exh,inds_exh = run_remove_self_cuda(dists_exh,inds_exh,qstart,
                                                      stride0,n_h0,n_w0)

        # -- shape for next step --
        dists_exh = dists_exh.view(B*Q,-1)#.contiguous()
        inds_exh = inds_exh.view(B*Q,-1,3)#.contiguous()

        # -- topk --
        if use_k:
            dists,inds = allocate_rtn(B*Q,k,device,dtype)
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
        dists = dists.view(B,Q,-1)
        inds = inds.view(B,Q,-1,3)

        # -- for backward --
        ctx.save_for_backward(inds,vid0,vid1)
        ctx.vid_shape = vid0.shape
        ctx.qstart,ctx.stride0 = qstart,stride0
        ctx.ps,ctx.pt,ctx.dil = ps,pt,dilation
        ctx.reflect_bounds = reflect_bounds
        ctx.rbwd,ctx.exact = rbwd,exact
        ctx.use_adj,ctx.nbwd = use_adj,nbwd
        ctx.n_h0,ctx.n_w0 = n_h0,n_w0
        ctx.h0_off,ctx.w0_off = h0_off, w0_off
        ctx.h1_off,ctx.w1_off = h1_off, w1_off
        return dists,inds

        # # th.cuda.synchronize()
        # # -- shape for output --
        # # b = dists_exh.shape[0]
        # dists_exh=dists_exh.view(B,Q,-1)#.contiguous()
        # inds_exh=inds_exh.view(B,Q,-1,3)#.contiguous()

        # # -- remove self --
        # if remove_self:
        #     dists_exh,inds_exh = run_remove_self_cuda(dists_exh,inds_exh,qstart,
        #                                               stride0,n_h0,n_w0)

        # # -- top k --
        # if use_k:
        #     topk_k = k if anchor_self else k-1
        #     dists,inds = allocate_rtn(B*Q,k,device,dtype)
        #     dists_exh = dists_exh.view(B*Q,-1)#.contiguous()
        #     inds_exh = inds_exh.view(B*Q,-1,3)#.contiguous()
        #     topk_with_anchor(dists_exh,inds_exh,dists,inds,self_dists,anchor_self)
        #     # if anchor_self:
        #     #     get_topk_prod(dists_exh,inds_exh,dists[:,1:],inds[:,1:])
        #     #     run_anchor_self(dists,inds,self_dists,dists_exh,inds_exh)
        #     #     #,wt,ws_h,ws_w)
        #     # else:
        #     #     get_topk_prod(dists_exh,inds_exh,dists,inds)
        # else:
        #     # args = th.where(th.isnan(dists_exh))
        #     # dists_exh[args] = -th.inf # fix nan
        #     # b = dists_exh.shape[0]
        #     dists = dists_exh.view(B,Q,-1)#.contiguous()
        #     inds = inds_exh.view(B,Q,-1,3)#.contiguous()

        # # -- fill nans --
        # args = th.where(th.isnan(dists))
        # dists[args] = -th.inf # fix nan

        # # -- shape with heads -
        # dists = dists.view(B,Q,-1)
        # inds = inds.view(B,Q,-1,3)

        # # -- contiguous --
        # # dists = dists.contiguous()
        # # inds = inds.contiguous()

        # # -- for backward --
        # ctx.save_for_backward(dists,inds,vid0,vid1)
        # ctx.vid_shape = vid0.shape
        # ctx.use_adj = use_adj
        # ctx.ps,ctx.pt = ps,pt
        # ctx.rbwd = rbwd
        # ctx.nbwd = nbwd
        # ctx.lam = lam
        # ctx.use_k = use_k
        # ctx.reflect_bounds = reflect_bounds
        # ctx.full_ws = full_ws
        # ctx.exact = exact
        # ctx.h0_off = h0_off
        # ctx.w0_off = w0_off
        # ctx.h1_off = h1_off
        # ctx.w1_off = w1_off
        # ctx.dilation = dilation
        # ctx.stride0 = stride0
        # ctx.qstart = qstart

        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds_is_none):

        # -- unpack --
        dists,inds,vid0,vid1 = ctx.saved_tensors
        vid_shape,exact = ctx.vid_shape,ctx.exact
        lam,ps,pt,dil = ctx.lam,ctx.ps,ctx.pt,ctx.dilation
        qstart,stride0 = ctx.qstart,ctx.stride0
        full_ws,nbwd = ctx.full_ws,ctx.nbwd
        rbwd = ctx.rbwd
        h0_off = ctx.h0_off
        w0_off = ctx.w0_off
        h1_off = ctx.h1_off
        w1_off = ctx.w1_off
        use_adj = ctx.use_adj
        reflect_bounds = ctx.reflect_bounds
        n_h0,n_w0 = get_num_img(vid0[0].shape,stride0,ps,dil)

        # -- gradient --
        vid0_grad = allocate_vid(vid_shape,grad_dists.device)
        vid1_grad = allocate_vid(vid_shape,grad_dists.device)

        # -- contiguous --
        grad_dists = grad_dists.contiguous()

        # -- allow for repeated exec --
        if nbwd == 1:
            stnls_cuda.search_prod_with_index_backward(
                vid0_grad,vid1_grad,vid0,vid1,
                grad_dists,inds,
                qstart,stride0,n_h0,n_w0,
                ps,pt,lam,use_adj,reflect_bounds,
                h0_off,w0_off,h1_off,w1_off,full_ws,rbwd,exact)
        else:
            for _ in range(nbwd):
                grad_vid0_i = allocate_vid(vid_shape,grad_dists.device)
                grad_vid1_i = allocate_vid(vid_shape,grad_dists.device)
                stnls_cuda.search_prod_with_index_backward(
                    vid0_grad,vid1_grad,vid0,vid1,
                    grad_dists,inds,
                    qstart,stride0,n_h0,n_w0,
                    ps,pt,lam,use_adj,reflect_bounds,
                    h0_off,w0_off,h1_off,w1_off,full_ws,rbwd,exact)
                grad_vid0 += grad_vid0_i
                grad_vid1 += grad_vid1_i
            grad_vid0 /= nbwd
            grad_vid1 /= nbwd

        # th.cuda.synchronize()
        return vid0_grad,vid1_grad,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None

class ProductPfSearch_with_index(th.nn.Module):

    def __init__(self, fflow, bflow, k, ps, pt, ws, wt,
                 h0_off=0, w0_off=0, h1_off=0, w1_off=0,
                 chnls=-1, stride0=1, stride1=1, dilation=1, lam = 1.,
                 search_abs=False, reflect_bounds=True, use_adj=True, use_k=True,
                 anchor_self = False, use_self=False, remove_self=False,
                 full_ws=False, nbwd=1, rbwd=True, exact=True):
        super().__init__()
        self.k = k
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.wt = wt
        self.fflow = fflow
        self.bflow = bflow
        self.chnls = chnls
        self.stride0 = stride0
        self.stride1 = stride1
        self.dilation = dilation
        self.lam = lam
        self.search_abs = search_abs
        self.anchor_self = anchor_self
        self.use_self = use_self
        self.reflect_bounds = reflect_bounds
        self.use_adj = use_adj
        self.use_k = use_k
        self.h0_off = h0_off
        self.w0_off = w0_off
        self.h1_off = h1_off
        self.w1_off = w1_off
        self.remove_self = remove_self
        self.full_ws = full_ws
        self.nbwd = nbwd
        self.rbwd = rbwd
        self.exact = exact

    def _get_args(self,vshape):
        # -- unpack --
        ws,wt,k,chnls = self.ws,self.wt,self.k,self.chnls
        b,t,c,h,w = vshape

        # -- compute --
        n_h,n_w = get_num_img(vshape[1:],self.stride1,self.ps,self.dilation)
        ws_h,ws_w = ws,ws
        if ws == -1: ws_h,ws_w = n_h,n_w
        if k == -1: k = ws_h**2 * (2*wt + 1)
        if chnls <= 0: chnls = c
        # print("ws_h,ws_w,wt,k,chnls: ",ws_h,ws_w,wt,k,chnls)
        return ws_h,ws_w,wt,k,chnls

    # def update_flow(self,vshape,device,flows=None):
    #     b,t,c,h,w = vshape
    #     zflow = th.zeros((b,t-1,t,2,h,w),device=device)
    #     noflow = flows is None
    #     # print("noflow: ",noflow)
    #     self.fflow = zflow if noflow else flows.fflow
    #     self.bflow = zflow if noflow else flows.bflow

    def set_flows(self,flows,vid):
        (b,t,c,h,w),device = vid.shape,vid.device
        nh = (h-1)//self.stride0+1
        nw = (w-1)//self.stride0+1
        zflow = th.zeros((b,t-1,t,2,nh,nw),device=device,dtype=th.int32)
        zflow = th.zeros((b,t-1,t,2,nh,nw),device=device,dtype=th.int32)
        noflow = flows is None
        self.fflow = zflow if noflow else flows.fflow
        self.bflow = zflow if noflow else flows.bflow

    def _update_flow(self,vshape,device,stride0):
        b,t,c,h,w = vshape
        nh = (h-1)//stride0+1
        nw = (w-1)//stride0+1
        tgt_shape = (b,t-1,t,2,nh,nw)
        zflow = th.zeros((b,t-1,t,2,nh,nw),device=device,dtype=th.int32)
        if self.fflow is None: self.fflow = zflow
        if self.bflow is None: self.bflow = zflow
        # for (i,j) in zip([0,2,4,5],[0,1,3,4]):
        # print(self.fflow.shape,self.bflow.shape,tgt_shape)
        for i in range(len(tgt_shape)):
            assert self.fflow.shape[i] == tgt_shape[i],"Must be equal size: %d" % i
            assert self.bflow.shape[i] == tgt_shape[i],"Must be equal size: %d" % i

    def forward(self, vid0, vid1, qstart=0, nqueries=-1):
        self._update_flow(vid0.shape,vid0.device,self.stride0)
        ws_h,ws_w,wt,k,chnls = self._get_args(vid0.shape)
        return ProductPfSearchFunction_with_index.apply(
            vid0,vid1,self.fflow,self.bflow,qstart,nqueries,
            k,self.ps,self.pt,ws_h,ws_w,wt,chnls,
            self.stride0,self.stride1,self.dilation,self.lam,
            self.search_abs,self.reflect_bounds,
            self.use_adj,self.use_k,
            self.h0_off,self.w0_off,self.h1_off,self.w1_off,
            self.anchor_self,self.use_self,self.remove_self,
            self.full_ws,self.nbwd,self.rbwd,self.exact)

    def wrap_fwd(self,vid0,qstart,nqueries,vid1,_):
        return self(vid0,qstart,nqueries,vid1)

    def flops(self,T,C,H,W):

        # -- unpack --
        vshape = (1,T,C,H,W)
        ws_h,ws_w,wt,k,chnls = self._get_args(vshape)
        nheads = 1
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
