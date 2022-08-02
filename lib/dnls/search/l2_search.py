
# -- python --
import torch as th
import numpy as np

# -- padding --
from dnls.utils.pads import comp_pads

# -- cpp cuda kernel --
import dnls_cuda

# -- local --
from .search_utils import *


class L2SearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, qinds, fflow, bflow,
                h0_off, w0_off, h1_off, w1_off,
                k, ps, pt, ws_h, ws_w, wt, chnls,
                dilation=1,stride=1,use_k=True,use_adj=True,
                reflect_bounds=True,search_abs=False,
                full_ws=False,remove_self=False,exact=False):
        """
        vid0 = [T,C,H,W]
        qinds = [NumQueries,K,3]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """

        # -- unpack --
        device = qinds.device
        nq = qinds.shape[0]
        t,c,h,w = vid0.shape
        qinds = qinds.type(th.int32)

        # -- allocs --
        bufs = allocate_bufs(nq,t,ws_h,ws_w,wt,device)
        dists_exh,inds_exh = allocate_exh(nq,wt,ws_h,ws_w,device)

        # -- pre-computed search offsets --
        tranges,n_tranges,min_tranges = create_frame_range(t,wt,wt,pt,device)
        # print(tranges)

        # -- forward --
        dnls_cuda.search_l2_forward(vid0, vid1, qinds, fflow, bflow,
                                 dists_exh, inds_exh,
                                 h0_off, w0_off, h1_off, w1_off,
                                 ps, pt, ws_h, ws_w,
                                 wt, chnls, dilation, stride, use_adj,
                                 reflect_bounds, search_abs,  full_ws,
                                 bufs, tranges, n_tranges, min_tranges)
        # -- shape for output --
        b = dists_exh.shape[0]
        dists_exh=dists_exh.view(b,-1)#.contiguous()
        inds_exh=inds_exh.view(b,-1,3)#.contiguous()

        # -- remove self --
        if remove_self:
            dists_exh,inds_exh = run_remove_self(dists_exh,inds_exh,qinds)

        # -- topk --
        if use_k:
            dists,inds = allocate_rtn(nq,k,device)
            get_topk(dists_exh,inds_exh,dists,inds)
        else:
            dists,inds = dists_exh,inds_exh

        # -- for backward --
        ctx.save_for_backward(qinds,inds,vid0,vid1)
        ctx.vid_shape = vid0.shape
        ctx.ps,ctx.pt,ctx.dil = ps,pt,dilation
        ctx.reflect_bounds = reflect_bounds
        ctx.exact = exact
        ctx.use_adj = use_adj
        ctx.h0_off,ctx.w0_off = h0_off, w0_off
        ctx.h1_off,ctx.w1_off = h1_off, w1_off
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds):
        qinds,inds,vid0,vid1 = ctx.saved_tensors
        vid_shape = ctx.vid_shape
        ps,pt,dil = ctx.ps,ctx.pt,ctx.dil
        exact,use_adj = ctx.exact,ctx.use_adj
        reflect_bounds = ctx.reflect_bounds
        h0_off, w0_off = ctx.h0_off,ctx.w0_off
        h1_off, w1_off = ctx.h1_off,ctx.w1_off
        grad_vid0 = allocate_vid(vid_shape,grad_dists.device)
        grad_vid1 = allocate_vid(vid_shape,grad_dists.device)
        dnls_cuda.search_l2_backward(grad_vid0,grad_vid1,vid0,vid1,
                                  grad_dists,inds,qinds,
                                  h0_off, w0_off, h1_off, w1_off,
                                  ps,pt,dil, use_adj,reflect_bounds,exact)
        th.cuda.synchronize()

        return grad_vid0,grad_vid1,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None

class L2Search(th.nn.Module):

    def __init__(self, fflow, bflow, k, ps, pt, ws, wt, chnls=-1,
                 dilation=1, stride=1,
                 use_k=True, use_adj=True, reflect_bounds=True,
                 search_abs=False, full_ws = False, exact=False,
                 h0_off=0,w0_off=0,h1_off=0,w1_off=0,remove_self=False):
        super(L2Search, self).__init__()
        self.k = k
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.wt = wt
        self.fflow = fflow
        self.bflow = bflow
        self.chnls = chnls
        self.dilation = dilation
        self.stride = stride
        self.h0_off = h0_off
        self.w0_off = w0_off
        self.h1_off = h1_off
        self.w1_off = w1_off
        self.use_adj = use_adj
        self.use_k = use_k
        self.reflect_bounds = reflect_bounds
        self.search_abs = search_abs
        self.full_ws = full_ws
        self.remove_self = remove_self
        self.exact = exact

    def _get_args(self,vshape):
        # -- unpack --
        ws,wt,k,chnls = self.ws,self.wt,self.k,self.chnls
        ps,stride,dil = self.ps,self.stride,self.dilation
        t,c,h,w = vshape

        # -- compute --
        _,_,hp,wp = comp_pads(vshape, ps, stride, dil)
        n_h = (hp - (ps-1)*dil - 1)//stride + 1
        n_w = (wp - (ps-1)*dil - 1)//stride + 1
        ws_h,ws_w = ws,ws
        if ws == -1: ws_h,ws_w = n_h,n_w
        if k == -1: k = ws**2 * (2*wt + 1)
        if chnls <= 0: chnls = c
        return ws_h,ws_w,wt,k,chnls

    def _update_flow(self,vshape,device):
        t,c,h,w = vshape
        zflow = th.ones((t,2,h,w),device=device)
        if self.fflow is None: self.fflow = zflow
        if self.bflow is None: self.bflow = zflow
        for i in [0,2,3]:
            assert self.fflow.shape[i] == vshape[i],"Must be equal size: %d" % i
            assert self.bflow.shape[i] == vshape[i],"Must be equal size: %d" % i

    def forward(self, vid0, qinds, vid1=None):
        if vid1 is None: vid1 = vid0
        self._update_flow(vid0.shape,vid0.device)
        ws_h,ws_w,wt,k,chnls = self._get_args(vid0.shape)
        return L2SearchFunction.apply(vid0,vid1,qinds,self.fflow,self.bflow,
                                      self.h0_off,self.w0_off,self.h1_off,self.w1_off,
                                      k,self.ps,self.pt,ws_h,ws_w,wt,chnls,
                                      self.dilation,self.stride,
                                      self.use_k,self.use_adj,
                                      self.reflect_bounds,self.search_abs,
                                      self.full_ws,self.remove_self,self.exact)

