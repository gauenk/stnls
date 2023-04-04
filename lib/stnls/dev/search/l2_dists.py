
# -- python --
import torch as th
import numpy as np

# -- cpp cuda kernel --
import stnls_cuda

# -- local --
from .search_utils import *

class L2DistsFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, inds,
                qstart, stride0,
                h0_off, w0_off, h1_off, w1_off,
                ps, pt, chnls, dilation=1,use_adj=True,
                reflect_bounds=True,anchor_self=False,remove_self=False,
                nbwd=1,use_rand=True,exact=False):
        """
        vid0 = [T,C,H,W]
        qinds = [NumQueries,K,3]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """

        # -- unpack --
        device = vid0.device
        b,t,c,h,w = vid0.shape
        n_h0,n_w0 = get_num_img(vid0[0].shape,stride0,ps,dilation)

        # -- allocs --
        b,nq,nn,_ = inds.shape
        dists = th.zeros((b,nq,nn),dtype=th.float32,device=device)

        # -- forward --
        gpuid = th.cuda.current_device()
        th.cuda.set_device(device)
        stnls_cuda.l2_dists_forward(vid0, vid1, dists, inds,
                                   qstart, stride0, n_h0, n_w0,
                                   h0_off, w0_off, h1_off, w1_off,
                                   ps, pt, dilation, chnls,
                                   use_adj, reflect_bounds, anchor_self)

        # -- remove self --
        if remove_self:
            dists,inds = run_remove_self_cuda(dists,inds,qstart,
                                              stride0,n_h0,n_w0)

        # -- fill if anchored --
        if anchor_self:
            args = th.where(dists == -100)
            dists[args] = 0.

        # -- for backward --
        ctx.save_for_backward(inds,vid0,vid1)
        ctx.vid_shape = vid0.shape
        ctx.qstart,ctx.stride0 = qstart,stride0
        ctx.ps,ctx.pt,ctx.dil = ps,pt,dilation
        ctx.reflect_bounds = reflect_bounds
        ctx.use_rand,ctx.exact = use_rand,exact
        ctx.use_adj,ctx.nbwd = use_adj,nbwd
        ctx.n_h0,ctx.n_w0 = n_h0,n_w0
        ctx.h0_off,ctx.w0_off = h0_off, w0_off
        ctx.h1_off,ctx.w1_off = h1_off, w1_off
        ctx.chnls = chnls
        return dists

    @staticmethod
    def backward(ctx, grad_dists):

        # -- unpack --
        inds,vid0,vid1 = ctx.saved_tensors
        vid_shape,nbwd = ctx.vid_shape,ctx.nbwd
        qstart,stride0 = ctx.qstart,ctx.stride0
        ps,pt,dil = ctx.ps,ctx.pt,ctx.dil
        chnls = ctx.chnls # untested.
        use_rand = ctx.use_rand
        exact,use_adj = ctx.exact,ctx.use_adj
        reflect_bounds = ctx.reflect_bounds
        n_h0,n_w0 = ctx.n_h0,ctx.n_w0
        h0_off, w0_off = ctx.h0_off,ctx.w0_off
        h1_off, w1_off = ctx.h1_off,ctx.w1_off
        grad_vid0 = allocate_vid(vid_shape,grad_dists.device)
        grad_vid1 = allocate_vid(vid_shape,grad_dists.device)

        # -- allow for repeated exec --
        if nbwd == 1:
            stnls_cuda.l2_dists_backward(grad_vid0,grad_vid1,
                                        vid0,vid1,
                                        grad_dists,inds,
                                        qstart,stride0,n_h0,n_w0,
                                        h0_off, w0_off, h1_off, w1_off,
                                        ps, pt, dil, chnls, use_adj,
                                        reflect_bounds,use_rand,exact)
        else:
            for _ in range(nbwd):
                grad_vid0_i = allocate_vid(vid_shape,grad_dists.device)
                grad_vid1_i = allocate_vid(vid_shape,grad_dists.device)
                stnls_cuda.l2_dists_backward(grad_vid0_i,grad_vid1_i,
                                            vid0,vid1,
                                            grad_dists,inds,
                                            qstart,stride0,n_h0,n_w0,
                                            h0_off, w0_off, h1_off, w1_off,
                                            ps, pt, dil, chnls, use_adj,
                                            reflect_bounds,use_rand,exact)
                grad_vid0 += grad_vid0_i
                grad_vid1 += grad_vid1_i
            grad_vid0 /= nbwd
            grad_vid1 /= nbwd

        return grad_vid0,grad_vid1,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None

class L2Dists(th.nn.Module):

    def __init__(self, ps, pt, chnls=-1,
                 dilation=1, stride0=1,
                 use_adj=True, reflect_bounds=True,
                 nbwd=1, exact=False,
                 h0_off=0,w0_off=0,h1_off=0,w1_off=0,remove_self=False,
                 anchor_self=False,use_rand=True):
        super().__init__()
        self.ps = ps
        self.pt = pt
        self.chnls = chnls
        self.dilation = dilation
        self.stride0 = stride0
        self.h0_off = h0_off
        self.w0_off = w0_off
        self.h1_off = h1_off
        self.w1_off = w1_off
        self.use_adj = use_adj
        self.reflect_bounds = reflect_bounds
        self.anchor_self = anchor_self
        self.remove_self = remove_self
        self.nbwd = nbwd
        self.exact = exact
        self.use_rand = use_rand

    def forward(self, vid0, vid1, inds, qstart):
        if self.chnls <= 0: chnls = vid0.shape[-3]
        else: chnls = self.chnls
        return L2DistsFunction.apply(vid0,vid1,inds,
                                     qstart,self.stride0,
                                     self.h0_off,self.w0_off,
                                     self.h1_off,self.w1_off,
                                     self.ps,self.pt,chnls,
                                     self.dilation,
                                     self.use_adj,
                                     self.reflect_bounds,
                                     self.anchor_self,
                                     self.remove_self,
                                     self.nbwd,self.use_rand,self.exact)

