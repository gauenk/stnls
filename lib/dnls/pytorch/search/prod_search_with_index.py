
# -- python --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- utils --
from ...utils.timer import ExpTimer

# -- cpp cuda kernel --
import dnls_cuda

# -- local --
from .search_utils import *


def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_bufs(nq,t,ws_h,ws_w,wt,device):
    if wt <= 0:
        bufs = th.zeros(1,1,1,1,1,dtype=th.int32,device=device)
    else:
        bufs = th.zeros(nq,3,t,ws_h,ws_w,dtype=th.int32,device=device)
    return bufs

# def allocate_exh(nq,ws_h,ws_w,wt,device):
#     dists = th.zeros((nq,2*wt+1,ws_h,ws_w),device=device,dtype=th.float32)
#     dists[...] = -float("inf")
#     inds = th.zeros((nq,2*wt+1,ws_h,ws_w,3),device=device,dtype=th.int32)
#     inds[...] = -1
#     return dists,inds

def allocate_rtn(nq,k,device):
    dists = th.zeros((nq,k),device=device,dtype=th.float32)
    inds = th.zeros((nq,k,3),device=device,dtype=th.int32)
    return dists,inds

def create_frame_range(nframes,nWt_f,nWt_b,ps_t,device):
    tranges,n_tranges,min_tranges = [],[],[]
    for t_c in range(nframes-ps_t+1):

        # -- limits --
        shift_t = min(0,t_c - nWt_b) + max(0,t_c + nWt_f - nframes + ps_t)
        t_start = max(t_c - nWt_b - shift_t,0)
        t_end = min(nframes - ps_t, t_c + nWt_f - shift_t)+1

        # -- final range --
        trange = [t_c]
        trange_s = np.arange(t_c+1,t_end)
        trange_e = np.arange(t_start,t_c)[::-1]
        for t_i in range(trange_s.shape[0]):
            trange.append(trange_s[t_i])
        for t_i in range(trange_e.shape[0]):
            trange.append(trange_e[t_i])

        # -- aug vars --
        n_tranges.append(len(trange))
        min_tranges.append(np.min(trange))

        # -- add padding --
        for pad in range(nframes-len(trange)):
            trange.append(-1)

        # -- to tensor --
        trange = th.IntTensor(trange).to(device)
        tranges.append(trange)

    tranges = th.stack(tranges).to(device).type(th.int32)
    n_tranges = th.IntTensor(n_tranges).to(device).type(th.int32)
    min_tranges = th.IntTensor(min_tranges).to(device).type(th.int32)
    return tranges,n_tranges,min_tranges

class ProductSearchFunction_with_index(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, fflow, bflow, qstart, nqueries,
                k, ps, pt, ws_h, ws_w, wt, chnls,
                stride0, stride1, dilation,lam,
                use_search_abs, reflect_bounds, use_adj, use_k,
                oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
                use_rand, exact):
        """
        vid = [T,C,H,W]
        ws = xsearch Window Spatial (ws)
        wt = xsearch Window Time (wt)
        """

        # -- unpack --
        device = vid0.device
        nq = nqueries
        t,c,h,w = vid0.shape
        n_h0,n_w0 = get_num_img(vid0.shape,stride0,ps,dilation)

        # -- allocs --
        # bufs = allocate_bufs(nq,t,ws_h,ws_w,wt,device)
        dists_exh,inds_exh = allocate_exh_prod(nq,wt,ws_h,ws_w,device)

        # -- pre-computed xsearch offsets --
        tranges,n_tranges,min_tranges = create_frame_range(t,wt,wt,pt,device)

        # -- forward --
        dnls_cuda.search_prod_with_index_forward(
            vid0, vid1, fflow, bflow,
            dists_exh, inds_exh,
            qstart, stride0, n_h0, n_w0,
            ps, pt, ws_h, ws_w, wt, chnls, stride1, dilation,
            use_search_abs, reflect_bounds, use_adj, full_ws,
            oh0, ow0, oh1, ow1,
            tranges,n_tranges,min_tranges)

        th.cuda.synchronize()
        # -- shape for output --
        b = dists_exh.shape[0]
        dists_exh=dists_exh.view(b,-1)#.contiguous()
        inds_exh=inds_exh.view(b,-1,3)#.contiguous()

        # -- remove self --
        if remove_self:
            dists_exh,inds_exh = run_remove_self_cuda(dists_exh,inds_exh,qstart,
                                                      stride0,n_h0,n_w0)

        # -- top k --
        if use_k:
            dists,inds = allocate_rtn(nq,k,device)
            get_topk_prod(dists_exh,inds_exh,dists,inds)
            dists = dists.contiguous()
            inds = inds.contiguous()
        else:
            args = th.where(th.isnan(dists_exh))
            dists_exh[args] = -th.inf # fix nan
            b = dists_exh.shape[0]
            dists=dists_exh.view(b,-1)#.contiguous()
            inds=inds_exh.view(b,-1,3)#.contiguous()

        # -- for backward --
        ctx.save_for_backward(dists,inds,vid0,vid1)
        ctx.vid_shape = vid0.shape
        ctx.use_adj = use_adj
        ctx.ps,ctx.pt = ps,pt
        ctx.use_rand = use_rand
        ctx.nbwd = nbwd
        ctx.lam = lam
        ctx.use_k = use_k
        ctx.reflect_bounds = reflect_bounds
        ctx.full_ws = full_ws
        ctx.exact = exact
        ctx.oh0 = oh0
        ctx.ow0 = ow0
        ctx.oh1 = oh1
        ctx.ow1 = ow1
        ctx.dilation = dilation
        ctx.stride0 = stride0
        ctx.qstart = qstart

        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds_is_none):

        # -- unpack --
        dists,inds,vid0,vid1 = ctx.saved_tensors
        vid_shape,exact = ctx.vid_shape,ctx.exact
        lam,ps,pt,dil = ctx.lam,ctx.ps,ctx.pt,ctx.dilation
        qstart,stride0 = ctx.qstart,ctx.stride0
        full_ws,nbwd = ctx.full_ws,ctx.nbwd
        use_rand = ctx.use_rand
        oh0 = ctx.oh0
        ow0 = ctx.ow0
        oh1 = ctx.oh1
        ow1 = ctx.ow1
        use_adj = ctx.use_adj
        reflect_bounds = ctx.reflect_bounds
        n_h0,n_w0 = get_num_img(vid0.shape,stride0,ps,dil)

        # -- gradient --
        vid0_grad = allocate_vid(vid_shape,grad_dists.device)
        vid1_grad = allocate_vid(vid_shape,grad_dists.device)

        # -- allow for repeated exec --
        if nbwd == 1:
            dnls_cuda.search_prod_with_index_backward(
                vid0_grad,vid1_grad,vid0,vid1,
                grad_dists,inds,
                qstart,stride0,n_h0,n_w0,
                ps,pt,lam,use_adj,reflect_bounds,
                oh0,ow0,oh1,ow1,full_ws,use_rand,exact)
        else:
            for _ in range(nbwd):
                grad_vid0_i = allocate_vid(vid_shape,grad_dists.device)
                grad_vid1_i = allocate_vid(vid_shape,grad_dists.device)
                dnls_cuda.search_prod_with_index_backward(
                    vid0_grad,vid1_grad,vid0,vid1,
                    grad_dists,inds,
                    qstart,stride0,n_h0,n_w0,
                    ps,pt,lam,reflect_bounds,
                    oh0,ow0,oh1,ow1,full_ws,use_rand,exact)
                grad_vid0 += grad_vid0_i
                grad_vid1 += grad_vid1_i
            grad_vid0 /= nbwd
            grad_vid1 /= nbwd

        # th.cuda.synchronize()
        return vid0_grad,vid1_grad,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None

class ProductSearch_with_index(th.nn.Module):

    def __init__(self, fflow, bflow, k, ps, pt, ws, wt, oh0=0, ow0=0, oh1=0, ow1=0,
                 chnls=-1, stride0=1, stride1=1, dilation=1, lam = 1.,
                 search_abs=False, reflect_bounds=True, use_adj=True,
                 use_k=True, remove_self=False, full_ws=False, nbwd=1,
                 use_rand=True, exact=True):
        super(ProductSearch_with_index, self).__init__()
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
        self.reflect_bounds = reflect_bounds
        self.use_adj = use_adj
        self.use_k = use_k
        self.oh0 = oh0
        self.ow0 = ow0
        self.oh1 = oh1
        self.ow1 = ow1
        self.remove_self = remove_self
        self.full_ws = full_ws
        self.nbwd = nbwd
        self.use_rand = use_rand
        self.exact = exact

    def _get_args(self,vshape):
        # -- unpack --
        ws,wt,k,chnls = self.ws,self.wt,self.k,self.chnls
        t,c,h,w = vshape

        # -- compute --
        n_h,n_w = get_num_img(vshape,self.stride1,self.ps,self.dilation)
        ws_h,ws_w = ws,ws
        if ws == -1:
            ws_h = n_h
            ws_w = n_w
        if k == -1: k = ws_h**2 * (2*wt + 1)
        if chnls <= 0: chnls = c
        return ws_h,ws_w,wt,k,chnls

    def _update_flow(self,vshape,device):
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
        return ProductSearchFunction_with_index.apply(
            vid0,vid1,self.fflow,self.bflow,qstart,nqueries,
            k,self.ps,self.pt,ws_h,ws_w,wt,chnls,
            self.stride0,self.stride1,self.dilation,self.lam,
            self.search_abs,self.reflect_bounds,
            self.use_adj,self.use_k,self.oh0,self.ow0,
            self.oh1,self.ow1,self.remove_self,
            self.full_ws,self.nbwd,self.use_rand,self.exact)
