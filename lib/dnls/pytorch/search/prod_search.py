
# -- python --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- padding --
from dnls.utils.pads import comp_pads

# -- cpp cuda kernel --
import dnls_cuda
from dnls.utils.timer import ExpTimer

# -- local --
from .search_utils import *

def get_topk(l2_vals,l2_inds,vals,inds):

    # -- reshape exh --
    nq,st,ws,ws = l2_vals.shape
    l2_vals = l2_vals.view(nq,-1)
    l2_inds = l2_inds.view(nq,-1,3)

    # -- shape info --
    b,_ = l2_vals.shape
    _,k = vals.shape

    # -- fill nan --
    l2_vals[th.where(th.isnan(l2_vals))] = -th.inf # fix nan

    # -- take mins --
    order = th.argsort(l2_vals,dim=1,descending=True)
    vals[:b,:] = th.gather(l2_vals,1,order[:,:k])
    for i in range(inds.shape[-1]):
        inds[:b,:,i] = th.gather(l2_inds[:,:,i],1,order[:,:k])

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

class ProductSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, qinds, fflow, bflow,
                k, ps, pt, ws_h, ws_w, wt, chnls,
                stride,dilation,lam,
                use_search_abs, reflect_bounds, use_adj, use_k,
                oh0, ow0, oh1, ow1, nbwd, exact):
        """
        vid = [T,C,H,W]
        qinds = [NumQueries,K,3]
        ws = xsearch Window Spatial (ws)
        wt = xsearch Window Time (wt)
        """

        # -- unpack --
        device = qinds.device
        nq = qinds.shape[0]
        t,c,h,w = vid0.shape
        qinds = qinds.type(th.int32)

        # -- allocs --
        bufs = allocate_bufs(nq,t,ws_h,ws_w,wt,device)
        dists_exh,inds_exh = allocate_exh_prod(nq,ws_h,ws_w,wt,device)

        # -- pre-computed xsearch offsets --
        tranges,n_tranges,min_tranges = create_frame_range(t,wt,wt,pt,device)

        # -- forward --
        dnls_cuda.search_prod_forward(vid0, vid1, qinds, fflow, bflow,
                                  dists_exh, inds_exh,
                                  ps, pt, ws_h, ws_w, wt, chnls, stride, dilation,
                                  use_search_abs, reflect_bounds, use_adj,
                                  oh0, ow0, oh1, ow1,
                                  bufs,tranges,n_tranges,min_tranges)
        # th.cuda.synchronize()
        # th.cuda.empty_cache()
        # print(dists_exh[:3,:3,:3])
        # print("dists_exh._version:",dists_exh._version)
        # print("inds_exh._version:",inds_exh._version)

        # -- topk --
        # b = dists_exh.shape[0]
        # dists=dists_exh.view(b,-1)#.contiguous()
        # inds=inds_exh.view(b,-1,3)#.contiguous()
        # print(dists_exh[0],use_search_abs)
        # print(k, ps, pt, ws_h, ws_w, wt, chnls, use_search_abs, use_adj, reflect_bounds)
        # print(oh0, ow0, oh1, ow1)

        if use_k:
            dists,inds = allocate_rtn(nq,k,device)
            get_topk(dists_exh,inds_exh,dists,inds)
            dists = dists.contiguous()
            inds = inds.contiguous()
        else:
            args = th.where(th.isnan(dists_exh))
            dists_exh[args] = -th.inf # fix nan
            b = dists_exh.shape[0]
            dists=dists_exh.view(b,-1)#.contiguous()
            inds=inds_exh.view(b,-1,3)#.contiguous()
        # print(dists)

        # -- for backward --
        ctx.save_for_backward(dists,inds,
                              qinds,vid0,vid1)
        ctx.vid_shape = vid0.shape
        ctx.ps,ctx.pt = ps,pt
        ctx.nbwd = nbwd
        ctx.lam = lam
        ctx.use_k = use_k
        ctx.reflect_bounds = reflect_bounds
        ctx.exact = exact
        ctx.oh0 = oh0
        ctx.ow0 = ow0
        ctx.oh1 = oh1
        ctx.ow1 = ow1

        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds_is_none):
        dists,inds,qinds,vid0,vid1 = ctx.saved_tensors
        vid_shape,exact = ctx.vid_shape,ctx.exact
        lam,ps,pt,nbwd = ctx.lam,ctx.ps,ctx.pt,ctx.nbwd
        oh0,ow0 = ctx.oh0,ctx.ow0
        oh1,ow1 = ctx.oh1,ctx.ow1
        reflect_bounds = ctx.reflect_bounds
        # print("oh0, ow0, oh1, ow1: ",oh0, ow0, oh1, ow1)

        # -- start timer --
        # timer = ExpTimer()
        # timer.start("xsearch_bwd")

        # -- gradient --
        vid0_grad = allocate_vid(vid_shape,grad_dists.device)
        vid1_grad = allocate_vid(vid_shape,grad_dists.device)
        # th.cuda.synchronize()

        # -- allow for repeated exec --
        if nbwd == 1:
            dnls_cuda.search_prod_backward(vid0_grad,vid1_grad,vid0,vid1,
                                           qinds,grad_dists,inds,
                                           oh0,ow0,oh1,ow1,
                                           ps,pt,lam,reflect_bounds,exact)
        else:
            for _ in range(nbwd):
                grad_vid0_i = allocate_vid(vid_shape,grad_dists.device)
                grad_vid1_i = allocate_vid(vid_shape,grad_dists.device)
                dnls_cuda.search_prod_backward(vid0_grad_i,vid1_grad_i,vid0,vid1,
                                               qinds,grad_dists,inds,
                                               oh0,ow0,oh1,ow1,
                                               ps,pt,lam,reflect_bounds,exact)
                grad_vid0 += grad_vid0_i
                grad_vid1 += grad_vid1_i
            grad_vid0 /= nbwd
            grad_vid1 /= nbwd

        # th.cuda.synchronize()
        return vid0_grad,vid1_grad,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None

class ProductSearch(th.nn.Module):

    def __init__(self, fflow, bflow, k, ps, pt, ws, wt, oh0=0, ow0=0, oh1=0, ow1=0,
                 chnls=-1,stride=1, dilation=1, lam = 1., use_search_abs=False,
                 reflect_bounds=True, use_adj=True, use_k=True, nbwd=1, exact=True):
        super(ProductSearch, self).__init__()
        self.k = k
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.wt = wt
        self.fflow = fflow
        self.bflow = bflow
        self.chnls = chnls
        self.stride = stride
        self.dilation = dilation
        self.lam = lam
        self.use_search_abs = use_search_abs
        self.reflect_bounds = reflect_bounds
        self.use_adj = use_adj
        self.use_k = use_k
        self.oh0 = oh0
        self.ow0 = ow0
        self.oh1 = oh1
        self.ow1 = ow1
        self.nbwd = nbwd
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
        if ws == -1:
            ws_h = n_h
            ws_w = n_w
        if k == -1: k = ws**2 * (2*wt + 1)
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

    def forward(self, vid0, iqueries, vid1=None):
        if vid1 is None: vid1 = vid0
        self._update_flow(vid0.shape,vid0.device)
        ws_h,ws_w,wt,k,chnls = self._get_args(vid0.shape)
        return ProductSearchFunction.apply(vid0,vid1,iqueries,self.fflow,self.bflow,
                                           k,self.ps,self.pt,ws_h,ws_w,wt,chnls,
                                           self.stride,self.dilation,self.lam,
                                           self.use_search_abs,self.reflect_bounds,
                                           self.use_adj,self.use_k,
                                           self.oh0,self.ow0,self.oh1,self.ow1,
                                           self.nbwd, self.exact)
