
# -- python --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- padding --
from dnls.utils.pads import comp_pads

# -- cpp cuda kernel --
import dnls_cuda


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

def allocate_bufs(nq,t,ws,device):
    bufs = th.zeros(nq,3,t,ws,ws,dtype=th.int32,device=device)
    return bufs

def allocate_exh(nq,ws,wt,device):
    dists = th.zeros((nq,2*wt+1,ws,ws),device=device,dtype=th.float32)
    dists[...] = -float("inf")
    inds = th.zeros((nq,2*wt+1,ws,ws,3),device=device,dtype=th.int32)
    inds[...] = -1
    return dists,inds

def allocate_rtn(nq,k,device):
    nlDists = th.zeros((nq,k),device=device,dtype=th.float32)
    nlInds = th.zeros((nq,k,3),device=device,dtype=th.int32)
    return nlDists,nlInds

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

class CrossSearchNlFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, queryInds, fflow, bflow,
                k, ps, pt, ws, wt, chnls,
                stride,dilation,lam,
                use_search_abs, use_bounds, use_adj, use_k,
                oh0, ow0, oh1, ow1, exact):
        """
        vid = [T,C,H,W]
        queryInds = [NumQueries,K,3]
        ws = xsearch Window Spatial (ws)
        wt = xsearch Window Time (wt)
        """

        # -- unpack --
        device = queryInds.device
        nq = queryInds.shape[0]
        t,c,h,w = vid0.shape
        queryInds = queryInds.type(th.int32)

        # -- allocs --
        bufs = allocate_bufs(nq,t,ws,device)
        nlDists,nlInds = allocate_rtn(nq,k,device)
        nlDists_exh,nlInds_exh = allocate_exh(nq,ws,wt,device)

        # -- pre-computed xsearch offsets --
        tranges,n_tranges,min_tranges = create_frame_range(t,wt,wt,pt,device)

        # -- forward --
        dnls_cuda.xsearch_forward(vid0, vid1, queryInds, fflow, bflow,
                                  nlDists_exh, nlInds_exh,
                                  ps, pt, ws, wt, chnls, stride, dilation,
                                  use_search_abs, use_bounds, use_adj,
                                  oh0, ow0, oh1, ow1,
                                  bufs,tranges,n_tranges,min_tranges)
        # -- topk --
        if use_k:
            get_topk(nlDists_exh,nlInds_exh,nlDists,nlInds)
            nlDists = nlDists.contiguous()
            nlInds = nlInds.contiguous()
        else:
            nlDists_exh[th.where(th.isnan(nlDists_exh))] = -th.inf # fix nan
            b = nlDists_exh.shape[0]
            nlDists=nlDists_exh[:,0].view(b,-1).contiguous()
            nlInds=nlInds_exh[:,0].view(b,-1,3).contiguous()

        # -- fix "self" to 1. --
        nlDists[th.where(nlDists == th.inf)] = 1 # fix "self" to 1.

        # -- for backward --
        # print(nlDists.shape,nlInds.shape)
        ctx.save_for_backward(nlDists,nlInds,queryInds,vid0,vid1)
        ctx.vid_shape = vid0.shape
        ctx.ps,ctx.pt = ps,pt
        ctx.lam = lam
        ctx.use_k = use_k
        ctx.use_bounds = use_bounds
        ctx.exact = exact
        ctx.oh0 = oh0
        ctx.ow0 = ow0
        ctx.oh1 = oh1
        ctx.ow1 = ow1

        return nlDists,nlInds

    @staticmethod
    def backward(ctx, grad_nlDists, grad_nlInds):
        # print("unpacking.")
        nlDists,nlInds,queryInds,vid0,vid1 = ctx.saved_tensors
        vid_shape,exact = ctx.vid_shape,ctx.exact
        lam,ps,pt = ctx.lam,ctx.ps,ctx.pt
        oh0 = ctx.oh0
        ow0 = ctx.ow0
        oh1 = ctx.oh1
        ow1 = ctx.ow1
        use_bounds = ctx.use_bounds
        vid0_grad = allocate_vid(vid_shape,grad_nlDists.device)
        vid1_grad = allocate_vid(vid_shape,grad_nlDists.device)
        # th.cuda.synchronize()
        dnls_cuda.xsearch_backward(vid0_grad,vid1_grad,vid0,vid1,
                                   queryInds,grad_nlDists,nlInds,
                                   ps,pt,lam,use_bounds,exact)
        # th.cuda.synchronize()
        return vid0_grad,vid1_grad,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None

class CrossSearchNl(th.nn.Module):

    def __init__(self, fflow, bflow, k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                 chnls=1,stride=1, dilation=1, lam = 1., use_search_abs=False,
                 use_bound=True, use_adj=True, use_k=True, exact=True):
        super(CrossSearchNl, self).__init__()
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
        self.use_bound = use_bound
        self.use_adj = use_adj
        self.use_k = use_k
        self.oh0 = oh0
        self.ow0 = ow0
        self.oh1 = oh1
        self.ow1 = ow1
        self.exact = exact

    def _get_args(self,vshape):
        ws,wt,k = self.ws,self.wt,self.k
        ps,stride,dil = self.ps,self.stride,self.dilation
        t,c,h,w = vshape
        _,_,hp,wp = comp_pads(vshape, ps, stride, dil)
        n_h = (hp - (ps-1)*dil - 1)//stride + 1
        n_w = (wp - (ps-1)*dil - 1)//stride + 1
        if ws == -1: ws = n_h # hope its square
        if k == -1: k = ws**2 * (2*wt + 1)
        return ws,wt,k

    def _update_flow(self,vshape,device):
        t,c,h,w = vshape
        zflow = th.ones((t,2,h,w),device=device)
        if self.fflow is None: self.fflow = zflow
        if self.bflow is None: self.bflow = zflow
        for i in [0,2,3]:
            assert self.fflow.shape[i] == vshape[i],"Must be equal size"
            assert self.bflow.shape[i] == vshape[i],"Must be equal size"

    def forward(self, vid0, iqueries, vid1=None):
        if vid1 is None: vid1 = vid0
        self._update_flow(vid0.shape,vid0.device)
        ws,wt,k = self._get_args(vid0.shape)
        return CrossSearchNlFunction.apply(vid0,vid1,iqueries,self.fflow,self.bflow,
                                           k,self.ps,self.pt,ws,wt,self.chnls,
                                           self.stride,self.dilation,self.lam,
                                           self.use_search_abs,self.use_bound,
                                           self.use_adj,self.use_k,
                                           self.oh0,self.ow0,self.oh1,self.ow1,
                                           self.exact)

