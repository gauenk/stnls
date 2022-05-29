
# -- python --
import torch as th
import numpy as np

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

    # -- take mins --
    order = th.argsort(l2_vals,dim=1,descending=False)
    vals[:b,:] = th.gather(l2_vals,1,order[:,:k])
    for i in range(inds.shape[-1]):
        inds[:b,:,i] = th.gather(l2_inds[:,:,i],1,order[:,:k])

def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_bufs(nq,t,ws,device):
    bufs = th.zeros(nq,3,t,ws,ws,dtype=th.float32,device=device)
    return bufs

def allocate_exh(nq,ws,wt,device):
    dists = th.zeros((nq,2*wt+1,ws,ws),device=device,dtype=th.float32)
    dists[...] = float("inf")
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

class SearchNlFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid, queryInds, fflow, bflow,
                k, ps, pt, ws, wt, chnls,
                dilation=1,stride=1,lam=1.):
        """
        vid = [T,C,H,W]
        queryInds = [NumQueries,K,3]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """

        # -- unpack --
        device = queryInds.device
        nq = queryInds.shape[0]
        t,c,h,w = vid.shape
        queryInds = queryInds.type(th.int32)

        # -- allocs --
        bufs = allocate_bufs(nq,t,ws,device)
        nlDists,nlInds = allocate_rtn(nq,k,device)
        nlDists_exh,nlInds_exh = allocate_exh(nq,ws,wt,device)

        # -- pre-computed search offsets --
        tranges,n_tranges,min_tranges = create_frame_range(t,wt,wt,pt,device)

        # -- forward --
        dnls_cuda.search_forward(vid, queryInds, fflow, bflow,
                                 nlDists_exh, nlInds_exh,
                                 ps, pt, ws, wt, chnls, dilation, stride,
                                 bufs,tranges,n_tranges,min_tranges)
        # -- topk --
        get_topk(nlDists_exh,nlInds_exh,nlDists,nlInds)
        nlDists[:,0] = 0. # fix the "-100" hack to 0.

        # -- for backward --
        ctx.save_for_backward(nlDists,nlInds)
        ctx.vid_shape = vid.shape
        ctx.ps,ctx.pt = ps,pt
        ctx.lam = lam

        return nlDists,nlInds

    @staticmethod
    def backward(ctx, grad_nlDists, grad_nlInds):
        nlDists,nlInds = ctx.saved_tensors
        bkwd_nlDists = nlDists * grad_nlDists
        vid_shape = ctx.vid_shape
        lam,ps,pt = ctx.lam,ctx.ps,ctx.pt
        vid = allocate_vid(vid_shape,grad_nlDists.device)
        dnls_cuda.search_backward(vid,bkwd_nlDists,nlInds,ps,pt,lam)
        return vid,None,None,None,None,None,None,None,None,None,None,None,None,None

class SearchNl(th.nn.Module):

    def __init__(self, fflow, bflow, k, ps, pt, ws, wt, chnls=1,
                 dilation=1, stride=1, lam = 1.):
        super(SearchNl, self).__init__()
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
        self.lam = lam

    def forward(self, vid, queryInds):
        return SearchNlFunction.apply(vid,queryInds,self.fflow,self.bflow,
                                      self.k,self.ps,self.pt,self.ws,self.wt,
                                      self.chnls,self.dilation,self.stride,self.lam)

