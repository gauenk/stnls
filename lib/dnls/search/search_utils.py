
# -- python --
import torch as th
import numpy as np
from einops import repeat

# -- padding --
from dnls.utils.pads import comp_pads

# -- cpp cuda kernel --
import dnls_cuda

def get_topk(l2_vals,l2_inds,vals,inds):

    # -- reshape exh --
    nq = l2_vals.shape[0]
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

def run_remove_self_cuda(dists,inds,qstart,stride,n_h,n_w):
    nq,k = dists.shape
    mask = th.zeros((nq,k),device=dists.device,dtype=th.bool)
    dnls_cuda.remove_self_from_search(inds,mask,qstart,stride,n_h,n_w)
    mask = th.logical_not(mask)
    # print(dists.shape)
    # print(mask.sum(1))
    # args = th.where(mask.sum(1)==k)
    # mask[args,-1] = 0
    # print(mask.sum(1))
    # print(th.all(mask.sum(1)==1))
    rm_dists = th.masked_select(dists,mask)
    # print(rm_dists.shape)
    rm_dists = th.masked_select(dists,mask).view(nq,k-1)
    mask = repeat(mask,'a b -> a b c',c=3)
    rm_inds = th.masked_select(inds,mask).view(nq,k-1,3)

    return rm_dists,rm_inds

def run_remove_self(dists,inds,qinds):
    q,k,_ = inds.shape
    # new_dists = th.zeros((q,k-1),device=inds.device,dtype=th.float32)
    # new_inds = th.zeros((q,k-1,3),device=inds.device,dtype=th.int32)
    # print(q,k)
    # for qi in range(q):
    #     is_q = th.where(th.abs(inds[qi] - qinds[qi]).sum(1) < 1e-10)
    #     not_q = th.where(th.abs(inds[qi] - qinds[qi]).sum(1) > 1e-10)
    #     print(qi,len(not_q[0]))
    #     if len(is_q[0]) != 1:
    #         print(inds[qi])
    #         print(qinds[qi])
    #     assert len(is_q[0]) == 1
    q,k,_ = inds.shape
    not_q = th.where(th.abs(inds - qinds[:,None]).sum(2) > 1e-10)
    # print(inds)
    # print(qinds)
    # print(not_q)
    # print(len(not_q[0]))
    # print(len(not_q[1]))
    # print(q*k,q*(k-1))
    dists = dists[not_q].view(q,k-1)
    inds = inds[not_q].view(q,k-1,3)
    return dists,inds

def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_bufs(nq,t,ws_h,ws_w,wt,device):
    if wt <= 0:
        bufs = th.zeros(1,1,1,1,1,dtype=th.int32,device=device)
    else:
        st = min(t,2*wt+1)
        bufs = th.zeros(nq,3,st,ws_h,ws_w,dtype=th.int32,device=device)
    return bufs

def allocate_exh(nq,wt,ws_h,ws_w,device):
    dists = th.zeros((nq,2*wt+1,ws_h,ws_w),device=device,dtype=th.float32)
    dists[...] = float("inf")
    inds = th.zeros((nq,2*wt+1,ws_h,ws_w,3),device=device,dtype=th.int32)
    inds[...] = -1
    return dists,inds

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

def get_num_img(vshape,stride,ps,dil,only_full=True,use_pad=True):
    if use_pad:
        _,_,h,w = comp_pads(vshape, ps, stride, dil)
    else:
        _,_,h,w = vshape

    if only_full:
        n_h = (h - (ps-1)*dil - 1)//stride + 1
        n_w = (w - (ps-1)*dil - 1)//stride + 1
    else:
        t,c,h,w = vshape
        n_h = (h - 1)//stride + 1
        n_w = (w - 1)//stride + 1
    return n_h,n_w