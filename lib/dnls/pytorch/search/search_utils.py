
# -- python --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- nn --
import torch.nn.functional as nnf

# -- padding --
from ...utils.pads import comp_pads
from .unique_topk import unique_topk

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

def get_topk_l2(l2_vals,l2_inds,vals,inds):
    return get_topk(l2_vals,l2_inds,vals,inds)

def get_topk_prod(l2_vals,l2_inds,vals,inds):

    # -- reshape exh --
    nq = l2_vals.shape[0]
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

def get_topk_prod_b(l2_vals,l2_inds,vals,inds):

    # -- view --
    B,Q = l2_vals.shape[:2]
    l2_vals = l2_vals.view(B*Q,-1)
    l2_inds = l2_inds.view(B*Q,-1,3)
    vals = vals.view(B*Q,-1)
    inds = inds.view(B*Q,-1,3)

    # -- topk fill --
    get_topk_prod(l2_vals,l2_inds,vals,inds)

def topk_with_anchor(dists_exh,inds_exh,dists,inds,self_dists,anchor_self):
    if anchor_self:
        # get_topk_prod(dists_exh,inds_exh,dists[:,1:],inds[:,1:])
        topk_anchor(dists,inds,self_dists,dists_exh,inds_exh)
    else:
        get_topk_prod(dists_exh,inds_exh,dists,inds)

def topk_anchor(dists,inds,self_dists,dists_exh,inds_exh):#,wt,ws_h,ws_w):

    # -- reshape exh --
    nq = dists_exh.shape[0]
    dists_exh = dists_exh.view(nq,-1)
    inds_exh = inds_exh.view(nq,-1,3)
    self_dists = self_dists.view(-1)

    # -- shape info --
    b,_ = dists_exh.shape
    _,k = dists.shape

    # -- fill nan --
    dists_exh[th.where(th.isnan(dists_exh))] = -th.inf # fix nan
    # print("dists.shape: ",dists.shape)
    # print("inds.shape: ",inds.shape)
    # print("self_dists.shape: ",self_dists.shape)
    # print("dists_exh.shape: ",dists_exh.shape)

    # -- take maxs --
    order = th.argsort(dists_exh,dim=1,descending=True)
    dists[:b,:] = th.gather(dists_exh,1,order[:,:k])
    for i in range(inds.shape[-1]):
        inds[:b,:,i] = th.gather(inds_exh[:,:,i],1,order[:,:k])

    # -- fill dists --
    dists[:b,0] = self_dists

def run_anchor_self(dists,inds,self_dists,dists_exh,inds_exh):#,wt,ws_h,ws_w):

    # -- shape --
    # st = 2*wt+1
    BQ = inds_exh.shape[0]

    # -- fill dists --
    dists[:,0] = self_dists.view(BQ)
    print("self_dists.shape: ",self_dists.shape)
    print("dists_exh.shape: ",dists_exh.shape)
    print("dists.shape: ",dists.shape)
    print("inds.shape: ",inds.shape)
    print("inds_exh.shape: ",inds_exh.shape)

    # -- fill inds --
    isinf = th.isinf(dists_exh)
    ispos = dists_exh>0
    args0 = th.where(th.logical_and(isinf,ispos))
    print(inds_exh)
    print("minmax: ",th.min(th.stack(args0)),th.max(th.stack(args0)))
    print(args0)
    # exit(0)
    inds_self = []
    for i in range(3):
        print("inds_exh[...,i].shape: ",inds_exh[...,i].shape)
        inds_i = inds_exh[...,i][args0].view(BQ)
        inds_self.append(inds_i)
    inds_self = th.stack(inds_self,-1)
    # print("inds_self.shape: ",inds_self.shape)
    inds[:,0] = inds_self
    th.cuda.synchronize()
    # c_st = wt
    # c_ws_h = ws_h//2
    # c_ws_w = ws_w//2
    # inds[:,0] = inds_exh[:,c_st,c_ws_h,c_ws_w]

def run_remove_self_cuda(dists,inds,qstart,stride,n_h,n_w):
    # print("dists.shape,inds.shape:" ,dists.shape,inds.shape,n_h,n_w)
    b,nq,k = dists.shape
    mask = th.zeros((b,nq,k),device=dists.device,dtype=th.bool)
    dnls_cuda.remove_self_from_search(inds,mask,qstart,stride,n_h,n_w)
    # th.cuda.synchronize()
    mask = th.logical_not(mask)
    # print(dists.shape)
    # print(mask.sum(1))
    # args = th.where(mask.sum(1)==k)
    # mask[args,-1] = 0
    # print(mask.sum(1))
    # print(th.all(mask.sum(1)==1))
    # rm_dists = th.masked_select(dists,mask)
    rm_dists = th.masked_select(dists,mask).view(b,nq,k-1)
    rm_inds = []
    for i in range(3): # |(t,h,w)| == 3
        rm_inds_i = th.masked_select(inds[...,i],mask).view(b,nq,k-1)
        rm_inds.append(rm_inds_i)
    rm_inds = th.stack(rm_inds,-1)

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
        bufs = th.zeros(nq,3,1,ws_h,ws_w,dtype=th.int32,device=device)
    else:
        st = min(t,2*wt+1)
        bufs = th.zeros(nq,3,st,ws_h,ws_w,dtype=th.int32,device=device)
    return bufs

def allocate_exh_prod(nq,wt,ws_h,ws_w,device,dtype=th.float32):
    dists = th.zeros((nq,2*wt+1,ws_h,ws_w),device=device,dtype=dtype)
    dists[...] = -float("inf")
    inds = th.zeros((nq,2*wt+1,ws_h,ws_w,3),device=device,dtype=th.int32)
    inds[...] = -1
    return dists,inds

def allocate_exh_l2(nq,wt,ws_h,ws_w,device):
    return allocate_exh(nq,wt,ws_h,ws_w,device)

def allocate_exh(nq,wt,ws_h,ws_w,device,dtype=th.float32):
    dists = th.zeros((nq,2*wt+1,ws_h,ws_w),device=device,dtype=dtype)
    dists[...] = float("inf")
    inds = th.zeros((nq,2*wt+1,ws_h,ws_w,3),device=device,dtype=th.int32)
    inds[...] = -1
    return dists,inds

def allocate_rtn_b(b,nq,k,device,dtype=th.float32):
    dists = th.zeros((b,nq,k),device=device,dtype=dtype)
    inds = th.zeros((b,nq,k,3),device=device,dtype=th.int32)
    return dists,inds

def allocate_rtn(nq,k,device,dtype=th.float32):
    dists = th.zeros((nq,k),device=device,dtype=dtype)
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

def create_window_partition(in_h,in_w,ws_h,ws_w,device):


    # -- add padding if needed --
    h_pad = ws_h - (in_h % ws_h)
    w_pad = ws_w - (in_w % ws_w)
    h = in_h + h_pad
    w = in_w + w_pad

    # -- create index image --
    img = np.arange(h*w).reshape(h,w)
    img_h = img // w
    img_w = img % w
    img = np.stack([img_h,img_w])

    # -- unravel to windows --
    shape_str = 'two (nh h) (nw w) -> two nh nw h w'
    img = rearrange(img,shape_str,h=ws_h,w=ws_w)
    _,nh,nw,_,_ = img.shape
    img = img.reshape(2,nh*nw,-1)

    # -- set each region to min --
    h_min = np.min(img[0,:,:],1)
    w_min = np.min(img[1,:,:],1)
    img[0,:,:] = h_min[:,None]
    img[1,:,:] = w_min[:,None]

    # -- finalize format --
    shape_str = 'two (nh nw) (h w) -> (nh h) (nw w) two'
    img = rearrange(img,shape_str,nh=nh,nw=nw,h=ws_h,w=ws_w)

    # -- to tensor --
    img = img.astype(np.int32)
    img = th.from_numpy(img).to(device)

    # -- remove padding --
    img = img[:in_h,:in_w]

    return img

def only_unique(dists,inds,k):

    # -- compute --
    B,K = dists.shape
    B,K,_ = inds.shape
    args = unique_topk(dists,k)

    # -- gather --
    dists_u = th.gather(dists,1,args)
    inds_u = []
    for i in range(3):
        inds_i = th.gather(inds[...,i],1,args)
        inds_u.append(inds_i)
    inds_u = th.stack(inds_u,-1)

    return dists_u,inds_u

def unique(x, dim=-1):
    x = (1000*x).type(th.int32)
    unique, inverse = th.unique_consecutive(x, return_inverse=True, dim=dim)
    # print("unique.shape: ",unique.shape)
    # print(unique)
    # print(unique[0,:10])
    # print(th.unique_consecutive(unique[0,:10],dim=dim))
    print(unique[:10,0])
    print(th.unique_consecutive(unique[:10,0],dim=dim))
    exit(0)
    perm = th.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    # return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)
    args = inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)
    return args


def upscale_inds(inds,stride,H,W):
    B,Q,K,_ = inds.shape
    nH = (H-1)//stride+1
    inds = rearrange(inds,'b (h w) k tr -> (b k) tr h w',h=nH)
    inds_t = inds[:,[0]].contiguous()
    inds_i = inds[:,1:].contiguous()
    inds_t = nnf.interpolate(inds_t,mode='nearest-exact',size=(H,W))
    inds_i = nnf.interpolate(inds_i,'bilinear',size=(H,W))
    inds = th.cat([inds_t,inds_i],1)
    inds = rearrange(inds,'(b k) tr h w -> b (h w) k tr')
    return inds
