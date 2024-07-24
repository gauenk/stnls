

import torch as th
from einops import rearrange
from easydict import EasyDict as edict

#
#
# -- Allocate Memory for Search --
#
#

def allocate_pair(base_shape,device,dtype,idist_val,itype_str):
    dists = th.zeros(base_shape,device=device,dtype=dtype)
    dists[...] = idist_val
    inds = th.zeros(base_shape+(3,),device=device,dtype=get_itype(itype_str))
    inds[...] = -1e8
    return dists,inds

def allocate_pair_2d(base_shape,device,dtype,idist_val,itype_str):
    dists = th.zeros(base_shape,device=device,dtype=dtype)
    dists[...] = idist_val
    inds = th.zeros(base_shape+(2,),device=device,dtype=get_itype(itype_str))
    inds[...] = -1e8
    return dists,inds

def allocate_inds(base_shape,device,itype_str):
    inds = th.zeros(base_shape+(3,),device=device,dtype=get_itype(itype_str))
    inds[...] = -1e8
    return inds

def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def get_ctx_shell(tensor,use_shell):
    if use_shell:
        device = tensor.device
        dtype = tensor.dtype
        tensor = th.zeros((1,)*tensor.ndim,device=device,dtype=dtype)
        return tensor
    else:
        return tensor


def get_ctx_flows_v0(itype,fflow,bflow):
    if itype == "int":
        device = fflow.device
        dtype = fflow.dtype
        flow = th.zeros((1,)*5,device=device,dtype=dtype)
        flow = th.zeros((1,)*5,device=device,dtype=dtype)
        return flow,flow
    else:
        return fflow,bflow

def get_ctx_qinds(itype,qinds):
    if itype == "int":
        device = qinds.device
        dtype = qinds.dtype
        enable_api = th.zeros((1,)*5,device=device,dtype=dtype)
        return enable_api
    else:
        return qinds

def allocate_grad_flows(itype,f_shape,device):
    if itype == "int":
        grad_fflow = th.zeros((1,)*7,device=device,dtype=th.float32)
    else:
        grad_fflow = th.zeros(f_shape,device=device,dtype=th.float32)
    return grad_fflow


# def allocate_grad_qinds(itype,ishape,device):
#     if itype == "int":
#         grad_qinds = th.zeros((1,)*5,device=device,dtype=th.int)
#     else:
#         B,HD,Q,K,_ = ishape
#         grad_qinds = th.zeros((B,HD,Q,K,3),device=device,dtype=th.float32)
#     return grad_qinds

def get_itype(itype_str):
    if itype_str in ["int","int32"]:
        return th.int32
    elif itype_str == "float":
        return th.float32
    else:
        raise ValueError(f"Uknown itype [{itype_str}]")

def get_inds(inds,itype):
    inds = inds.contiguous()
    if itype == "int" and th.is_floating_point(inds):
        return inds.round().int()
    elif itype == "float" and not(th.is_floating_point(inds)):
        return inds.float()
    else:
        return inds

def get_dists(inds,itype):
    inds = inds.contiguous()
    if itype == "int" and th.is_floating_point(inds):
        return inds.round().int()
    elif itype == "float" and not(th.is_floating_point(inds)):
        return inds.float()
    else:
        return inds



#
#
# -- Filtering Indices for Approximate Search Methods
#
#

def filter_k(inds,kr,k=None):
    # inds.shape = (B,HD,K,T,H,W,2or3)
    K = inds.shape[-2] if k is None else k
    # K = inds.shape[-5] if k is None else k
    kr = K if kr is None else kr
    if kr <= 0: return inds
    if isinstance(kr,float):
        assert (0 < kr and kr <= 1)
        Ks = int(K*kr)
    else: Ks = int(kr)
    return inds[...,:Ks,:].contiguous()
    # return inds[...,:Ks,:,:,:,:].contiguous()


def ensure_paired_flow_dim(flow,num):
    if flow.ndim == num:
        flow = flow[:,None] # add nheads
    assert flow.ndim == (num+1)
    return flow

def ensure_flow_shape(flow):
    if flow.ndim == 5:
        B,T,_,H,W = flow.shape
        # flow = flow.view(B,T,1,2,H,W)
        flow = flow.view(B,1,T,2,H,W)
    return flow
#
#
# -- Shaping input videos with Heads --
#
#

def shape_flows(nheads,flows):
    # B,T,W_t,2,H,W = flows.shape
    # B,HD,T,W_t,2,H,W = flows.shape
    ndim = flows.ndim
    if flows.ndim == 7:
        return flows
    elif flows.ndim == 6:
        return flows[:,None] # 1 head
    else:
        msg = f"Input flows are wrong dimension. Must be 6 or 7 but is [{ndim}]"
        raise ValueError(msg)

def shape_refinement_flows(nheads,flows,B,nH,nW):
    # print(flows.shape)
    if flows.ndim == 4:
        B,HD,Q,tr = flows.shape
        flows=rearrange(flows,'b hd (t nh nw) thr -> b hd t nh nw thr',nh=nH,nw=nW)
    # elif flows.ndim == 3:
    #     BHD,Q,tr = flows.shape
    #     shape_str = '(b hd) (t nh nw) k thr -> b hd t nh nw k thr'
    #     flows=rearrange(flows,,b=B,nh=nH,nw=nW)
    elif flows.ndim == 5:
        B,HD,Q,K,tr = flows.shape
        shape_str = 'b hd (t nh nw) k thr -> b hd t nh nw k thr'
        flows=rearrange(flows,shape_str,b=B,nh=nH,nw=nW)
    elif flows.ndim == 6:
        flows=rearrange(flows,'(b hd) t nh nw thr -> (b hd) t nh nw thr',b=B)
    assert flows.ndim == 7
    return flows


def shape_vids(nheads,vids):
    _vids = []
    for vid in vids:
        # -- reshape with heads --
        assert vid.ndim in [5,6], "Must be 5 or 6 dims."
        if vid.ndim == 5:
            c = vid.shape[2]
            assert c % nheads == 0,"must be multiple of each other."
            shape_str = 'b t (HD c) h w -> b HD t c h w'
            vid = rearrange(vid,shape_str,HD=nheads).contiguous()
        assert vid.shape[1] == nheads
        _vids.append(vid)
    return _vids

def shape_frames(nheads,vids):
    _vids = []
    for vid in vids:
        # -- reshape with heads --
        assert vid.ndim in [4,5], "Must be 4 or 5 dims."
        if vid.ndim == 4:
            c = vid.shape[1]
            assert c % nheads == 0,"must be multiple of each other."
            shape_str = 'b (HD c) h w -> b HD c h w'
            vid = rearrange(vid,shape_str,HD=nheads).contiguous()
        # assert vid.shape[1] == nheads
        _vids.append(vid)
    return _vids


# -- get empty flow --
def empty_flow(vid):
    b,t,c,h,w = vid.shape
    zflow = th.zeros((b,t,2,h,w),dtype=vid.dtype,device=vid.device)
    return zflow

#
#
# -- Handling Distance Type [Prod or L2] --
#
#

def dist_type_select(dist_type):
    dist_type_i = dist_menu(dist_type)
    descending = descending_menu(dist_type)
    dval = init_dist_val_menu(dist_type)
    return dist_type_i,descending,dval

def dist_menu(dist_type):
    menu = {"prod":0,"l2":1}
    return menu[dist_type]

def descending_menu(dist_type):
    menu = {"prod":True,"l2":False}
    return menu[dist_type]

def init_dist_val_menu(dist_type):
    menu = {"prod":-th.inf,"l2":th.inf}
    return menu[dist_type]

# #
# #
# # -- API Utils --
# #
# #

# def extract_pairs(pairs,_cfg):
#     cfg = edict()
#     for key,default in pairs.items():
#         if key in _cfg:
#             cfg[key] = _cfg[key]
#         else:
#             cfg[key] = pairs[key]
#     return cfg

#
# -- interface --
#

def search_kwargs_wrap(name,search):
    """
    All for all inputs to enable easier benchmarking

        All calls must use keywords!

      vid0,vid1,fflow,bflow,inds,afflow,abflow

    """
    if "refine" in name:
        def wrap(vid0,vid1,**kwargs):
            inds = kwargs['inds']
            return search(vid0,vid1,inds)
        return wrap
    elif "pf" in name:
        def wrap(vid0,vid1,**kwargs):
            afflow = kwargs['afflow']
            abflow = kwargs['abflow']
            return search(vid0,vid1,afflow,abflow)
        return wrap
    else:
        def wrap(vid0,vid1,**kwargs):
            fflow = kwargs['fflow']
            bflow = kwargs['bflow']
            return search(vid0,vid1,fflow,bflow)
        return wrap

def search_wrap(name,search):
    """
    All for all inputs to enable easier benchmarking

        All calls must include all variables!

      vid0,vid1,fflow,bflow,inds,afflow,abflow

    """
    if "refine" in name:
        def wrap(vid0,vid1,fflow,bflow,inds,afflow,abflow):
            return search(vid0,vid1,inds)
        return wrap
    elif "pf" in name:
        def wrap(vid0,vid1,fflow,bflow,inds,afflow,abflow):
            return search(vid0,vid1,afflow,abflow)
        return wrap
    else:
        def wrap(vid0,vid1,fflow,bflow,inds,afflow,abflow):
            return search(vid0,vid1,fflow,bflow)
        return wrap

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#              Paired Utils
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_time_window_inds(ti,wt,T):
    swap = False
    t_inc = 0
    prev_t = ti
    t_shift = min(0,ti-wt) + max(0,ti + wt - (T-1))
    t_max = min(T-1,ti + wt - t_shift);
    # print(t_shift,t_max)
    tj = ti
    inds = []
    for _tj in range(2*wt+1):
        # -- update search frame --
        prev_t = tj
        tj = prev_t + t_inc
        swap = tj > t_max
        t_inc = 1 if (t_inc == 0) else t_inc
        t_inc = -1 if swap else t_inc
        tj = ti-1 if swap else tj
        prev_t = ti if swap else prev_t
        # print(ti,tj,t_inc,swap)
        inds.append(tj)
    return inds

def get_flows(flows):
    assert flows.ndim in [6,7]
    if flows.ndim == 6:
        flows = flows[:,None]
    return flows

def paired_vids(forward, vid0, vid1, flows, wt, skip_self=False):
    dists,inds = [],[]
    T = vid0.shape[1]
    # print("[a] flows.shape: ",flows.shape)
    flows = get_flows(flows)
    # print("[b] flows.shape: ",flows.shape)
    # print(vid0.shape,flows.shape,wt)
    zflow = th.zeros_like(flows[:,:,0,0])
    for ti in range(T):
        t_grid = get_time_window_inds(ti,wt,T)
        dists_i,inds_i = [],[]
        for _tj in range(2*wt+1):
            # -- update search frame --
            tj = t_grid[_tj]
            if (ti == tj) and skip_self: continue
            # print(ti,tj,_tj-1)
            frame0 = vid0[:,ti]
            frame1 = vid1[:,tj]
            if _tj > 0: flow = flows[:,:,ti,_tj-1]
            else: flow = zflow
            flow = flow.float()
            dists_ij,inds_ij = forward(frame0,frame1,flow)
            # print("flow_ij.shape: ",inds_ij.shape)
            # exit()
            inds_t = (tj-ti)*th.ones_like(inds_ij[...,[0]])
            # inds_t = (tj-ti)*th.ones_like(inds_ij[...,[0],:,:])
            # print(inds_t.shape,inds_ij.shape)
            # inds_ij = th.cat([inds_t,inds_ij],-3)
            inds_ij = th.cat([inds_t,inds_ij],-1)
            dists_i.append(dists_ij)
            inds_i.append(inds_ij)
        # -- stack across K --
        # dists_i = th.cat(dists_i,-3)
        # inds_i = th.cat(inds_i,-4)
        dists_i = th.cat(dists_i,-1)
        inds_i = th.cat(inds_i,-2)
        dists.append(dists_i)
        inds.append(inds_i)
    # -- stack across time --
    dists = th.stack(dists,-4)
    inds = th.stack(inds,-5)
    # dists = th.stack(dists,-4)
    # inds = th.stack(inds,-4)
    return dists,inds

def paired_vids_refine(forward, vid0, vid1, flows, wt, skip_self=False, check_time=True):
    """

    Only really for testing...

    ... why? why not for use?

    """
    dists,inds = [],[]
    T = vid0.shape[1]
    # print(vid0.shape,flows.shape)
    flows = get_flows(flows)
    zflow = th.zeros_like(flows[:,:,0,0])
    K_total = flows.shape[-2]
    # print("utils: ",flows.shape)
    Wt = 2*wt+1
    Wt = Wt-1 if skip_self else Wt
    assert ((K_total % Wt) == 0),"Must be divisible by Wt."
    K_each = K_total // Wt
    for ti in range(T):
        t_grid = get_time_window_inds(ti,wt,T)
        dists_i,inds_i = [],[]
        ix = 0
        for _tj in range(2*wt+1):
            # -- update search frame --
            tj = t_grid[_tj]
            if (ti == tj) and skip_self:
                continue
            frame0 = vid0[:,ti]
            frame1 = vid1[:,tj]
            # ks0,ks1 = _tj*K_each,(_tj+1)*K_each
            ks0,ks1 = ix*K_each,(ix+1)*K_each
            flow = flows[:,:,ti,:,:,ks0:ks1,:].float()
            # print(flow.shape,ks0,ks1,_tj*K_each,(_tj+1)*K_each)
            if check_time:
                assert th.all(flow[...,0] == (tj-ti)),"Must all be same frame."
            dists_ij,inds_ij = forward(frame0,frame1,flow[...,1:])
            inds_t = (tj-ti)*th.ones_like(inds_ij[...,[0]])
            inds_ij = th.cat([inds_t,inds_ij],-1)
            dists_i.append(dists_ij)
            inds_i.append(inds_ij)
            ix+=1
        # -- stack across K --
        dists_i = th.cat(dists_i,-1)
        inds_i = th.cat(inds_i,-2)
        dists.append(dists_i)
        inds.append(inds_i)
    # -- stack across time --
    dists = th.stack(dists,-4)
    inds = th.stack(inds,-5)
    # print("inds.shape: ",inds.shape)
    return dists,inds

def paired_vids_old(forward, vid0, vid1, acc_flows, wt, skip_self=False):
    dists,inds = [],[]
    T = vid0.shape[1]
    zflow = th.zeros_like(acc_flows.fflow[:,0,0])
    for ti in range(T):
        # if ti != 1: continue

        swap = False
        t_inc = 0
        prev_t = ti
        t_shift = min(0,ti-wt) + max(0,ti + wt - (T-1))
        t_max = min(T-1,ti + wt - t_shift);
        # print(t_shift,t_max)
        tj = ti

        dists_i,inds_i = [],[]
        for _tj in range(2*wt+1):

            # -- update search frame --
            prev_t = tj
            tj = prev_t + t_inc
            swap = tj > t_max
            t_inc = 1 if (t_inc == 0) else t_inc
            t_inc = -1 if swap else t_inc
            tj = ti-1 if swap else tj
            prev_t = ti if swap else prev_t
            # print(ti,tj,t_inc,swap)

            frame0 = vid0[:,ti]
            frame1 = vid1[:,tj]
            if (ti == tj) and skip_self: continue
            if ti == tj:
                flow = zflow
            elif ti < tj:
                # print("fwd: ",ti,tj,tj-ti-1)
                # flow = acc_flows.fflow[:,tj - ti - 1]
                flow = acc_flows.fflow[:,ti,tj-ti-1]
            elif ti > tj:
                # print("bwd: ",ti,tj,ti-tj-1)
                # flow = acc_flows.bflow[:,ti - tj - 1]
                flow = acc_flows.bflow[:,ti,ti-tj-1]
            flow = flow.float()
            dists_ij,inds_ij = forward(frame0,frame1,flow)
            inds_t = tj*th.ones_like(inds_ij[...,[0]])
            inds_ij = th.cat([inds_t,inds_ij],-1)
            # print("inds_ij.shape: ",inds_ij.shape,inds_t.shape)
            dists_i.append(dists_ij)
            inds_i.append(inds_ij)
        dists_i = th.cat(dists_i,-1)
        inds_i = th.cat(inds_i,-2)
        dists.append(dists_i)
        inds.append(inds_i)
    dists = th.cat(dists,-2)
    inds = th.cat(inds,-3)
    # print("inds.shape: ",inds.shape)
    return dists,inds

