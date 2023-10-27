

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
    inds[...] = -1
    return dists,inds

def allocate_pair_2d(base_shape,device,dtype,idist_val,itype_str):
    dists = th.zeros(base_shape,device=device,dtype=dtype)
    dists[...] = idist_val
    inds = th.zeros(base_shape+(2,),device=device,dtype=get_itype(itype_str))
    inds[...] = -1
    return dists,inds

def allocate_inds(base_shape,device,itype_str):
    inds = th.zeros(base_shape+(3,),device=device,dtype=get_itype(itype_str))
    inds[...] = -1
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
    K = inds.shape[-2] if k is None else k
    kr = K if kr is None else kr
    if kr <= 0: return inds
    if isinstance(kr,float):
        assert (0 < kr and kr <= 1)
        Ks = int(K*kr)
    else: Ks = int(kr)
    return inds[...,:Ks,:].contiguous()


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
