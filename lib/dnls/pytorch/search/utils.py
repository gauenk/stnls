

import torch as th
from einops import rearrange
from easydict import EasyDict as edict

#
#
# -- Allocate Memory for Search --
#
#

def allocate_pair(base_shape,device,dtype,idist_val):
    dists = th.zeros(base_shape,device=device,dtype=dtype)
    dists[...] = idist_val
    inds = th.zeros(base_shape+(3,),device=device,dtype=th.int32)
    inds[...] = -1
    return dists,inds

def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

#
#
# -- Filtering Indices for Approximate Search Methods
#
#

def filter_k(inds,kr,k=None):
    K = inds.shape[-2] if k is None else k
    if kr <= 0: return inds
    if isinstance(kr,float):
        assert (0 < kr and kr <= 1)
        Ks = int(K*kr)
    else: Ks = int(kr)
    return inds[...,:Ks,:].contiguous()

#
#
# -- Shaping input videos with Heads --
#
#

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

#
#
# -- API Utils --
#
#

def extract_pairs(pairs,_cfg):
    cfg = edict()
    for key,default in pairs.items():
        if key in _cfg:
            cfg[key] = _cfg[key]
        else:
            cfg[key] = pairs[key]
    return cfg

#
# -- interface --
#

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