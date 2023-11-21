
import random
import numpy as np
import torch as th
import pickle
from einops import rearrange

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def optional_delete(pydict,key):
    if pydict is None: return
    elif key in pydict: del pydict[key]
    else: return

def assert_nonan(tensor):
    assert th.any(th.isnan(tensor)).item() is False

def rslice(vid,coords):
    if coords is None: return vid
    if len(coords) == 0: return vid
    if th.is_tensor(coords):
        coords = coords.type(th.int)
        coords = list(coords.cpu().numpy())
    fs,fe,t,l,b,r = coords
    return vid[fs:fe,:,t:b,l:r]

def write_pickle(fn,obj):
    with open(str(fn),"wb") as f:
        pickle.dump(obj,f)

def read_pickle(fn):
    with open(str(fn),"rb") as f:
        obj = pickle.load(f)
    return obj

def get_space_grid(H,W,dtype=th.float,device="cuda"):
    # -- create mesh grid --
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                 th.arange(0, W, dtype=dtype, device=device))
    grid = th.stack((grid_x, grid_y), -1).float()[None,:]  # 2, W(x), H(y)
    grid.requires_grad = False
    return grid

def flip_flows(flows_k,T,H,W):
    B,HD,T,nH,nW,K,three = flows_k.shape
    assert three == 3,"Must be three."
    return -flows_k

def flow2inds(flow,stride0):
    device = flow.device
    B = flow.shape[0]
    ndim = flow.ndim
    if ndim == 7:
        flow = rearrange(flow,'b hd t nh nw k tr -> (b hd) t nh nw k tr')
    _,T,nH,nW,K,three = flow.shape
    space_grid = stride0*get_space_grid(nH,nW).to(device)
    # print(space_grid.shape,space_grid[:,None,:,:,None].shape)
    inds = flow.clone()
    inds[...,1:] = flow[...,1:] + space_grid[:,None,:,:,None].flip(-1)
    inds[...,0] = flow[...,0] + th.arange(T).view(1,T,1,1,1).to(device)

    if ndim == 7:
        inds = rearrange(inds,'(b hd) t nh nw k tr -> b hd t nh nw k tr',b=B)

    return inds

def inds2flow(inds,stride0):
    device = inds.device
    B = inds.shape[0]
    ndim = inds.ndim
    if ndim == 7:
        inds = rearrange(inds,'b hd t nh nw k tr -> (b hd) t nh nw k tr')
    _,T,nH,nW,K,three = inds.shape
    space_grid = stride0*get_space_grid(nH,nW).to(device)
    flow = inds.clone()
    flow[...,1:] = inds[...,1:] - space_grid[:,None,:,:,None].flip(-1)
    flow[...,0] = inds[...,0] - th.arange(T).view(1,T,1,1,1).to(device)

    if ndim == 7:
        flow = rearrange(flow,'(b hd) t nh nw k tr -> b hd t nh nw k tr',b=B)

    return flow

