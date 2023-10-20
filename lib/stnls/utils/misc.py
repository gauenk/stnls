
import random
import numpy as np
import torch as th
import pickle

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
    grid = th.stack((grid_x, grid_y), 0).float()[None,:]  # 2, W(x), H(y)
    grid.requires_grad = False
    return grid

def flow2inds(flow,stride0):
    B,T,nH,nW,three = flow.shape
    space_grid = stride0*get_space_grid(nH,nW)
    inds = flow.clone()
    inds = flow[...,1:] + space_grid
    inds = flow[...,0] + th.arange(T).view(1,T,1,1)
    return inds
