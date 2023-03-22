

"""

Get/Compare the patches from optical flow to assess their quality for nls

"""

import torch as th
import dnls
from einops import rearrange
from easydict import EasyDict as edict

# -- api --
def get_patches(vid,flows,ps):

    patches = edict()
    UnfoldK = dnls.UnfoldK(ps)
    for f in ["fflow","bflow"]:
        
        # -- non-local indices --
        inds = flow2inds(flows[f],f)

        # -- patches --
        patches[f] = UnfoldK(vid,inds)

    return patches

# -- api --
def get_mse(flows):
    patches = get_patches(flows)
    mse = th.mean((patches[:,:,0] - patches[:,:,1])**2)
    return mse

def flow2inds(flow,fdir):

    # -- init --
    B,T,_,H,W = flow.shape
    raster = get_raster_inds(flow)
    inds = th.zeros((B,T-1,2,3,H,W),device="cuda:0")
    # print(flow[0,:,:3,:3])
    flow = th.flip(flow,(2,))
    # print(flow[0,:,:3,:3])

    # -- frame 0 is identity --
    for t in range(T-1):

        if fdir == "fflow":
            t_curr = t
            t_next = t+1
        else:
            t_curr = t+1
            t_next = t

        inds[:,t,0,0] = t_curr
        inds[:,t,0,1:] = raster

        inds[:,t,1,0,:,:] = t_next
        inds[:,t,1,1:,:,:] = flow[:,t_curr]+raster+0.5

    inds = rearrange(inds,'b s k tr h w -> b (s h w) k tr').contiguous()
    inds = inds.type(th.int32)
    return inds

def get_raster_inds(vid):
    B,T,_,H,W = vid.shape
    inds = th.zeros((B,2,H,W),device="cuda:0")
    inds[:,0,:,:] = th.arange(H)[:,None]
    inds[:,1,:,:] = th.arange(W)[None,:]
    # print(inds[:,:3,:3])
    # print(inds[:,-3:,-3:])
    return inds

