
# -- local imports --
from .non_local_gather_gt import non_local_gather
from . import data
from . import find_duplicate_inds as find_duplicate_inds_f
from . import gradcheck

find_duplicate_inds = find_duplicate_inds_f.run



# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#              Misc Functions
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import torch as th


def check_shuffled_inds(inds_gt,inds_te,eps=1e-3):
    inds_gt,inds_te = 1.*inds_gt,1.*inds_te
    args = th.where(th.mean(th.abs(inds_gt-inds_te),dim=-1)>eps)
    i0,i1 = [],[]
    for i in range(3):
        i0.append(inds_gt[...,i][args])
        i1.append(inds_te[...,i][args])
    i0 = th.stack(i0,-1)
    i1 = th.stack(i1,-1)
    cdist = th.cdist(i0[None,:],i1[None,:])**2/3.
    idiffs = th.cdist(i0[None,:],i1[None,:])[0]
    mins = th.min(idiffs,1).values
    diff = th.mean(mins).item()
    return diff < eps


def int_spaced_vid(B,T,F,H,W):
    from einops import rearrange
    from positional_encodings.torch_encodings import PositionalEncoding3D

    # device = "cuda:0"
    # dtype = th.float32
    # grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
    #                              th.arange(0, W, dtype=dtype, device=device))
    # grid = th.stack((grid_x, grid_y), 0).float()[None,:]  # 2, W(x), H(y)
    # vid = []
    # assert (F>1) and (F % 2 == 1),"Odd features."
    # for ti in range(T):
    #     g0 = 10*grid[:,[0]].repeat(B,(F-1)//2,1,1)/W
    #     g1 = 10*grid[:,[1]].repeat(B,(F-1)//2,1,1)/H
    #     # g0 += th.rand_like(g0)
    #     # g1 += th.rand_like(g1)
    #     tN = (ti+1)*(th.ones_like(g0)/T)
    #     frame = th.cat([0.5*tN*g0,1.1*tN*g1,tN],1)
    #     # print(frame.shape)
    #     vid.append(frame) # noise less than int
    # vid = th.stack(vid,1)
    # # print("vid.shape: ",vid.shape)
    p_enc_3d = PositionalEncoding3D(F)
    vid = th.ones((B,T,F,H,W)).to("cuda:0")
    vid = rearrange(vid,'b t c h w -> b h w t c')
    vid = p_enc_3d(vid)
    vid = rearrange(vid,'b h w t c -> b t c h w')
    return vid

