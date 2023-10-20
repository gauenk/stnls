"""

Shared Logical Units

"""

import stnls
import torch as th
from einops import rearrange
from torch.nn import functional as F

def manage_self(dists,inds,anchor_self,remove_self,qshift,stride0,H,W):
    assert not(remove_self and anchor_self)
    if remove_self:
        # B,HD,T,nH,nW,W_t,ws,ws = dists.shape
        dists = dists[...,1:,:,:]
        inds = inds[...,1:,:,:,:]
        return dists,inds
    if anchor_self:
        B,HD,T,nH,nW,W_t,ws,ws = dists.shape
        dists = dists.view(B,HD,Q,-1)
        d2or3 = inds.shape[-1]
        inds = inds.view(B,HD,Q,-1,d2or3)
        order = stnls.nn.anchor_self(dists,inds,stride0,H,W,qshift)[-1]
        dists=dists.reshape(B,HD,T,nH0,nW0,W_t,ws*ws)
        inds=inds.reshape(B,HD,T,nH0,nW0,W_t,ws*ws,d2or3)
    return dists,inds


# def manage_self(dists,inds,kselect,anchor_self,remove_self,wr):
#     assert not(remove_self and anchor_self)
#     if remove_self:
#         dists=dists.view(B,HD,Q,Ks,wr*wr)
#         inds=inds.view(B,HD,Q,Ks,wr*wr,3)
#         dists = dists[...,wr*wr:]
#         inds = inds[...,wr*wr:,:]
#         kselect = kselect[...,wr*wr:,:]


#     if anchor_self:
#         B,HD,T,nH,nW,W_t,ws2 = dists.shape
#         dists = dists.view(B,HD,Q,-1)
#         d2or3 = inds.shape[-1]
#         inds = inds.view(B,HD,Q,-1,d2or3)
#         stnls.nn.anchor_self(dists,inds,stride0,H,W,qshift)
#         dists=dists.reshape(B,HD,T,nH0,nW0,W_t,ws*ws)
#         inds=inds.reshape(B,HD,T,nH0,nW0,W_t,ws*ws,d2or3)
#     return dists,inds


def normz_bwd(ctx,grad_vid0,grad_vid1):
    # -- normz --
    # from torch.nn.functional import fold
    # from torchvision.transforms.functional import center_crop
    # from stnls import iFoldz
    assert int(ctx.stride1) == ctx.stride1,"stride1 must be an int."


    # -- unpack --
    B,T,C,H,W = grad_vid0.shape
    nH0 = (H-1)//ctx.stride0+1
    nW0 = (W-1)//ctx.stride0+1
    nH1 = (H-1)//ctx.stride1+1
    nW1 = (W-1)//ctx.stride1+1
    ps = ctx.ps
    reflect_bounds = ctx.reflect_bounds
    dilation = ctx.dil

    # -- normalize grad_vid0 --
    counts = th.ones((1,ctx.ps*ctx.ps,nH0*nW0),device=grad_vid0.device)
    pad = (ps-1)//2
    Hp,Wp = H+2*pad,W+2*pad
    # Hp = (nH0 - 1) * ctx.stride0 + ctx.ps
    # Wp = (nW0 - 1) * ctx.stride0 + ctx.ps
    sH,sW = (Hp-H+1)//2,(Wp-W+1)//2
    counts = F.fold(counts, (Hp,Wp), [ctx.ps]*2, 1, [0,0], ctx.stride0)
    counts = counts[...,sH:sH+H,sW:sW+W]
    grad_vid0 /= counts

    # -- normalize grad_vid1 --
    counts = th.ones((1,ctx.ps*ctx.ps,nH1*nW1),device=grad_vid0.device)
    pad = (ps-1)//2
    Hp,Wp = H+2*pad,W+2*pad
    # Hp = (nH0 - 1) * ctx.stride0 + ctx.ps
    # Wp = (nW0 - 1) * ctx.stride0 + ctx.ps
    sH,sW = (Hp-H+1)//2,(Wp-W+1)//2
    counts = F.fold(counts, (Hp,Wp), [ctx.ps]*2, 1, [0,0], ctx.stride1)
    counts = counts[...,sH:sH+H,sW:sW+W]
    grad_vid1 /= counts


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#      Fold/Unfold Utils
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# -- full fold (removes padding) --
def run_fold(patches,H,W,ps,stride,dilation):

    # -- unpack --
    pad = dilation*((ps-1)//2)
    Hp,Wp = H+2*pad,W+2*pad

    # -- folded --
    vid_pad = F.fold(patches,(Hp,Wp),(ps,ps),stride=stride,dilation=dilation)
    vid = vid_pad[:,:,pad:pad+H,pad:pad+W]

    # -- weigthed vid --
    ones = th.ones_like(patches)
    wvid_pad = F.fold(ones,(Hp,Wp),(ps,ps),stride=stride,dilation=dilation)
    wvid = wvid_pad[:,:,pad:pad+H,pad:pad+W]

    return vid,wvid

# -- full unfold (requires padding) --
def run_unfold(imgs,ps,stride,dilation,reflect_bounds):
    mode = "reflect" if reflect_bounds else "constant"
    imgs = pad_video(imgs,ps,stride,dilation,mode)
    patches = F.unfold(imgs,ps,stride=stride)
    return patches

def pad_video(vid,ps,stride,dil,mode):
    # -- reflect to include ps//2 around edges if needed --
    pad = dil*(ps//2)
    pads = (pad,pad,pad,pad)
    vid = F.pad(vid,pads,mode=mode)
    return vid


