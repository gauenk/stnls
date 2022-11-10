"""

Simple version of Patch Fully-Connected

"""


# -- torch imports --
import torch as th
import torch.nn as nn
from torch.nn.functional import fold,unfold,pad
from einops import rearrange,repeat

# def run(vid0,vid1,stride0,stride1,ps,weights,bias):
def run(vid,stride,ps,fc_layer,dil=1):

    # -- reshape --
    B = vid.shape[0]
    device = vid.device

    # -- rearrange --
    vid = rearrange(vid,'b t c h w -> (b t) c h w')
    T,C,H,W = vid.shape

    # -- unfold --
    patches = _unfold(vid,stride,ps)
    # print("patches.shape: ",patches.shape)

    # -- viz --
    # print(patches[0,:49,0].view(7,7))
    # print(patches[0,:49,1].view(7,7))
    # print(patches[0,:49,2].view(7,7))

    # -- xform --
    patches = patches.transpose(2,1)
    xformed = fc_layer(patches)
    # xformed = patches
    xformed = xformed.transpose(2,1)
    # print("xformed.shape: ",xformed.shape)

    # -- viz --
    # print(xformed[0,:49,0].view(7,7))
    # print(xformed[0,:49,1].view(7,7))
    # print(xformed[0,:49,2].view(7,7))

    # -- fold --
    vid_out = _fold(xformed,ps,H,W,device,stride)

    # -- reshape --
    vid_out = rearrange(vid_out,'(b t) c h w -> b t c h w',b=B)
    return vid_out

def _fold(patches,ps,H,W,device,stride,dil=1,full_pads=False):
    ipad = (ps)//2 if full_pads else 0
    # print("ipad: ",ipad)
    H,W = H+2*ipad,W+2*ipad
    # print("[fold] patches.shape: ",patches.shape,H,W,ps,pad,H,W,stride)
    ones = th.ones_like(patches)
    vid_pad = fold(patches,(H,W),(ps,ps),stride=stride,dilation=dil)
    ones_pad = fold(ones,(H,W),(ps,ps),stride=stride,dilation=dil)
    # print("ipad: ",ipad)
    # print("vid_pad.shape: ",vid_pad.shape)
    if ipad > 0:
        vid = vid_pad[...,ipad:-ipad,ipad:-ipad] / ones_pad[...,ipad:-ipad,ipad:-ipad]
    else:
        vid = vid_pad / ones_pad
    return vid

def _unfold(vid,stride,ps,full_pads=False):
    ipad = (ps)//2 if full_pads else 0
    # print("ipad: ",ipad)
    pads = [ipad,]*4
    vid_pad = pad(vid,pads,mode="reflect")
    patches = unfold(vid_pad,(ps,ps),stride=stride)
    return patches

