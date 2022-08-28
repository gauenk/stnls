"""

Baseline for Window Search from Uformer

"""

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- misc --
import torch.nn.functional as nnf

def run(vid,win_size=8,nheads=4):
    t,c,h,w = vid.shape
    windows = window_partition(vid,win_size)
    windows = window_attn(windows,nheads=nheads,scale=10)
    vid = window_reverse(windows,win_size,t,h,w)
    return vid

def qkv(x,nheads):
    xh = rearrange(x,'t hw (H c) -> t H hw c',H=nheads)
    return xh,xh,xh

def window_partition(vid, win_size):
    # -- reshape --
    t,c,h,w = vid.shape
    vid = rearrange(vid,'t c h w -> t h w c')

    # -- view --
    vid = vid.view(t, h // win_size, win_size, w // win_size, win_size, c)
    windows = vid.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, win_size, win_size, c) # B' ,Wh ,Ww ,C

    return windows

def window_reverse(windows, win_size, t, h, w):
    windows = windows.view(t, h // win_size, w // win_size, win_size, win_size, -1)
    vid = windows.permute(0, 1, 3, 2, 4, 5)
    vid = vid.contiguous().view(t, h, w, -1)
    vid = rearrange(vid,'t h w c -> t c h w')
    return vid

def window_attn(x,nheads=4,scale=10):
    # -- attn --
    x = rearrange(x,'b nh nw c -> b (nh nw) c')
    B_, N, C = x.shape
    q, k, v = qkv(x,nheads)
    q = q * scale
    attn = (q @ k.transpose(-2, -1))
    attn = nnf.softmax(attn,-1)
    x = (attn @ v).transpose(1, 2)
    x = x.reshape(B_, N, C)
    return x
