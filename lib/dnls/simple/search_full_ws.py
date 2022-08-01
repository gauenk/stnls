"""
Show the correct indexing for full_ws

"""

# -- python-only kernel --
from numba import cuda,jit
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat


def run(vid0,ws,stride,ti,hi,wi,fill_val=5.):

    # -- unpack --
    t,c,h,w = vid0.shape
    vid2fill = th.zeros_like(vid0)
    wsHalf = ws//2

    # -- compute top,left of square --
    print("hi,wi: ",hi,wi)
    wsOff_h = (hi-max(hi-stride*wsHalf,0))//stride;
    wsOff_w = (wi-max(wi-stride*wsHalf,0))//stride;
    print(wsOff_h,wsOff_w)
    print(hi+stride*(-wsOff_h),hi+stride*(ws-wsOff_h-1),min(hi+stride*(ws-wsOff_h-1),h-1))
    print("misc: ",((-1)//stride)+1,stride,int(-1./stride),-1//stride)

    if hi+stride*(ws-wsOff_h-1) >= h:
        wsOff_h += int((hi+stride*(ws-wsOff_h-1) - min(hi+stride*(ws-wsOff_h-1),h-1)-1.)/stride)+1
    if wi+stride*(ws-wsOff_w-1) >= w:
        wsOff_w += int((wi+stride*(ws-wsOff_w-1) - min(wi+stride*(ws-wsOff_w-1),w-1)-1.)/stride)+1
    print(wsOff_h,wsOff_w)
    print(hi+stride*(-wsOff_h),hi+stride*(ws-wsOff_h-1),min(hi+stride*(ws-wsOff_h-1),h-1))

    # -- fill --
    for ti in range(t):
        for ws_i in range(ws):
            for ws_j in range(ws):
                hk = hi + stride*(ws_i - wsOff_h)
                wk = wi + stride*(ws_j - wsOff_w)
                # print(ws_i,ws_j,hk,wk)
                vid2fill[ti,:,hk,wk] = 1.
    vid2fill[ti,:,hi,wi] = fill_val

    # -- normalize --
    vid2fill /= vid2fill.max()

    return vid2fill
