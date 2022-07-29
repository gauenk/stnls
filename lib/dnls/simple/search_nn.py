
# -- python-only kernel --
from numba import cuda,jit
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- padding --
from dnls.utils.pads import same_padding,comp_pads

# -- fold/unfold
from torch.nn.functional import fold,unfold,pad,softmax,log_softmax

def run_nn(vid,ps,stride=4,dilation=1,mode="reflect",vid1=None,vid2=None):
    if vid1 is None: vid1 = vid
    if vid2 is None: vid2 = vid
    dil = dilation
    vid_pad_s,_ = same_padding(vid,ps,stride,dil,mode)
    patches_s = unfold(vid_pad_s,ps,stride=stride,dilation=dil) # t (c h w) n
    vid_pad_1,_ = same_padding(vid1,ps,1,dil,mode)
    patches_1 = unfold(vid_pad_1,ps,stride=1,dilation=dil) # t (c h w) n
    patches_s = rearrange(patches_s,'t d n -> t n d')
    patches_1 = rearrange(patches_1,'t d n -> t n d')
    score = th.cdist(patches_s,patches_1)**2

    return score[0]
