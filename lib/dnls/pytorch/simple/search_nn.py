
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
from ...utils.pads import same_padding,comp_pads

# -- fold/unfold
from torch.nn.functional import fold,unfold,pad,softmax,log_softmax

def run_nn_batch(vid,ps,stride=4,dilation=1,mode="reflect",
                 vid1=None,vid2=None,stride1=1):
    B = vid.shape[0]
    scores = []
    for b in range(B):
        vid1_b = None if vid1 is None else vid1[b]
        vid2_b = None if vid2 is None else vid2[b]
        scores_b = run_nn(vid,ps,stride,dilation,mode,
                          vid1=vid1_b,vid2=vid2_b,stride1=stride1)
        scores.append(scores_b)
    scores = th.stack(scores)
    return scores

def run_nn(vid,ps,stride=4,dilation=1,mode="reflect",vid1=None,vid2=None,stride1=1):
    if vid1 is None: vid1 = vid
    if vid2 is None: vid2 = vid
    dil = dilation
    vid_pad_s,_ = same_padding(vid,ps,stride,dil,mode)
    patches_s = unfold(vid_pad_s,ps,stride=stride,dilation=dil) # t (c h w) n
    vid_pad_1,_ = same_padding(vid1,ps,stride1,dil,mode)
    patches_1 = unfold(vid_pad_1,ps,stride=stride1,dilation=dil) # t (c h w) n

    # -- cdist --
    patches_s = rearrange(patches_s,'t d n -> t n d')
    patches_1 = rearrange(patches_1,'t d n -> t n d')
    score = th.cdist(patches_s,patches_1)**2

    # -- product --
    # patches_s = patches_s.permute(0, 2, 1)
    # score = th.matmul(patches_s,patches_1)


    return score[0]

