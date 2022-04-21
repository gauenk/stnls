
import torch as th
from einops import rearrange,repeat

def get_query_batch(index,qSize,qStride,h,w,device):
    ti32 = th.int32
    start = index * qSize
    stop = ( index + 1 ) * qSize
    srch_inds = th.arange(start,stop,dtype=ti32,device=device)[:,None]
    srch_inds = get_3d_inds(srch_inds,h,w)
    srch_inds = srch_inds.contiguous()
    return srch_inds

def get_3d_inds(inds,h,w):

    # -- unpack --
    hw = h*w # no "chw" in this code-base; its silly.
    bsize,num = inds.shape
    device = inds.device

    # -- shortcuts --
    tdiv = th.div
    tmod = th.remainder

    # -- init --
    aug_inds = th.zeros((3,bsize,num),dtype=th.int64)
    aug_inds = aug_inds.to(inds.device)

    # -- fill --
    aug_inds[0,...] = tdiv(inds,hw,rounding_mode='floor') # inds // chw
    aug_inds[1,...] = tdiv(tmod(inds,hw),w,rounding_mode='floor') # (inds % hw) // w
    aug_inds[2,...] = tmod(inds,w)
    aug_inds = rearrange(aug_inds,'three b n -> (b n) three')

    return aug_inds
