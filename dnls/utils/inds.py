
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- numba --
import numba
from numba import njit,prange

def get_exh_inds(vid,stride=1):
    t,c,h,w = vid.shape
    qSearch = t*h*w // stride
    return get_query_batch(0,qSearch,stride,t,h,w,vid.device)

def get_query_batch(index,qSearch,qStride,t,h,w,device):
    srch_inds = numba_query_launcher(index,qSearch,qStride,t,h,w,device)
    return srch_inds
    # ti32 = th.int32
    # start = index * qSearch
    # stop = ( index + 1 ) * qSearch
    # srch_inds = th.arange(start,stop,dtype=ti32,device=device)[:,None]
    # srch_inds = get_3d_inds(srch_inds,qStride,t,h,w)
    # srch_inds = srch_inds.contiguous()
    # return srch_inds

def numba_query_launcher(index,qSearch,qStride,t,h,w,device):
    srch_inds = np.zeros((qSearch,3),dtype=np.int64)
    numba_query_batch(srch_inds,index,qSearch,qStride,t,h,w)
    srch_inds = th.from_numpy(srch_inds).to(device).contiguous()
    return srch_inds

@njit
def numba_query_batch(srch_inds,index,qSearch,qStride,t,h,w):
    qSearchTotal_t = h*w//qStride
    qStride_sr = np.sqrt(qStride)
    nT = qSearchTotal_t
    wf = w*1.
    # How to evenly distribution points in a grid? I.E. how to use "stride"
    nX = np.sqrt((wf/h)*nT + (wf-h)**2/(4.*h**2)) - (wf-h)/(2.*h)
    nX = int(nX)
    start = index * qSearch
    for qi in prange(qSearch):

        # -- ind -> ti --
        ind = start + qi
        ti = ind // qSearchTotal_t
        ind = ind %  qSearchTotal_t

        # -- ind -> hi --
        hi = qStride_sr*(ind // nX)
        hi = int(hi)
        if hi >= h:
            hi = h-1

        # -- ind -> hi --
        wi = qStride_sr*(ind % nX)
        wi = int(wi)
        if wi >= w:
            wi = w-1

        # -- fill --
        srch_inds[qi,0] = ti
        srch_inds[qi,1] = hi
        srch_inds[qi,2] = wi

def get_3d_inds(inds,stride,t,h,w):

    # -- unpack --
    hw = h*w
    bsize,num = inds.shape
    device = inds.device
    qSearchTotal_t = h*w//stride

    # -- shortcuts --
    tdiv = th.div
    tmod = th.remainder

    # -- init --
    aug_inds = th.zeros((3,bsize,num),dtype=th.int64)
    aug_inds = aug_inds.to(inds.device)

    # -- fill frame index --
    aug_inds[0,...] = tdiv(inds,qSearchTotal_t,rounding_mode='floor') # t = inds // hw

    # -- correct for stride offsets --
    delta = compute_stride_offsets(stride,t,h,w,device)
    inds_mod = tmod(inds,qSearchTotal_t)
    for ti in range(t):
        args = th.where(ti == aug_inds[0])
        print(ti,len(args[0]))
        # print("[0]: ",inds[args][0])
        # print("[a]: ",inds_mod[args][0])
        inds_mod[args] -= delta[ti]
        print("[b]: ",inds_mod[args])
        # print("[c]: ",inds_mod[args][0]*stride)
    exit(0)

    # -- fill spatial inds --
    aug_inds[1,...] = tdiv(inds_mod,w,rounding_mode='floor') # (inds % hw) // w
    aug_inds[2,...] = tmod(inds_mod,w)
    aug_inds = rearrange(aug_inds,'three b n -> (b n) three')

    return aug_inds

def compute_stride_offsets(stride,t,h,w,device):
    assert stride < h and stride < w
    delta = th.zeros(t,device=device,dtype=th.int32)
    hw = h*w
    qSearchTotal_t = hw//stride
    for ti in range(1,t):
        final_ind = (ti*stride*qSearchTotal_t) % hw
        delta[ti] = (hw - final_ind) % stride
    return delta


