
# -- python-only kernel --
from numba import cuda,jit

# # -- [for testing] delete me. --
# from dnls.utils import color

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- scatter --
from .scatter_bwd import run_bwd

def run(vid,inds,ps,pt=1,dilation=1):

    # -- allocate patches --
    patches = allocate_patches(vid,inds,ps,pt)

    # -- exec scatter --
    numba_launcher(patches,vid,inds,dilation)

    return patches


def allocate_patches(vid,inds,ps,pt):
    # -- device --
    device = inds.device
    assert pt == 1

    # -- unpack shapes --
    t,c,h,w = vid.shape
    nq,k,three = inds.shape

    # -- patches --
    pshape = (nq,k,pt,c,ps,ps)
    patches = th.zeros(pshape,device=device,dtype=th.float32)
    return patches

def numba_launcher(patches,vid,inds,dilation):

    # -- numbify all params --
    patches_nba = cuda.as_cuda_array(patches)
    vid_nba = cuda.as_cuda_array(vid)
    inds_nba = cuda.as_cuda_array(inds)

    # -- kernel blocks --
    nq,k,pt,c,ps,ps = patches.shape
    qpb = 10 # queries per block
    nblocks = (nq-1)//qpb+1

    # -- kernel threads --
    MAX_THREADS = 1024
    dim = ps*ps
    n_kthreads = MAX_THREADS//dim # number of "k" managed per block
    kpt = max(k - n_kthreads,1) # remaining "k" per thread
    nthreads = (n_kthreads,ps,ps)

    # -- exec kernel --
    numba_scatter[nblocks,nthreads](patches_nba,vid_nba,inds_nba,dilation,kpt,qpb)

# -- reflect padding --
@cuda.jit(debug=False,max_registers=64)
def numba_scatter(patches,vid,inds,dilation,kpt,qpb):

    # -- reflective boundary --
    # def bounds(val,lim):
    #     if val < 0: val = (-val-1)
    #     if val >= lim: val = (2*lim - val - 1)
    #     return int(val)
    def bounds(val,lim):
        if val < 0: val = -val
        if val >= lim: val = 2*(lim-1) - val
        return int(val)

    # -- shapes --
    t,c,h,w = vid.shape
    nq,k,pt,c,ps,ps = patches.shape
    psHalf = ps//2

    # -- cuda threads --
    tidX = cuda.threadIdx.x
    tidY = cuda.threadIdx.y
    tidZ = cuda.threadIdx.z
    blkDimX = cuda.blockDim.x
    blkDimY = cuda.blockDim.y

    # -- batching --
    query_start = cuda.blockIdx.x*qpb
    k_start = cuda.threadIdx.x*kpt

    for _qi in range(qpb):

        # -- query index --
        qi = query_start + _qi
        if qi >= nq: continue

        for _ki in range(kpt):

            # -- k index --
            ki = k_start + _ki
            if ki >= k: continue

            # -- fill --
            ti = inds[qi,ki,0]
            hi = inds[qi,ki,1]
            wi = inds[qi,ki,2]

            # -- fill across cuda threads --
            pi = tidY
            pj = tidZ
            vi_h = bounds(hi+dilation*(pi - psHalf),h)
            vi_w = bounds(wi+dilation*(pj - psHalf),w)

            # -- spatially valid --
            valid_hw = vi_h >= 0 and vi_h < h
            valid_hw = valid_hw and (vi_w >= 0 and vi_w < w)

            # -- iterate over loop --
            for pk in range(pt):
                vi_t = bounds(ti + pk,t)

                # -- check valid --
                valid_t = vi_t >= 0 and vi_t < t
                valid = valid_hw and valid_t

                for ci in range(c):
                    if valid: pix = vid[vi_t,ci,vi_h,vi_w]
                    else: pix = 0.
                    patches[qi,ki,pk,ci,pi,pj] = pix


# -- zero padding --
@cuda.jit(debug=False,max_registers=64)
def numba_scatter_zp(patches,vid,inds,dilation,kpt,qpb):

    # -- reflective boundary --
    def bounds(val,lim):
        val = int(val)
        return val

    # -- shapes --
    t,c,h,w = vid.shape
    nq,k,pt,c,ps,ps = patches.shape
    psHalf = ps//2

    # -- cuda threads --
    tidX = cuda.threadIdx.x
    tidY = cuda.threadIdx.y
    tidZ = cuda.threadIdx.z
    blkDimX = cuda.blockDim.x
    blkDimY = cuda.blockDim.y

    # -- batching --
    query_start = cuda.blockIdx.x*qpb
    k_start = cuda.threadIdx.x*kpt

    for _qi in range(qpb):

        # -- query index --
        qi = query_start + _qi
        if qi >= nq: continue

        for _ki in range(kpt):

            # -- k index --
            ki = k_start + _ki
            if ki >= k: continue

            # -- fill --
            _ti = inds[qi,ki,0]
            _hi = inds[qi,ki,1]
            _wi = inds[qi,ki,2]

            # -- fill across cuda threads --
            pi = tidY
            pj = tidZ
            hi = bounds(_hi+dilation*(pi - psHalf),h)
            wi = bounds(_wi+dilation*(pj - psHalf),w)

            # -- valid check for zero padding --
            valid = hi >= 0 and hi < h
            valid = (wi >= 0 and wi < w) and valid
            valid = (ti >= 0 and ti < t) and valid

            # -- iterate over loop --
            for pk in range(pt):
                ti = bounds(_ti + pk,t)
                for ci in range(c):
                    if valid: pix = vid[ti,ci,hi,wi]
                    else: pix = 0.
                    patches[qi,ki,pk,ci,pi,pj] = pix



