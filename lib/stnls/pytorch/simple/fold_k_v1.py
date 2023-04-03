"""
The fold_k function that includes the race condition

Used as a benchmark for reference

"""

# -- python-only kernel --
import math
from numba import cuda,jit

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat


def run(patches,nlDists,nlInds,vid=None,wvid=None,shape=None):

    # -- allocate videos --
    if vid is None:
        vid = allocate_vid(shape,patches.device)
    if wvid is None:
        wvid = allocate_vid(shape,patches.device)

    # -- exec fold_k --
    numba_launcher(vid,wvid,patches,nlDists,nlInds)

    return vid,wvid

def allocate_vid(shape,device):
    vid = th.zeros(shape,device=device,dtype=th.float32)
    return vid

def numba_launcher(vid,wvid,patches,nlDists,nlInds):

    # -- numbify all params --
    vid_nba = cuda.as_cuda_array(vid)
    wvid_nba = cuda.as_cuda_array(wvid)
    patches_nba = cuda.as_cuda_array(patches)
    nlDists_nba = cuda.as_cuda_array(nlDists)
    nlInds_nba = cuda.as_cuda_array(nlInds)

    # -- hyper --
    lamb = 1.

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
    numba_fold_k[nblocks,nthreads](vid_nba,wvid_nba,patches_nba,
                                   nlDists_nba,nlInds_nba,lamb,kpt,qpb)


@cuda.jit(debug=False,max_registers=64)
def numba_fold_k(vid,wvid,patches,nlDists,nlInds,lamb,kpt,qpb):

    # -- reflective boundary --
    def bounds(val,lim):
        if val < 0: val = (-val-1)
        if val >= lim: val = (2*lim - val - 1)
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
            ti = nlInds[qi,ki,0]
            hi = nlInds[qi,ki,1]
            wi = nlInds[qi,ki,2]
            pweight = 1.#math.exp(-lamb*nlDists[qi,ki])

            # -- fill across cuda threads --
            pi = tidY
            pj = tidZ
            vi_h = bounds(hi - psHalf + pi,h)
            vi_w = bounds(wi - psHalf + pj,w)
            for pk in range(pt):
                vi_t = bounds(ti + pk,t)
                for ci in range(c):
                    pix = patches[qi,ki,pk,ci,pi,pj]
                    vid[vi_t,ci,vi_h,vi_w] += pix * pweight
                    wvid[vi_t,ci,vi_h,vi_w] += pweight
