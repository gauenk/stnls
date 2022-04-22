
# -- python-only kernel --
from numba import cuda,jit

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat


def run(vid,nlInds,ps,pt):

    # -- allocate patches --
    patches = allocate_patches(vid,nlInds,ps,pt)

    # -- exec scatter --
    numba_launcher(patches,vid,nlInds)

    return patches


def allocate_patches(vid,nlInds,ps,pt):
    # -- device --
    device = nlInds.device

    # -- unpack shapes --
    t,c,h,w = vid.shape
    nq,k,three = nlInds.shape

    # -- patches --
    pshape = (nq,k,pt,c,ps,ps)
    patches = th.zeros(pshape,device=device,dtype=th.float32)
    return patches

def numba_launcher(patches,vid,nlInds):

    # -- numbify all params --
    patches_nba = cuda.as_cuda_array(patches)
    vid_nba = cuda.as_cuda_array(vid)
    nlInds_nba = cuda.as_cuda_array(nlInds)

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
    numba_scatter[nblocks,nthreads](patches_nba,vid_nba,nlInds_nba,kpt,qpb)


@cuda.jit(debug=False,max_registers=64)
def numba_scatter(patches,vid,nlInds,kpt,qpb):

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

            # -- fill across cuda threads --
            pi = tidY
            pj = tidZ
            vi_h = bounds(hi - psHalf + pi,h)
            vi_w = bounds(wi - psHalf + pj,w)
            for pk in range(pt):
                vi_t = bounds(ti + pk,t)
                for ci in range(c):
                    pix = vid[vi_t,ci,vi_h,vi_w]
                    patches[qi,ki,pk,ci,pi,pj] = pix



