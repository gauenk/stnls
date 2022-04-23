
# -- linalg --
import torch
import numpy as np
from einops import rearrange

# -- python-only numba --
import math
from numba import njit,cuda

def run(patches,nlDists,nlInds,vid=None,wvid=None,shape=None):

    # -- misc --
    device = patches.device

    # -- allocate videos --
    if vid is None: vid = allocate_vid(shape,device)
    if wvid is None: wvid = allocate_vid(shape,device)

    # -- inds2threads --
    thInds,thRegs,thCode = allocate_inds2threads(nlInds,nthreads,device)

    # -- exec gather --
    numba_launcher(vid,wvid,patches,nlDists,nlInds,thInds,thRegs,thCode)

    return vid,wvid

def allocate_vid(shape,device):
    vid = th.zeros(shape,device=device,dtype=th.float32)
    return vid

def allocate_inds2threads(nlInds,nthreads,device):
    num_per_thread = 0
    threadInds = th.zeros((nthreads,num_per_thread),dtype=th.int32)
    threadRegions = th.zeros(nthreads,dtype=th.int32)
    threadCode = 123
    return threadInds,threadRegions,threadCode

def numba_launcher(vid,wvid,patches,nlDists,nlInds,thInds,thRegs,thCode):

    # -- numbify all params --
    vid_nba = cuda.as_cuda_array(vid)
    wvid_nba = cuda.as_cuda_array(wvid)
    patches_nba = cuda.as_cuda_array(patches)
    nlDists_nba = cuda.as_cuda_array(nlDists)
    nlInds_nba = cuda.as_cuda_array(nlInds)
    thInds_nba = cuda.as_cuda_array(thInds)
    thRegs_nba = cuda.as_cuda_array(thRegs)
    thCode_nba = cuda.as_cuda_array(thCode)

    # -- exec gather --
    numba_gather(vid_nba,wvid_nba,patches_nba,nlDists_nba,
                 nlInds_nba,thInds_nba,thRegs_nba,thCode_nba)


def numba_gather(vid,wvid,patches,nlDists,nlInds,thInds,thRegs,thCode):
    pass

