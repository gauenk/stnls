
# -- python-only kernel --
import math
from numba import cuda,jit

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

def run(patches,nlDists,nlInds,ps,pt,vid=None,wvid=None,shape=None):

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

#
# TODO fix me up
#

# -- python deps --
import torch
import scipy
import numpy as np
from einops import rearrange

# -- numba --
from numba import njit,cuda

# -- package --
import npc.search_mask as imask
from npc.utils import groups2patches
from npc.utils import Timer


def agg_patches_pweight(patches,images,bufs,args,cs_ptr=None,denom="hw"):
    # -- default stream --
    if cs_ptr is None:
        cs_ptr = torch.cuda.default_stream().cuda_stream

    # -- filter by valid --
    valid = torch.nonzero(torch.all(bufs.inds!=-1,1))[:,0]
    vnoisy = patches.noisy[valid]
    vinds = bufs.inds[valid]
    vvals = bufs.vals[valid]

    # -- iterate over "nkeep" --
    if args.nkeep != -1:
        vinds = bufs.inds[:,:args.nkeep]

    compute_agg_batch_pweight(images.deno,vnoisy,vinds,images.weights,
                             vvals,images.vals,args.ps,args.ps_t,cs_ptr,denom=denom)


def compute_agg_batch_pweight(deno,patches,inds,weights,vals,ivals,
                              ps,ps_t,cs_ptr,denom="hw"):

    # -- numbify the torch tensors --
    deno_nba = cuda.as_cuda_array(deno)
    patches_nba = cuda.as_cuda_array(patches)
    inds_nba = cuda.as_cuda_array(inds)
    weights_nba = cuda.as_cuda_array(weights)
    vals_nba = cuda.as_cuda_array(vals)
    ivals_nba = cuda.as_cuda_array(ivals)
    cs_nba = cuda.external_stream(cs_ptr)

    # -- launch params --
    bsize,num = inds.shape
    c,ph,pw = patches.shape[-3:]
    threads = (c,ph,pw)
    blocks = (bsize,num)

    # -- launch kernel --
    # exec_agg_cuda[blocks,threads,cs_nba](deno_nba,patches_nba,inds_nba,weights_nba,
    #                                      vals_nba_,ivals_nba,ps,ps_t)
    exec_agg_simple_pweight(deno,patches,inds,weights,vals,ivals,ps,ps_t,denom=denom)

def exec_agg_simple_pweight(deno,patches,inds,weights,vals,ivals,ps,ps_t,denom="hw"):

    # -- numbify --
    device = deno.device
    deno_nba = deno.cpu().numpy()
    patches_nba = patches.cpu().numpy()
    inds_nba = inds.cpu().numpy()
    weights_nba = weights.cpu().numpy()
    vals_nba = vals.cpu().numpy()
    ivals_nba = ivals.cpu().numpy()

    # -- exec numba --
    exec_agg_simple_pweight_numba(deno_nba,patches_nba,inds_nba,
                                 weights_nba,vals_nba,ivals_nba,ps,ps_t,
                                 denom=denom)

    # -- back pack --
    deno_nba = torch.FloatTensor(deno_nba).to(device)
    deno[...] = deno_nba
    weights_nba = torch.FloatTensor(weights_nba).to(device)
    weights[...] = weights_nba
    ivals_nba = torch.FloatTensor(ivals_nba).to(device)
    ivals[...] = ivals_nba



@njit
def exec_agg_simple_pweight_numba(deno,patches,inds,weights,vals,
                                  ivals,ps,ps_t,denom="chw"):

    # -- shape --
    nframes,color,height,width = deno.shape
    chw = color*height*width
    hw = height*width
    bsize,npatches = inds.shape # "npatches" _must_ be from "inds"
    Z = chw if denom == "chw" else hw
    npatches_f = 1.*npatches
    psHalf = ps//2
    psHalf2 = psHalf * psHalf

    for bi in range(bsize):
        for ni in range(npatches):
            ind = inds[bi,ni]
            if ind == -1: continue
            t0 = ind // Z
            h0 = (ind % hw) // width
            w0 = ind % width

            # print(t0,h0,w0)
            for pt in range(ps_t):
                for pi in range(ps):
                    for pj in range(ps):
                        t1 = (t0+pt)# % nframes
                        h1 = (h0+pi)# % height
                        w1 = (w0+pj)# % width

                        if t1 < 0 or t1 >= nframes: continue
                        if h1 < 0 or h1 >= height: continue
                        if w1 < 0 or w1 >= width: continue

                        # weight = np.exp(-100.0 * (ni/50.))#npatches_f))
                        # weight = 1.*(ni==0)#np.exp(-2.0 * (ni/50.))#npatches_f))
                        pleft = (pi - psHalf)**2/psHalf2
                        pright = (pj - psHalf)**2/psHalf2
                        pdist = pleft + pright
                        weight = np.exp(-10000.*pdist)
                        for ci in range(color):
                            gval = patches[bi,ni,pt,ci,pi,pj]
                            deno[t1,ci,h1,w1] += weight * gval
                        weights[t1,h1,w1] += weight
                        # if ni > 0:
                        #     ivals[t0+pt,h0+pi,w0+pj] += vals[bi,ni]

