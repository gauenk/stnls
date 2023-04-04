
# -- python-only kernel --
from numba import cuda,jit

# # -- [for testing] delete me. --
# from stnls.utils import color

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat


def run_bwd(patches,inds,t,h,w,dilation=1):

    # -- allocate patches --
    vid = allocate_vid(patches,inds,t,h,w)

    # -- exec scatter --
    numba_launcher(vid,patches,inds,dilation)

    return vid

def allocate_vid(patches,inds,t,h,w):
    # -- patches --
    nq,k,pt,c,ph,pw = patches.shape
    pshape = (t,c,h,w)
    patches = th.zeros(pshape,device=inds.device,dtype=th.float32)
    return patches

def numba_launcher(vid,patches,inds,dilation):

    # -- numbify all params --
    vid_nba = vid.cpu().numpy()
    patches_nba = patches.cpu().numpy()
    inds_nba = inds.cpu().numpy()

    # -- exec kernel --
    numba_scatter(vid_nba,patches_nba,inds_nba,dilation)
    # numba_scatter_cycle(vid_nba,patches_nba,inds_nba,dilation)

    # -- copy vid --
    vid_nba = th.from_numpy(vid_nba).to(vid.device)
    vid[...] = vid_nba[...]

# -- reflect padding --
@jit(nopython=True,debug=False)
def numba_scatter(vid,patches,inds,dilation):

    # -- "inline" function --
    def bounds(val,lim):
        if val < 0: val = -val
        if val >= lim: val = 2*(lim-1) - val
        return int(val)

    # -- shapes --
    t,c,h,w = vid.shape
    nq,k,pt,c,ps,ps = patches.shape
    psHalf = ps//2

    # -- over queries and neighbors --
    for qi in range(nq):
        for ki in range(k):

            # -- center index --
            _ti = inds[qi,ki,0]
            _hi = inds[qi,ki,1]
            _wi = inds[qi,ki,2]

            # -- fill across cuda threads --
            for pk in range(pt):
                for pi in range(ps):
                    for pj in range(ps):

                        # -- reflect boundary --
                        ti = bounds(_ti + pk,t)
                        hi = bounds(_hi+dilation*(pi - psHalf),h)
                        wi = bounds(_wi+dilation*(pj - psHalf),w)

                        # -- spatially valid --
                        valid_hw = hi >= 0 and hi < h
                        valid_hw = valid_hw and (wi >= 0 and wi < w)

                        # -- check valid --
                        valid_t = ti >= 0 and ti < t
                        valid = valid_hw and valid_t

                        # -- aggregate from patches --
                        for ci in range(c):
                            if valid: pix = patches[qi,ki,pk,ci,pi,pj]
                            else: pix = 0.
                            vid[ti,ci,hi,wi] += pix



# -- reflect padding --
@jit(nopython=True,debug=True)
def numba_scatter_cycle(vid,patches,inds,dilation):

    # -- "inline" function --
    def bounds(val,lim):
        if val < 0: val = -val
        if val >= lim: val = 2*(lim-1) - val
        return int(val)

    # -- shapes --
    t,c,h,w = vid.shape
    nq,k,pt,c,ps,ps = patches.shape
    psHalf = ps//2
    colors = vid.shape[1]

    # -- simulate cuda threads --
    qpt,cpt = 10,4
    nthread_x = 32
    nthread_y = (colors-1) / cpt + 1
    nblock_x = (nq-1) / (qpt*nthread_x) + 1

    # -- simulate query threads --
    for block_x in range(nblock_x):
        for thread_x in range(nthread_x):
            qi_start = qpt * (thread_x + block_x * nthread_x)

            # -- over queries and neighbors --
            for _qi in range(qpt):
                qi = _qi + qi_start
                if qi >= nq: continue
                for ki in range(k):

                    # -- center index --
                    _ti = inds[qi,ki,0]
                    _hi = inds[qi,ki,1]
                    _wi = inds[qi,ki,2]

                    # -- fill across cuda threads --
                    for pk in range(pt):
                        for pi in range(ps):
                            for pj in range(ps):

                                # -- reflect boundary --
                                ti = bounds(_ti + pk,t)
                                hi = bounds(_hi+dilation*(pi - psHalf),h)
                                wi = bounds(_wi+dilation*(pj - psHalf),w)

                                # -- spatially valid --
                                valid_hw = hi >= 0 and hi < h
                                valid_hw = valid_hw and (wi >= 0 and wi < w)

                                # -- check valid --
                                valid_t = ti >= 0 and ti < t
                                valid = valid_hw and valid_t

                                # -- simulate color threads --
                                for thread_y in range(nthread_y):

                                    # -- color end-points --
                                    c0_start = thread_y * cpt
                                    c0_end = min(c0_start + cpt,colors)
                                    c0_offset = thread_x % colors
                                    c0_dist = c0_end - c0_start

                                    # -- for colors in range --
                                    for _c0 in range(c0_start,c0_end):

                                        # -- aggregate from patches --
                                        c0 = (_c0 + c0_offset) % c0_dist + c0_start
                                        if valid: pix = patches[qi,ki,pk,c0,pi,pj]
                                        else: pix = 0.
                                        vid[ti,c0,hi,wi] += pix

