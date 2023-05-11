
# -- python-only kernel --
from numba import cuda,jit,prange

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_vid0,
# torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_vid1,
# const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid0,
# const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid1,
# const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_dists,
# const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
# const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
# int qstart, int stride0, int n_h0, int n_w0,
# int h0_off, int w0_off, int h1_off, int w1_off,
# int ps, int pt, int dilation, bool use_adj, bool reflect_bounds,
# int bpt, int npt, int cpt) {

def run_batch(grad_dists,vid0,vid1,inds,qstart,stride0,
              ps,pt,dilation,use_adj,reflect_bounds,dist_type="l2"):
    B = grad_dists.shape[0]
    grad_vid0,grad_vid1 = [],[]
    for b in range(B):
        vid0_b,vid1_b = vid0[b],vid1[b]
        grad_dists_b,inds_b = grad_dists[b],inds[b]
        grad_vid0_b,grad_vid1_b = run(grad_dists_b,vid0_b,vid1_b,inds_b,
                                      qstart,stride0,ps,pt,dilation,
                                      use_adj,reflect_bounds,dist_type)
        grad_vid0.append(grad_vid0_b)
        grad_vid1.append(grad_vid1_b)
    grad_vid0 = th.stack(grad_vid0)
    grad_vid1 = th.stack(grad_vid1)
    return grad_vid0,grad_vid1

def run(grad_dists,vid0,vid1,inds,qstart,stride0,
        ps,pt,dilation,use_adj,reflect_bounds,dist_type="l2"):

    # -- device --
    device = grad_dists.device

    # -- num --
    t,c,h,w = vid0.shape
    nh0 = (h-1)//stride0+1
    nw0 = (w-1)//stride0+1

    # -- allocate vids --
    grad_vid0 = th.zeros_like(vid0)
    grad_vid1 = th.zeros_like(vid0)

    # -- exec scatter --
    numba_launcher(grad_vid0,grad_vid1,grad_dists,vid0,vid1,
                   inds,qstart,stride0,nh0,nw0,
                   ps,pt,dilation,use_adj,reflect_bounds,dist_type)
    assert th.any(th.isnan(grad_vid0)).item() is False
    assert th.any(th.isnan(grad_vid1)).item() is False

    return grad_vid0,grad_vid1

def numba_launcher(grad_vid0,grad_vid1,grad_dists,vid0,vid1,
                   inds,qstart,stride0,nh0,nw0,
                   ps,pt,dilation,use_adj,reflect_bounds,dist_type):

    # -- numbify all params --
    device = grad_vid0.device
    grad_vid0_nba = grad_vid0.cpu().numpy()
    grad_vid1_nba = grad_vid1.cpu().numpy()
    grad_dists = grad_dists.cpu().numpy()
    vid0 = vid0.cpu().numpy()
    vid1 = vid1.cpu().numpy()
    inds = inds.cpu().numpy()

    # -- check --
    assert dist_type in ["prod","l2"],"Must be either [prod] or [l2]."

    # -- exec kernel --
    numba_search_bwd(grad_vid0_nba,grad_vid1_nba,grad_dists,
                     vid0,vid1,inds,ps,pt,stride0,dilation,
                     use_adj,reflect_bounds,dist_type)

    # -- copy vids --
    grad_vid0_nba = th.from_numpy(grad_vid0_nba).to(device)
    grad_vid0[...] = grad_vid0_nba[...]
    grad_vid1_nba = th.from_numpy(grad_vid1_nba).to(device)
    grad_vid1[...] = grad_vid1_nba[...]

# -- reflect padding --
@jit(nopython=True,debug=False)
def numba_search_bwd(grad_vid0,grad_vid1,grad_dists,vid0,vid1,inds,
                     ps,pt,stride0,dilation,use_adj,reflect_bounds,
                     dist_type):

    # -- "inline" function --
    def bounds(val,lim):
        vval = val
        if val < 0: vval = -val
        if val >= lim: vval = 2*(lim-1) - val
        return vval#int(vval)

    # -- shapes --
    t,c,h,w = vid0.shape
    nh = (h-1)//stride0+1
    nw = (w-1)//stride0+1
    psHalf = ps//2
    rbounds = reflect_bounds
    nqueries,nneighs = inds.shape[:2]

    # -- independent channels --
    for c0 in prange(c):
        # -- over queries and neighbors --
        for qi in range(nqueries):
            # -- [refecence] center index --
            _ti = qi//(nh*nw)
            q_mod = qi % (nh*nw)
            _hi = stride0*(q_mod // nw)
            _wi = stride0*(q_mod % nw)
    
            for ki in range(nneighs):
    
                # -- access weight --
                weight = grad_dists[qi,ki]
    
                # -- [search] center index --
                _tj = inds[qi,ki,0]
                _hj = inds[qi,ki,1]
                _wj = inds[qi,ki,2]
    
                # -- fill across cuda threads --
                for pk in range(pt):
                    for pi in range(ps):
                        for pj in range(ps):
    
                            # -- [reference] --
                            ti = bounds(_ti + pk,t)
                            hi = _hi+dilation*(pi - psHalf)
                            wi = _wi+dilation*(pj - psHalf)
                            hi = bounds(hi,h) if rbounds else hi
                            wi = bounds(wi,w) if rbounds else wi
    
                            # -- [search] --
                            tj = bounds(_tj + pk,t)
                            hj = _hj+dilation*(pi - psHalf)
                            wj = _wj+dilation*(pj - psHalf)
                            hj = bounds(hj,h) if rbounds else hj
                            wj = bounds(wj,w) if rbounds else wj
    
                            # -- check valid --
                            valid_tj = (tj >= 0) and (tj < t)
                            valid_hj = (hj >= 0) and (hj < h)
                            valid_wj = (wj >= 0) and (wj < w)
                            valid_j = valid_tj and valid_hj and valid_wj
    
                            valid_ti = (ti >= 0) and (ti < t)
                            valid_hi = (hi >= 0) and (hi < h)
                            valid_wi = (wi >= 0) and (wi < w)
                            valid_i = valid_ti and valid_hi and valid_wi
    
                            # -- aggregate from patches --
                            if dist_type == "l2":
                                pix0 = vid0[ti,c0,hi,wi] if valid_i else 0.
                                pix1 = vid1[tj,c0,hj,wj] if valid_j else 0.
                                pix = 2 * weight * (pix0 - pix1)
                                if valid_i:
                                    grad_vid0[ti,c0,hi,wi] += pix
                                if valid_j:
                                    grad_vid1[tj,c0,hj,wj] -= pix
                            elif dist_type == "prod":
                                # pix0 = weight*vid0[ref[0]][c0][ref[1]][ref[2]];
                                # pix1 = weight*vid1[prop[0]][c0][prop[1]][prop[2]];
		                #   grad_vid0[ref[0]][c0][ref[1]][ref[2]] += pix1;
		                #   grad_vid1[prop[0]][c0][prop[1]][prop[2]] += pix0;
                                if (valid_i and valid_j):
                                    pix0 = weight*vid0[ti,c0,hi,wi]
                                    pix1 = weight*vid1[tj,c0,hj,wj]
                                    grad_vid0[ti,c0,hi,wi] += pix1
                                    grad_vid1[tj,c0,hj,wj] += pix0

