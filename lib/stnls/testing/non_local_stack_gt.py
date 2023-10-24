"""

Stack Non-Local Patches


vid # [B HD T F H W] or [B T F' H W] with F' = (HD F) and HD = inds.shape[1]
inds.shape # B,HD,Q,K

stack = stnls.non_local_stack(vid,inds,ps=ps,stride0=stride0)

stack # [B HD T F H W]


"""


# -- python --
import math
import torch as th
import stnls
from einops import rearrange

# -- tiling --
# from stnls.pytorch.nn import UnfoldK

# def ensure_ndim6(vid,nheads):
#     if vid.ndim == 5:
#         B,T,HD_F,H,W = vid.shape
#         vid = rearrange(vid,'b t (hd f) h w -> b hd t f h w',hd=nheads)
#     assert vid.ndim == 6
#     return vid

# def revert_ndim(grad_vid,ndim):
#     if ndim == 5:
#         B,T,HD_F,H,W = grad_vid.shape
#         grad_vid = rearrange(grad_vid,'b hd t f h w -> b t (hd f) h w')
#     return grad_vid

# def get_inds(inds,itype):
#     inds = inds.contiguous()
#     if itype == "int" and th.is_floating_point(inds):
#         return inds.round().int()
#     elif itype == "float" and not(th.is_floating_point(inds)):
#         return inds.float()
#     else:
#         return inds

def non_local_stack(vid,weights,flow,ps,stride0,reflect_bounds=True,itype="float"):
    # inds = stnls.utils.flow2inds(flow,stride0)
    if vid.ndim == 5:
        HD = flow.shape[1]
        vid = rearrange(vid,'b t (hd f) h w -> b hd t f h w',hd=HD)
    if itype == "int":
        return non_local_stack_int(vid,weights,flow,ps,stride0,reflect_bounds)
    else:
        return non_local_stack_bilin2d(vid,weights,flow,ps,stride0,reflect_bounds)

def count_cond(bi,hi,ki,ti):
    return (bi == 0) and (hi == 0) and (ki == 0) and (ti == 0)

def illegal_hw(hi,wi,H,W):
    return (hi > (H-1)) or (hi < 0) or (wi > (W-1)) or (wi < 0)

def non_local_stack_int(vid,weights,inds,ps,stride0,reflect_bounds=True):
    B,HD,T,F,H,W = vid.shape
    B,HD,T,nH,nW,K,_ = inds.shape
    stack = th.zeros((B,HD,K,T,F,H,W),device=vid.device,dtype=th.float32)
    counts = th.zeros((H,W))
    for bi in range(B):
        for hi in range(HD):
            for nt_i in range(T):
                for nh_i in range(nH):
                    for nw_i in range(nW):
                        ref_t = nt_i
                        ref_h = nh_i*stride0
                        ref_w = nw_i*stride0
                        for ki in range(K):
                            prop_t = ref_t + inds[bi,hi,nt_i,nh_i,nw_i,ki,0]
                            prop_t = int(prop_t)
                            prop_h = ref_h + inds[bi,hi,nt_i,nh_i,nw_i,ki,1]
                            prop_w = ref_w + inds[bi,hi,nt_i,nh_i,nw_i,ki,2]

                            prop_t = bounds(prop_t,T)
                            prop_h = bounds(prop_h,H)
                            prop_w = bounds(prop_w,W)

                            weight = weights[bi,hi,nt_i,nh_i,nw_i,ki]

                            for pi in range(ps):
                                for pj in range(ps):
                                    refH_ij = ref_h - (ps//2) + pi
                                    refW_ij = ref_w - (ps//2) + pj
                                    propH_ij = prop_h - (ps//2) + pi
                                    propW_ij = prop_w - (ps//2) + pj
                                    if reflect_bounds:
                                        propH_ij = bounds(propH_ij,H)
                                        propW_ij = bounds(propW_ij,W)
                                    if illegal_hw(refH_ij,refW_ij,H,W):
                                        continue
                                    if count_cond(bi,hi,ki,ref_t):
                                        counts[refH_ij,refW_ij] += 1
                                    if illegal_hw(propH_ij,propW_ij,H,W):
                                        continue
                                    val = vid[bi,hi,prop_t,:,propH_ij,propW_ij]
                                    # val = read_bilin2d(vid[bi,hi,prop_t],
                                    #                    propH_ij,propW_ij,H,W)
                                    stack[bi,hi,ki,ref_t,:,refH_ij,refW_ij] += weight*val

    stack = stack/counts.to(vid.device)
    return stack


def non_local_stack_bilin2d(vid,weights,inds,ps,stride0,reflect_bounds=True):
    B,HD,T,nH,nW,K,_ = inds.shape
    B,HD,T,F,H,W = vid.shape
    stack = th.zeros((B,HD,K,T,F,H,W),device=vid.device,dtype=th.float32)
    counts = th.zeros((H,W))
    for bi in range(B):
        for hi in range(HD):
            for nt_i in range(T):
                for nh_i in range(nH):
                    for nw_i in range(nW):
                        ref_t = nt_i
                        ref_h = nh_i*stride0
                        ref_w = nw_i*stride0
                        for ki in range(K):
                            prop_t = ref_t + inds[bi,hi,nt_i,nh_i,nw_i,ki,0].int()
                            prop_h = ref_h + inds[bi,hi,nt_i,nh_i,nw_i,ki,1].item()
                            prop_w = ref_w + inds[bi,hi,nt_i,nh_i,nw_i,ki,2].item()
                            weight = weights[bi,hi,nt_i,nh_i,nw_i,ki]

                            prop_t = bounds(prop_t,T)
                            prop_h = bounds(prop_h,H)
                            prop_w = bounds(prop_w,W)

                            for pi in range(ps):
                                for pj in range(ps):
                                    refH_ij = ref_h - (ps//2) + pi
                                    refW_ij = ref_w - (ps//2) + pj
                                    propH_ij = prop_h - (ps//2) + pi
                                    propW_ij = prop_w - (ps//2) + pj
                                    if reflect_bounds:
                                        propH_ij = bounds(propH_ij,H)
                                        propW_ij = bounds(propW_ij,W)
                                    if illegal_hw(refH_ij,refW_ij,H,W):
                                        continue
                                    if count_cond(bi,hi,ki,ref_t):
                                        counts[refH_ij,refW_ij] += 1
                                    if illegal_hw(propH_ij,propW_ij,H,W):
                                        continue
                                    val = read_bilin2d(vid[bi,hi,prop_t],
                                                       propH_ij,propW_ij,H,W)
                                    stack[bi,hi,ki,ref_t,:,refH_ij,refW_ij] += weight*val

    stack = stack/counts.to(vid.device)
    return stack

def bounds(val,lim):
    if val < 0:
        return -val
    elif val > (lim-1):
        return 2*(lim-1)-val
    else:
        return val

def read_bilin2d(vid,h,w,H,W):
    pix = th.zeros_like(vid[:,0,0])
    z = 0
    for i in range(2):
        for j in range(2):
            hi = math.floor(h + i)
            wi = math.floor(w + j)
            weight = (1 - abs(h - hi)) * (1 - abs(w - wi))
            # print(weight,hi,wi,h,w,abs(h-hi),abs(w-wi))
            z += weight
            hi = bounds(hi,H)
            wi = bounds(wi,W)
            pix += weight * vid[:,hi,wi]
    # print(z)
    assert z == 1
    return pix

