"""

Stack Non-Local Patches


vid # [B HD T F H W] or [B T F' H W] with F' = (HD F) and HD = inds.shape[1]
inds.shape # B,HD,Q,K

stack = stnls.non_local_stack(vid,inds,ps=ps,stride0=stride0)

stack # [B HD T F H W]


"""


# -- python --
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

def non_local_stack(vid,weights,flow,ps,stride0,itype="float"):
    # inds = stnls.utils.flow2inds(flow,stride0)
    if vid.ndim == 5:
        HD = flow.shape[1]
        vid = rearrange(vid,'b t (hd f) h w -> b hd t f h w',hd=HD)
    if itype == "int":
        return non_local_stack_int(vid,weights,flow,ps,stride0)
    else:
        return non_local_stack_bilin2d(vid,weights,flow,ps,stride0)

def non_local_stack_int(vid,weights,inds,ps,stride0):
    B,HD,T,F,H,W = vid.shape
    B,HD,T,nH,nW,K,_ = inds.shape
    stack = th.zeros((B,HD,K,T,F,H,W),device=vid.device,dtype=th.float32)
    for bi in range(B):
        for hi in range(HD):
            for nt_i in range(T):
                for nh_i in range(nH):
                    for nw_i in range(nW):
                        ref_t = nt_i
                        ref_h = nh_i*stride0
                        ref_w = nw_i*stride0
                        for ki in range(K):
                            prop_t = ref_t + inds[bi,hi,ti,nh_i,nw_i,ki,0]
                            prop_t = int(prop_t)
                            prop_h = ref_h + inds[bi,hi,ti,nh_i,nw_i,ki,1]
                            prop_w = ref_w + inds[bi,hi,ti,nh_i,nw_i,ki,2]

                            for pi in range(ps):
                                for pj in range(ps):
                                    refH_ij = ref_h - (ps//2) + pi
                                    refW_ij = ref_w - (ps//2) + pj
                                    propH_ij = prop_h - (ps//2) + pi
                                    propW_ij = prop_w - (ps//2) + pj
                                    val = vid[bi,hi,prop_t,:,propH_ij,propW_j]
                                    stack[bi,hi,ki,ref_t,:,refH_ij,refW_ij] += val

    return stack


def non_local_stack_bilin2d(vid,weights,inds,ps,stride0):
    B,HD,T,nH,nW,K,_ = inds.shape
    B,HD,T,F,H,W = vid.shape
    stack = th.zeros((B,HD,K,T,F,H,W),device=vid.device,dtype=th.float32)
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

                            for pi in range(ps):
                                for pj in range(ps):
                                    refH_ij = ref_h - (ps//2) + pi
                                    refW_ij = ref_w - (ps//2) + pj
                                    propH_ij = prop_h - (ps//2) + pi
                                    propW_ij = prop_w - (ps//2) + pj
                                    val = read_bilin2d(vid[bi,hi,prop_t],
                                                       propH_ij,propW_ij,H,W)
                                    stack[bi,hi,ki,ref_t,:,refH_ij,refW_ij] += weight*val

    return stack

def bounds(val,lim):
    if val < 0:
        return -val
    elif val >= lim:
        return 2*(lim-1)-val
    else:
        return val

def read_bilin2d(vid,h,w,H,W):
    pix = th.zeros_like(vid[:,0,0])
    z = 0
    for i in range(2):
        for j in range(2):
            hi = int(h + i)
            wi = int(w + j)
            weight = (1 - abs(h - hi)) * (1 - abs(w - wi))
            print(weight,hi,wi,h,w,abs(h-hi),abs(w-wi))
            hi = bounds(hi,H)
            wi = bounds(wi,W)
            z += weight
            pix += weight * vid[:,hi,wi]
    assert z == 1
    return pix

# class NonLocalStackGt(th.nn.Module):

#     def __init__(self,ps=7,stride0=4,pt=1,reflect_bounds=True,dilation=1,use_adj=False,
#                  off_H0=0,off_W0=0,off_H1=0,off_W1=0):
#         super().__init__()
#         _vars = ["ps","stride0","pt","reflect_bounds","dilation","use_adj",
#                  "off_H0","off_W0","off_H1","off_W1"]
#         self._vars = _vars
#         for var in _vars:
#             setattr(self,var,eval(var))

#     def forward(self, vid, weights, inds):

#         # -- get 6-dim --
#         HD = inds.shape[1]
#         K = inds.shape[-2]
#         # print("vid.shape,inds.shape: ",vid.shape,inds.shape)
#         ndim = vid.ndim
#         vid = ensure_ndim6(vid,HD)
#         B,HD,T,F,H,W = vid.shape
#         vshape = (B,T,F,H,W)
#         # print("inds.shape: ",inds.shape)
#         # print(vid[0,0,0,0,:5,:5])
#         nH = (H-1)//self.stride0+1
#         nW = (W-1)//self.stride0+1
#         inds = get_inds(inds,"int")

#         # -- get non-local patches --
#         stack = []
#         for hi in range(HD):

#             # -- unfold --
#             unfold = UnfoldK(ps=self.ps)
#             # print("vid[:,hi].shape: ",vid[:,hi].shape)
#             inds_h = inds[:,hi].contiguous()
#             # shape_str = 'b (t nh nw) k tr -> b (t nw nh) k tr'
#             # inds_h = rearrange(inds_h,shape_str,nh=nH,nw=nW)
#             patches_h = unfold(vid[:,hi].contiguous(),inds_h)
#             # print("patches_h.shape: ",patches_h.shape)
#             # # print("vid[:,hi].shape,patches_h.shape: ",vid[:,hi].shape,patches_h.shape)
#             # print(patches_h[0,0,0,0,0])
#             # print(patches_h[0,1,0,0,0])
#             # print(patches_h[0,2,0,0,0])

#             # -- stack according to fold --
#             stack_h = []
#             for ki in range(K):

#                 # -- pick weights --
#                 _weights = weights[:,hi,:,ki,None,None,None,None]
#                 # print("patches_h.shape: ",patches_h.shape)
#                 # print("_weights.shape: ",_weights.shape)
#                 wp = _weights * patches_h[:,:,ki]

#                 # -- fold --
#                 # fold = iFoldz(vshape,stride=self.stride0)
#                 # _vid,_zvid = fold(wp[:,:,None].contiguous())

#                 # -- fold v2 --
#                 fold = NlFold(vshape,stride=self.stride0)
#                 _vid = fold(wp)

#                 # if ki == 0:
#                 #     print("ki: ",ki)
#                 #     print(_vid[0,0,0,30:34,30:34])
#                 #     # print(_zvid[0,0,0,30:34,30:34])
#                 # _vid = _vid / _zvid
#                 # print("ki,p.shape,w.shape: ",
#                 #       ki,patches_h[:,:,ki:ki+1].shape,_weights.shape)
#                 stack_h.append(_vid)
#             stack_h = th.stack(stack_h,1)
#             stack.append(stack_h)

#         stack = th.stack(stack,1)
#         # stack = revert_ndim(stack,ndim)
#         # print(stack.shape)

#         return stack
