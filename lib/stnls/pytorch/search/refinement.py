# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

# -- package --
import stnls

# -- api --
from stnls.utils import extract_pairs

# -- local --
from .utils import filter_k
from .utils import shape_vids,dist_type_select
from .utils import allocate_pair,allocate_vid
from .utils import get_ctx_qinds
from .shared import manage_self
# from .nls_bwd_impl import nls_backward
from .ref_bwd_impl import ref_backward

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Forward Logic
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refine_forward(vid0, vid1, qinds,
                   ws, ps, k, wr, stride0, stride1, dilation, pt,
                   dist_type, restricted_radius, reflect_bounds, full_ws,
                   topk_mode, self_action, patch_offset, itype_fwd):

    # -- fix negative Q --
    # if Q > 0:
    #     qinds = qinds[:,:,qshift:qshift+Q].contiguous()
    B,HD,T,nH,nW,K,_ = qinds.shape
    Q = T*nH*nW

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    device = qinds.device
    B,HD,Q,K = qinds.shape[:-1]
    base_shape = (B,HD,Q,K,wr,wr)
    dists,inds = allocate_pair(base_shape,device,vid0.dtype,idist_val,itype_fwd)
    imode = 0 if itype_fwd == "int" else 1
    patch_offset = 0 if use_adj else -(ps//2)

    # -- run --
    # print(vid0.shape,qinds.shape)
    if imode == 0:
        fwd_fxn = stnls_cuda.refinement_int_forward
    else:
        fwd_fxn = stnls_cuda.refinement_bilin2d_forward
        stride1 = float(stride1)

    # -- allow for int fwd when actually float --
    # print(qinds.dtype)
    q_dtype = qinds.dtype
    if itype_fwd == "int":
        inds = inds.int()
        if qinds.dtype == th.float:
            qinds = qinds.round().int()
        stride1 = int(max(1,int(stride1)))

    # -- forward --
    fwd_fxn(vid0, vid1, qinds, dists, inds,
            ws, ps, k, stride0, stride1, dilation, pt,
            reflect_bounds, full_ws, restricted_radius,
            patch_offset, dist_type_i)
    # print("dists [max,min]: ",th.max(dists).item(),th.min(dists).item())

    # -- allow for int fwd when actually float --
    if itype_fwd == "int" and q_dtype == th.float:
        qinds = qinds.float()
        inds = inds.float()

    # -- no negative --
    # if th.any(qinds[0]<0):
    #     print(qinds[0])
    #     print(inds[0])

    # -- compress search region --
    dists=dists.view(B,HD,Q,-1)
    inds=inds.view(B,HD,Q,-1,3)
    # print(dists[0,0,0])

    # -- manage self dists --
    H,W = vid0.shape[-2:]
    anchor_self = self_action == "anchor"
    remove_self = self_action == "remove"
    dists,inds = manage_self(dists,inds,anchor_self,
                             remove_self,qshift,stride0,H,W)

    # -- topk --
    qinds = rearrange(qinds,'b hd q k tr -> (b hd q) k tr')
    k = min(qinds.shape[1],k)
    # print(dists[0,0,0],inds[0,0,0],len(dists[0,0,0]),k)
    if topk_mode == "default":
        dists,inds = stnls.nn.topk(dists,inds,k,dim=3,anchor=anchor_self,
                                   descending=descending,unique=True,qinds=qinds)
    elif topk_mode == "time":
        wt = th.unique(inds[0,0,:,0])
        st = 2*wt+1
        assert k % st == 0
        ke = k//st
        dists,inds = stnls.nn.topk_time(dists,inds,ke,wr,dim=3,anchor=anchor_self,
                                        descending=descending,unique=True)
    else:
        raise ValueError(f"Unknown topk_mode [{topk_mode}]")

    # print(dists[0,0,0])
    # print("^"*30)

    return dists,inds

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Pytorch Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class RefineSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, qinds,
                ws, ps, k, wr, kr, nheads=1, stride0=4, stride1=1,
                dilation=1, pt=1, dist_type="l2",
                restricted_radius=False, reflect_bounds=True,
                full_ws=False, topk_mode="default", self_action=None,
                use_adj=False, normalize_bwd=False, k_agg=-1,
                itype_fwd="int", itype_bwd="int"):
        """
        Run the refinement search

        vid0 = [B,T,C,H,W] or [B,HD,T,C,H,W]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """


        # -- reshape with heads --
        dtype = vid0.dtype
        device = vid0.device
        vid0,vid1 = shape_vids(nheads,[vid0,vid1])
        B,HD,T,F,H,W = vid0.shape
        assert qinds.shape[1] == HD
        patch_offset = 0 if use_adj else -(ps//2)

        # -- filter only to kr --
        qinds = filter_k(qinds,kr)
        qinds = qinds.contiguous()

        # -- run fwd pass --
        dists,inds = refine_forward(vid0, vid1, qinds,
                                    ws, ps, k, wr, stride0, stride1, dilation, pt,
                                    dist_type, restricted_radius, reflect_bounds, full_ws,
                                    topk_mode, self_action, patch_offset, itype_fwd)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        qinds = get_ctx_qinds(itype_bwd,qinds)
        ctx.save_for_backward(inds,vid0,vid1,qinds)
        if itype_bwd == "int":
            ctx.mark_non_differentiable(inds)
        ctx.vid_shape = vid0.shape
        ctx_vars = {"stride0":stride0,"stride1":stride1,
                    "ps":ps,"pt":pt,"dil":dilation,
                    "reflect_bounds":reflect_bounds,"k_agg":k_agg,
                    "rbwd":rbwd,"exact":exact,"nbwd":nbwd,"use_atomic":use_atomic,
                    "use_adj":use_adj,"off_H0":off_H0,"off_W0":off_W0,
                    "off_H1":off_H1,"off_W1":off_W1,
                    "normalize_bwd":normalize_bwd,
                    "itype_bwd":itype_bwd,"dist_type_i":dist_type_i,
                    "queries_per_thread":queries_per_thread,
                    "neigh_per_thread":neigh_per_thread,
                    "channel_groups":channel_groups}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds):
        # print("refinement: ",grad_dists.shape,grad_inds_is_none)
        grad0,grad1,grad_qinds = ref_backward(ctx, grad_dists, grad_inds)
        return grad0,grad1,grad_qinds,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None

class RefineSearch(th.nn.Module):

    def __init__(self, ws, ps, k, wr, kr, nheads=1,
                 stride0=4, stride1=1, dilation=1, pt=1, dist_type="l2",
                 restricted_radius=False, reflect_bounds=True,
                 full_ws=False, topk_mode="default", self_action=None,
                 use_adj=False, normalize_bwd=False, k_agg=-1,
                 itype_fwd="int", itype_bwd="int"):
        super().__init__()

        # -- core search params --
        self.ws = ws
        self.ps = ps
        self.k = k
        self.wr = wr
        self.kr = kr
        self.nheads = nheads
        self.stride0 = stride0
        self.stride1 = stride1
        self.dilation = dilation
        self.pt = pt
        self.dist_type = dist_type

        # -- manage patch and search boundaries --
        self.restricted_radius = restricted_radius
        self.reflect_bounds = reflect_bounds
        self.full_ws = full_ws

        # -- special mods to "self" search --
        self.topk_mode = topk_mode
        self.self_action = self_action

        # -- with/without grads --
        self.itype_fwd = itype_fwd
        self.itype_bwd = itype_bwd

        # -- searching offsets --
        self.use_adj = use_adj

        # -- backprop params --
        self.normalize_bwd = normalize_bwd
        self.k_agg = k_agg

    def forward(self,vid0,vid1,qinds):
        return RefineSearchFunction.apply(vid0,vid1,qinds,
                                          self.ws,self.ps,self.k,
                                          self.wr,self.kr,self.nheads,
                                          self.stride0,self.stride1,
                                          self.dilation,self.pt,self.dist_type,
                                          self.restricted_radius,
                                          self.reflect_bounds,self.full_ws,
                                          self.topk_mode,self.self_action,
                                          self.use_adj,self.normalize_bwd,
                                          self.k_agg,self.itype_fwd,self.itype_bwd)

    def flops(self,T,F,H,W):
        return 0

    def radius(self,H,W):
        return self.ws


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Direct API]  stnls.search.refine(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid0, vid1, qinds,
           ws, ps, k, wr, kr=-1, nheads=1, stride0=4, stride1=1,
           dilation=1, pt=1, dist_type="l2",
           restricted_radius=False, reflect_bounds=True, full_ws=False,
           topk_mode="default", self_action=None, use_adj=False,
           normalize_bwd=False, k_agg=-1, itype_fwd="int", itype_bwd="int"):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = RefineSearchFunction.apply
    return fxn(vid0, vid1, qinds,
               ws, ps, k, wr, kr, nheads, stride0, stride1,
               dilation, pt, dist_type, restricted_radius, reflect_bounds,
               full_ws, topk_mode, self_action, use_adj,
               normalize_bwd, k_agg, itype_fwd, itype_bwd)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ws":-1,"wt":-1,"ps":7,"k":10,"wr":1,"kr":-1,
             "nheads":1, "stride0":4, "stride1":1, "dilation":1, "pt":1,
             "dist_type":"l2", "restricted_radius":False,
             "reflect_bounds":True, "full_ws":False,
             "topk_mode": "default", "self_action":None,
             "use_adj":False, "normalize_bwd": False, "k_agg":-1,
             "itype_fwd":"int", "itype_bwd":"int"}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    search = RefineSearch(cfg.ws, cfg.ps, cfg.k, cfg.wr, cfg.kr,
                          nheads=cfg.nheads, stride0=cfg.stride0,
                          stride1=cfg.stride1, dilation=cfg.dilation,
                          pt=cfg.pt, dist_type=cfg.dist_type,
                          restricted_radius=cfg.restricted_radius,
                          reflect_bounds=cfg.reflect_bounds, full_ws=cfg.full_ws,
                          topk_mode=cfg.topk_mode, self_action=cfg.self_action,
                          use_adj=cfg.use_adj, normalize_bwd=cfg.normalize_bwd,
                          k_agg=cfg.k_agg,itype_fwd=cfg.itype_fwd,itype_bwd=cfg.itype_bwd)

    return search

