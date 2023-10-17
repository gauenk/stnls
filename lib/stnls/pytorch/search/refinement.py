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

def refine_forward(vid0, vid1, flows,
                   ws, wr, k, kr, ps, stride0, stride1, dilation, pt,
                   dist_type, restricted_radius,
                   reflect_bounds, full_ws,
                   topk_mode, self_action, patch_offset, itype_fwd):

    # -- fix negative Q --
    # if Q > 0:
    #     flows = flows[:,:,qshift:qshift+Q].contiguous()
    B,HD,T,nH,nW,Ks,_ = flows.shape
    Q = T*nH*nW

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    device = flows.device
    B,HD,T,nH,nW,Ks = flows.shape[:-1]
    base_shape = (B,HD,T,nH,nW,Ks,wr,wr)
    # print(base_shape,flows.shape)
    dists,inds = allocate_pair(base_shape,device,vid0.dtype,idist_val,itype_fwd)


    # -- allow for int fwd when actually float --
    if itype_fwd == "int":
        inds = inds.int()
        if flows.dtype == th.float:
            flows = flows.round().int()
        kselect = th.zeros(0,device=flows.device)
    else:
        kselect = th.ones_like(dists).int()

    # -- run --
    if itype_fwd == "int":
        stride1 = int(max(1,int(stride1)))
        fwd_fxn = stnls_cuda.refinement_int_forward
        fwd_fxn(vid0, vid1, flows, dists, inds,
                ws, ps, k, stride0, stride1, dilation, pt,
                reflect_bounds, full_ws, restricted_radius,
                patch_offset, dist_type_i)
    else:
        stride1 = float(stride1)
        fwd_fxn = stnls_cuda.refinement_bilin2d_forward
        fwd_fxn(vid0, vid1, flows, dists, inds, kselect,
                ws, ps, k, stride0, stride1, dilation, pt,
                reflect_bounds, full_ws, restricted_radius,
                patch_offset, dist_type_i)

    # print(inds[0,0,0,0,0,:2])
    # -- no negative --
    # if th.any(flows[0]<0):
    #     print(flows[0])
    #     print(inds[0])

    # -- manage self dists --
    # # H,W = vid0.shape[-2:]
    # anchor_self = self_action == "anchor"
    # remove_self = self_action == "remove"
    # assert anchor_self is False
    # assert remove_self is False
    # return_order = not(kselect is None)
    # dists,inds,kselect = manage_self_ksel(dists,inds,kselect,self_action,wr)
    # # kselect = kselect[...,1:] if remove_self else kselect
    # # dists.shape = (B,H,Q,Ks,wr*wr)
    if not(self_action is None) and "anchor" in self_action:
        H,W = vid0.shape[-2:]
        stnls.nn.anchor_self_refine(dists,inds,flows,stride0,H,W)
    else:
        assert self_action == None

    # -- topk --
    assert self_action in [None,"anchor","anchor_each"]
    anchor_self = False if self_action is None else "anchor" in self_action
    if topk_mode == "all":
        dim = 3
        dists=dists.view(B,HD,Q,Ks*wr*wr)
        inds=inds.view(B,HD,Q,Ks*wr*wr,3)
        kselect = kselect.view(B,HD,Q,Ks*wr*wr) if not(kselect is None) else kselect
        dists,inds,order = stnls.nn.topk(dists,inds,k,dim=dim,anchor=anchor_self,
                                         descending=descending,unique=False,
                                         return_order=True)
        if not(kselect is None) and not(order is None):
            kselect = stnls.nn.topk_f.apply_topk(kselect,order,dim)
    elif topk_mode == "each":
        dists = rearrange(dists,'... wh ww -> ... (wh ww)')
        inds = rearrange(inds,'... wh ww d2or3 -> ... (wh ww) d2or3')
        kselect = rearrange(kselect,'... wh ww -> ... (wh ww)')
        dists,inds = stnls.nn.topk_each(dists,inds,k,descending,anchor_self=anchor_self)
        kselect = kselect[...,:k] # all same across dim
    else:
        raise ValueError(f"Unknown topk_mode [{topk_mode}]")


    # -- reshape for output --
    dists=dists.view(B,HD,T,nH,nW,-1)
    inds=inds.view(B,HD,T,nH,nW,-1,3)
    kselect = kselect.view(B,HD,T,nH,nW,-1) if not(kselect is None) else kselect

    return dists,inds,kselect

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Pytorch Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class RefineSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, flows,
                ws, wt, wr, k, kr=-1, ps=1, nheads=1,
                stride0=4, stride1=1, dilation=1, pt=1, dist_type="l2",
                restricted_radius=False, reflect_bounds=True,
                full_ws=False, topk_mode="all", self_action=None,
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
        flows_shape = flows.shape
        flows_requires_grad = flows.requires_grad
        assert flows.shape[1] == HD
        patch_offset = 0 if use_adj else -(ps//2)

        # -- filter only to kr --
        flows = filter_k(flows,kr)
        flows = flows.contiguous()
        # flows_t = flows.clone()
        # flows_t[...,0] = flows_t[...,0].round()

        # -- run fwd pass --
        dists,inds,kselect = refine_forward(vid0, vid1, flows,
                                            ws, wr, k, kr, ps, stride0, stride1,
                                            dilation, pt, dist_type, restricted_radius,
                                            reflect_bounds, full_ws, topk_mode,
                                            self_action, patch_offset, itype_fwd)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        ctx.save_for_backward(inds,vid0,vid1,kselect)
        if itype_bwd == "int":
            ctx.mark_non_differentiable(inds)
        ctx.vid_shape = vid0.shape
        ctx_vars = {"stride0":stride0,"stride1":stride1,
                    "ps":ps,"pt":pt,"dil":dilation,"ws":ws,"wt":wt,
                    "reflect_bounds":reflect_bounds,"k_agg":k_agg,
                    "use_adj":use_adj,"normalize_bwd":normalize_bwd,
                    "itype_bwd":itype_bwd,"dist_type_i":dist_type_i,
                    "flows_shape":flows_shape,"flows_requires_grad":flows_requires_grad}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds):
        grad0,grad1,grad_flows = ref_backward(ctx, grad_dists, grad_inds)
        return grad0,grad1,grad_flows,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None

class RefineSearch(th.nn.Module):

    def __init__(self, ws, wt, wr, k, kr, ps, nheads=1,
                 stride0=4, stride1=1, dilation=1, pt=1, dist_type="l2",
                 restricted_radius=False, reflect_bounds=True,
                 full_ws=False, topk_mode="all", self_action=None,
                 use_adj=False, normalize_bwd=False, k_agg=-1,
                 itype_fwd="int", itype_bwd="int"):
        super().__init__()

        # -- core search params --
        self.ws = ws
        self.wt = wt
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

    def forward(self,vid0,vid1,flows):
        # print(flows.requires_grad)
        return RefineSearchFunction.apply(vid0,vid1,flows,
                                          self.ws, self.wt, self.wr, self.k,
                                          self.kr, self.ps, self.nheads,
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

def _apply(vid0, vid1, flows,
           ws, wr, k, kr=-1, ps=1, stride0=4, stride1=1, dilation=1, pt=1,
           # ws, ps, k, wr, kr=-1, nheads=1, stride0=4, stride1=1,
           # dilation=1, pt=1,
           dist_type="l2", restricted_radius=False, reflect_bounds=True, full_ws=False,
           topk_mode="all", self_action=None, use_adj=False,
           normalize_bwd=False, k_agg=-1, itype_fwd="int", itype_bwd="int"):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = RefineSearchFunction.apply
    return fxn(vid0, vid1, flows,
               ws, wr, k, kr, ps, stride0, stride1, dilation, pt,
               # ws, ps, k, wr, kr, nheads, stride0, stride1, dilation, pt,
               dist_type, restricted_radius, reflect_bounds,
               full_ws, topk_mode, self_action, use_adj,
               normalize_bwd, k_agg, itype_fwd, itype_bwd)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ws":-1,"wt":-1,"ps":1,"k":10,"wr":1,"kr":-1,
             "nheads":1, "stride0":4, "stride1":1, "dilation":1, "pt":1,
             "dist_type":"l2", "restricted_radius":False,
             "reflect_bounds":True, "full_ws":False,
             "topk_mode": "all", "self_action":None,
             "use_adj":False, "normalize_bwd": False, "k_agg":-1,
             "itype_fwd":"int", "itype_bwd":"int"}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    search = RefineSearch(cfg.ws, cfg.wt, cfg.wr, cfg.k, kr=cfg.kr, ps=cfg.ps,
                          stride0=cfg.stride0, stride1=cfg.stride0,
                          dilation=cfg.dilation, pt=cfg.pt, dist_type=cfg.dist_type,
                          restricted_radius=cfg.restricted_radius,
                          reflect_bounds=cfg.reflect_bounds, full_ws=cfg.full_ws,
                          topk_mode=cfg.topk_mode, self_action=cfg.self_action,
                          use_adj=cfg.use_adj, normalize_bwd=cfg.normalize_bwd,
                          k_agg=cfg.k_agg,itype_fwd=cfg.itype_fwd,itype_bwd=cfg.itype_bwd)

    return search

