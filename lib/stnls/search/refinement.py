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
from stnls.search.utils import filter_k,shape_vids,dist_type_select
from stnls.search.utils import shape_refinement_flows as shape_flows
from stnls.search.shared import reflect_bounds_warning

# -- implementation --
from stnls.search.impl.refinement import forward,backward

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Pytorch Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class RefineSearchFunction(th.autograd.Function):


    @staticmethod
    def forward(ctx, vid0, vid1, flows,
                ws, wt, wr, k, kr=-1, ps=1, nheads=1,
                stride0=4, stride1=1, strideQ=None,
                dilation=1, pt=1, dist_type="l2",
                restricted_radius=False, reflect_bounds=True,
                full_ws=True, topk_mode="all", self_action=None,
                use_adj=False, normalize_bwd=False, k_agg=-1,
                off_Hq=0, off_Wq=0, itype="float"):
        """
        Run the refinement search

        vid0 = [B,T,C,H,W] or [B,HD,T,C,H,W]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """
        # print("[ref_search]: ",ws,wt,wr,ps,k,kr,nheads,stride0,stride1,
        #       dist_type,itype,topk_mode)

        # -- reshape with heads --
        dtype = vid0.dtype
        device = vid0.device
        vid0,vid1 = shape_vids(nheads,[vid0,vid1])
        B,HD,T,F,qH,qW = vid0.shape
        B,HD,T,F,kH,kW = vid1.shape
        nH,nW = (kH-1)//stride0+1,(kW-1)//stride0+1
        flows = shape_flows(nheads,flows,B,nH,nW)
        flows_shape = flows.shape
        flows_requires_grad = flows.requires_grad
        # print("HD,nheads,flows.shape[1]: ",HD,nheads,flows.shape[1])
        assert flows.shape[1] == HD
        patch_offset = 0 if use_adj else -(ps//2)
        reflect_bounds_warning(reflect_bounds)

        # -- input checking --
        assert isinstance(ps,int) and (ps>0),f"patch size is invalid [{ps}]"

        # -- filter only to kr --
        flows = filter_k(flows,kr)
        flows = flows.contiguous()
        # flows_t = flows.clone()
        # flows_t[...,0] = flows_t[...,0].round()

        # -- run fwd pass --
        dists,inds,kselect,reflect = forward(vid0, vid1, flows,
                                             ws, wr, k, kr, ps,
                                             stride0, stride1, strideQ,
                                             dilation, pt, dist_type, restricted_radius,
                                             reflect_bounds, full_ws, topk_mode,
                                             self_action, patch_offset,
                                             off_Hq, off_Wq, itype)

        # -- reshape --
        dists=dists.view(B,HD,T,nH,nW,-1)
        inds=inds.view(B,HD,T,nH,nW,-1,3)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        ctx.save_for_backward(inds,vid0,vid1,kselect,reflect)
        if itype == "int":
            ctx.mark_non_differentiable(inds)
        ctx.vid_shape = vid0.shape
        ctx_vars = {"stride0":stride0,"stride1":stride1,"strideQ":strideQ,
                    "ps":ps,"pt":pt,"dil":dilation,"ws":ws,"wt":wt,
                    "reflect_bounds":reflect_bounds,"k_agg":k_agg,
                    "use_adj":use_adj,"normalize_bwd":normalize_bwd,
                    "itype_bwd":itype,"dist_type_i":dist_type_i,
                    "off_Hq":off_Hq,"off_Wq":off_Wq,
                    "flows_shape":flows_shape,"flows_requires_grad":flows_requires_grad}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds):
        grad0,grad1,grad_flows = backward(ctx, grad_dists, grad_inds)
        return grad0,grad1,grad_flows,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,None,None

class RefineSearch(th.nn.Module):

    def __init__(self, ws, wt, wr, k, kr, ps, nheads=1,
                 stride0=4, stride1=1, strideQ=None,
                 dilation=1, pt=1, dist_type="l2",
                 restricted_radius=True, reflect_bounds=True,
                 full_ws=True, topk_mode="all", self_action=None,
                 use_adj=False, normalize_bwd=False, k_agg=-1,
                 off_Hq=0, off_Wq=0, itype="float"):
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
        self.strideQ = strideQ
        self.dilation = dilation
        self.pt = pt
        self.dist_type = dist_type

        # -- offsets --
        self.off_Hq = off_Hq
        self.off_Wq = off_Wq

        # -- manage patch and search boundaries --
        self.restricted_radius = restricted_radius
        self.reflect_bounds = reflect_bounds
        self.full_ws = full_ws

        # -- special mods to "self" search --
        self.topk_mode = topk_mode
        self.self_action = self_action

        # -- with/without grads --
        self.itype = itype

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
                                          self.stride0,self.stride1,self.strideQ,
                                          self.dilation,self.pt,self.dist_type,
                                          self.restricted_radius,
                                          self.reflect_bounds,self.full_ws,
                                          self.topk_mode,self.self_action,
                                          self.use_adj,self.normalize_bwd,self.k_agg,
                                          self.off_Hq,self.off_Wq,self.itype)

    def flops(self,T,F,H,W):
        return 0

    def radius(self,H,W):
        return self.ws


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Functional API]  stnls.search.refine(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid0, vid1, flows,
           ws, wt, wr, k, kr=-1, ps=1, nheads=1,
           stride0=4, stride1=1, dilation=1, pt=1, dist_type="l2",
           restricted_radius=False, reflect_bounds=True, full_ws=True,
           topk_mode="all", self_action=None, use_adj=False,
           normalize_bwd=False, k_agg=-1,
           off_Hq=0, off_Wq=0, strideQ=None, itype="float"):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = RefineSearchFunction.apply
    return fxn(vid0, vid1, flows,
               ws, wt, wr, k, kr, ps, nheads,
               stride0, stride1, dilation, pt, dist_type,
               restricted_radius, reflect_bounds, full_ws,
               topk_mode, self_action, use_adj,
               normalize_bwd, k_agg, off_Hq, off_Wq, strideQ, itype)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ws":-1,"wt":-1,"wr":1,"ps":1,"k":10,"kr":-1,
             "nheads":1, "stride0":4, "stride1":1, "dilation":1, "pt":1,
             "dist_type":"l2", "restricted_radius":False,
             "reflect_bounds":True, "full_ws":True,
             "topk_mode": "all", "self_action":None,
             "use_adj":False, "normalize_bwd": False, "k_agg":-1,
             "off_Hq":0,"off_Wq":0,"strideQ":None,"itype":"float"}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    cfg = extract_config(cfg,False)
    search = RefineSearch(cfg.ws, cfg.wt, cfg.wr, cfg.k, kr=cfg.kr, ps=cfg.ps,
                          nheads=cfg.nheads, stride0=cfg.stride0, stride1=cfg.stride1,
                          dilation=cfg.dilation, pt=cfg.pt, dist_type=cfg.dist_type,
                          restricted_radius=cfg.restricted_radius,
                          reflect_bounds=cfg.reflect_bounds, full_ws=cfg.full_ws,
                          topk_mode=cfg.topk_mode, self_action=cfg.self_action,
                          use_adj=cfg.use_adj, normalize_bwd=cfg.normalize_bwd,
                          k_agg=cfg.k_agg,off_Hq=cfg.off_Hq,off_Wq=cfg.off_Wq,
                          strideQ=cfg.strideQ,itype=cfg.itype)
    return search

