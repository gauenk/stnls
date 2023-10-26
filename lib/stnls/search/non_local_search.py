
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
from .utils import shape_vids,allocate_pair,dist_type_select,allocate_vid
from .utils import get_ctx_shell,ensure_flow_shape,shape_flows
from .shared import manage_self,reflect_bounds_warning
from .nls_bwd_impl import nls_backward

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Forward Logic
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def nls_forward(vid0, vid1, flows,
                ws, wt, ps, k, stride0, stride1,
                dist_type, dilation, pt,
                topk_mode, self_action,
                reflect_bounds, full_ws, use_adj, itype):

    # -- unpack --
    # itype = "int"
    device = vid0.device
    B,HD,T,C,H,W = vid0.shape
    patch_offset = 0 if use_adj else -(ps//2)
    # print(ps,k,dist_type,topk_mode,self_action,patch_offset)

    # -- derived shapes --
    nH0 = (H-1)//stride0+1
    nW0 = (W-1)//stride0+1
    Q = T*nH0*nW0
    # print(vid0.shape,nH0,nW0,Q)

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    W_t = min(2*wt+1,T)
    base_shape = (B,HD,Q,W_t,ws,ws)
    dists,inds = allocate_pair(base_shape,device,vid0.dtype,idist_val,itype)

    # -- check flows --
    assert flows.shape[3] in [W_t-1,W_t]

    # -- forward --
    if itype == "int":
        flows = flows.round().int()
        inds = inds.int()
        stride1 = max(1,int(stride1))
        fwd_fxn = stnls_cuda.non_local_search_int_forward
    else:
        fwd_fxn = stnls_cuda.non_local_search_bilin2d_forward
        stride1 = float(stride1)
    fwd_fxn(vid0, vid1, flows, dists, inds,
            ps, k, stride0, stride1, dilation, pt,
            reflect_bounds, full_ws, patch_offset,  dist_type_i)

    # -- anchor --
    assert self_action in [None,"anchor","anchor_each","remove","remove_ref_frame"]
    anchor_self = False if self_action is None else "anchor" in self_action
    if self_action == "anchor":
        stnls.nn.anchor_self(dists,inds,stride0,H,W)
    elif self_action == "anchor_each":
        stnls.nn.anchor_self_time(dists,inds,flows,wt,stride0,H,W)
    elif self_action == "remove":
        raise NotImplementedError("Not implemented self_action [remove].")
    elif self_action == "remove_ref_frame":
        assert wt > 0,"Cannot remove ref frame if not searching across time."
        dists = dists[...,1:,:,:].contiguous()
        inds = inds[...,1:,:,:,:].contiguous()
    elif self_action is None:
        pass
    else:
        raise ValueError(f"Uknown option for self_action [{self_action}]")

    # -- topk --
    if topk_mode == "all":
        dim = 3
        dists=dists.view(B,HD,Q,W_t*ws*ws)
        inds=inds.view(B,HD,Q,W_t*ws*ws,3)
        dists,inds = stnls.nn.topk(dists,inds,k,dim=dim,anchor=anchor_self,
                                   descending=descending)
    elif topk_mode == "each":
        dists = rearrange(dists,'... wh ww -> ... (wh ww)')
        inds = rearrange(inds,'... wh ww d2or3 -> ... (wh ww) d2or3')
        dists,inds = stnls.nn.topk_each(dists,inds,k,descending,anchor_self=anchor_self)
    else:
        raise ValueError(f"Unknown topk_mode [{topk_mode}]")

    # -- reshape --
    dists=dists.view(B,HD,T,nH0,nW0,-1)
    inds=inds.view(B,HD,T,nH0,nW0,-1,3)

    return dists,inds

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Pytorch Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class NonLocalSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, flows,
                ws, wt, ps, k, nheads=1,
                stride0=4, stride1=1, dist_type="l2",
                dilation=1, pt=1, topk_mode="all",
                self_action=None, reflect_bounds=True, full_ws=True,
                use_adj=False, normalize_bwd=False, k_agg=-1, itype="float"):

        """
        Run the non-local search

        vid0 = [B,T,C,H,W] or [B,HD,T,C,H,W]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """

        # -- reshape with heads --
        dtype = vid0.dtype
        device = vid0.device
        vid0,vid1 = shape_vids(nheads,[vid0,vid1])
        B,HD,T,F,H,W = vid0.shape
        reflect_bounds_warning(reflect_bounds)

        # -- manage forward shape --
        flow_ndim = flows.ndim
        flows = shape_flows(nheads,flows)
        B,HD,T,W_t,_,fH,fW = flows.shape

        # -- sample flow  --
        nH = (H-1)//stride0+1
        nW = (W-1)//stride0+1
        assert (fH == nH) and (fW == nW)

        # -- run [optionally batched] forward function --
        dists,inds = nls_forward(vid0, vid1, flows,
                                 ws, wt, ps, k, stride0, stride1,
                                 dist_type, dilation, pt,
                                 topk_mode, self_action,
                                 reflect_bounds, full_ws, use_adj, itype)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        flows =  get_ctx_shell(flows,itype=="int")
        dists_ctx = get_ctx_shell(dists,itype=="int")
        ctx.save_for_backward(dists_ctx,inds,vid0,vid1,flows)
        if itype == "int":
            ctx.mark_non_differentiable(inds)
        ctx.vid_shape = vid0.shape
        ctx_vars = {"stride0":stride0,"stride1":stride1,
                    "ps":ps,"pt":pt,"ws":ws,"wt":wt,"dil":dilation,
                    "reflect_bounds":reflect_bounds,
                    "normalize_bwd":normalize_bwd,
                    "k_agg":k_agg,"use_adj":use_adj,
                    "dist_type_i":dist_type_i,"itype":itype,
                    "flow_ndim":flow_ndim}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds):
        # # -- reshape --
        # dists=dists.view(B,HD,T,nH0,nW0,-1)
        # inds=inds.view(B,HD,T,nH0,nW0,-1,3)

        grad0,grad1,gfflow = nls_backward(ctx, grad_dists, grad_inds)

        return grad0,grad1,gfflow,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,None

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Pytorch Module
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class NonLocalSearch(th.nn.Module):

    def __init__(self, ws, wt, ps, k, nheads=1,
                 stride0=4, stride1=1, dist_type="l2",
                 dilation=1, pt=1, self_action=None, topk_mode="all",
                 reflect_bounds=True, full_ws=True, use_adj=False,
                 normalize_bwd=False, k_agg=-1, itype="float"):
        super().__init__()

        # -- core search params --
        self.ws = ws
        self.wt = wt
        self.ps = ps
        self.k = k
        self.nheads = nheads
        self.dist_type = dist_type
        self.stride0 = stride0
        self.stride1 = stride1
        self.dilation = dilation
        self.pt = pt

        # -- forward --
        self.itype = itype

        # -- manage patch and search boundaries --
        self.reflect_bounds = reflect_bounds
        self.full_ws = full_ws

        # -- special mods to "self" search --
        self.topk_mode = topk_mode
        self.self_action = self_action

        # -- searching offsets --
        self.use_adj = use_adj

        # -- backprop params --
        self.normalize_bwd = normalize_bwd
        self.k_agg = k_agg


    def forward(self,*args):
        assert self.ws > 0,"Must have nonzero spatial search window"
        assert self.wt >= 0,"Must have nonnegative time search window"
        vid0,vid1 = args[:2]
        if len(args) == 4:
            fflow,bflow = args[2:]
            flows = stnls.nn.search_flow(fflow,bflow,self.wt,self.stride0)
        elif len(args) == 3:
            flows = args[2]
        # if self.itype == "int": flows = flows.int()
        return NonLocalSearchFunction.apply(vid0,vid1,flows,
                                            self.ws,self.wt,self.ps,self.k,
                                            self.nheads,self.stride0,
                                            self.stride1,self.dist_type,
                                            self.dilation,self.pt,
                                            self.topk_mode,self.self_action,
                                            self.reflect_bounds,self.full_ws,
                                            self.use_adj,self.normalize_bwd,
                                            self.k_agg,self.itype)

    def flops(self,T,F,H,W):
        print("hi.")
        return 0

        # -- unpack --
        ps,pt = self.ps,self.pt

        # -- compute search --
        nrefs_hw = ((H-1)//self.stride0+1) * ((W-1)//self.stride0+1)
        nrefs = T * HD * nrefs_hw
        nsearch = ws * ws * (2*wt+1)
        flops_per_search = 2 * F * ps * ps * pt
        search_flops = nrefs * nsearch * flops_per_search
        flops = search_flops

        # -- compute top-k --
        if self.k > 0:
            sort_flops = nrefs * (nsearch * np.log(nsearch))
            flops += sort_flops

        return flops

    def radius(self,H,W):
        return self.ws

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#            [Functional API]  stnls.search.nls(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid0, vid1, flows,
           ws, wt, ps, k, nheads=1,
           stride0=1, stride1=1, dist_type="l2",
           dilation=1, pt=1, self_action=None,
           topk_mode="all",reflect_bounds=True,
           full_ws=True,use_adj=False,
           normalize_bwd=False, k_agg=-1, itype="float"):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = NonLocalSearchFunction.apply
    return fxn(vid0,vid1,flows,ws,wt,ps,k,
               nheads,stride0,stride1,dist_type,
               dilation,pt,self_action,topk_mode,
               reflect_bounds,full_ws,
               use_adj,normalize_bwd,k_agg,itype)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ws":-1,"wt":-1,"ps":3,"k":10,
             "nheads":1,"dist_type":"l2",
             "stride0":1, "stride1":1, "dilation":1, "pt":1,
             "reflect_bounds":True, "full_ws":True,
             "self_action":None,"use_adj":False,
             "normalize_bwd": False, "k_agg":-1,
             "itype":"float","topk_mode":"all",}
    return extract_pairs(cfg,pairs,restrict=restrict)


def init(cfg):
    cfg = extract_config(cfg)
    search = NonLocalSearch(cfg.ws, cfg.wt, cfg.ps, cfg.k, nheads=cfg.nheads,
                            stride0=cfg.stride0, stride1=cfg.stride1,
                            dist_type=cfg.dist_type, dilation=cfg.dilation, pt=cfg.pt,
                            self_action=cfg.self_action, topk_mode=cfg.topk_mode,
                            reflect_bounds=cfg.reflect_bounds, full_ws=cfg.full_ws,
                            use_adj=cfg.use_adj,normalize_bwd=cfg.normalize_bwd,
                            k_agg=cfg.k_agg,itype=cfg.itype)
    return search

