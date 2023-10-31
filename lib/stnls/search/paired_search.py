
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
from .utils import shape_frames,allocate_pair_2d,dist_type_select,allocate_vid
from .utils import get_ctx_shell,ensure_flow_shape
from .utils import ensure_paired_flow_dim as ensure_flow_dim
from .shared import manage_self,reflect_bounds_warning
from .utils import paired_vids as _paired_vids

# -- implementation --
from stnls.search.impl.paired_search import forward,backward

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Pytorch Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PairedSearchFunction(th.autograd.Function):


    @staticmethod
    def forward(ctx, frame0, frame1, flow,
                ws, ps, k, nheads=1, dist_type="prod",
                stride0=4, stride1=1, dilation=1, pt=1,
                reflect_bounds=True,full_ws=True, self_action=None,
                use_adj=False, normalize_bwd=False, k_agg=-1, itype="float"):

        """
        Run the non-local search

        frame0 = [B,T,C,H,W] or [B,HD,T,C,H,W]
        ws = search Window Spatial (ws)
        """

        # -- reshape with heads --
        dtype = frame0.dtype
        device = frame0.device
        ctx.in_ndim = frame0.ndim
        frame0,frame1 = shape_frames(nheads,[frame0,frame1])
        # print("frame0.shape: ",frame0.shape)
        flow = ensure_flow_dim(flow)
        # flow = ensure_flow_shape(flow)
        B,HD,F,H,W = frame0.shape
        flow = flow.contiguous()
        reflect_bounds_warning(reflect_bounds)

        # -- run [optionally batched] forward function --
        dists,inds = forward(frame0, frame1, flow,
                             ws, ps, k, dist_type,
                             stride0, stride1, dilation, pt,
                             self_action, reflect_bounds, full_ws,
                             use_adj, itype)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        flow = get_ctx_shell(flow,itype=="int")
        ctx.save_for_backward(inds,frame0,frame1,flow)
        if itype == "int":
            ctx.mark_non_differentiable(inds)
        ctx.vid_shape = frame0.shape
        ctx_vars = {"stride0":stride0,"stride1":stride1,
                    "ps":ps,"pt":pt,"ws":ws,"dil":dilation,
                    "reflect_bounds":reflect_bounds,
                    "normalize_bwd":normalize_bwd,
                    "k_agg":k_agg,"use_adj":use_adj,
                    "dist_type_i":dist_type_i,"itype":itype}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds):
        grad0,grad1,gflow = backward(ctx, grad_dists, grad_inds)
        return grad0,grad1,gflow,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,None

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Pytorch Module
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class PairedSearch(th.nn.Module):

    def __init__(self, ws, ps, k, nheads=1,
                 dist_type="l2", stride0=4, stride1=1,
                 dilation=1, pt=1, reflect_bounds=True,
                 full_ws=True, self_action=None, use_adj=False,
                 normalize_bwd=False, k_agg=-1, itype="float"):
        super().__init__()

        # -- core search params --
        self.ws = ws
        self.ps = ps
        self.k = k
        self.nheads = nheads
        self.dist_type = dist_type
        self.stride0 = stride0
        self.stride1 = stride1
        self.dilation = dilation
        self.pt = pt
        self.itype = itype

        # -- manage patch and search boundaries --
        self.reflect_bounds = reflect_bounds
        self.full_ws = full_ws
        self.use_adj = use_adj

        # -- special mods to "self" search --
        self.self_action = self_action

        # -- backprop params --
        self.normalize_bwd = normalize_bwd
        self.k_agg = k_agg


    def paired_vids(self, vid0, vid1, flows, wt, skip_self=False):
        return _paired_vids(self.forward, vid0, vid1, flows, wt, skip_self)

    def forward(self, frame0, frame1, flow):
        assert self.ws > 0,"Must have nonzero spatial search window"
        return PairedSearchFunction.apply(frame0,frame1,flow,
                                          self.ws,self.ps,self.k,
                                          self.nheads,self.dist_type,self.stride0,
                                          self.stride1,self.dilation,self.pt,
                                          self.reflect_bounds,self.full_ws,
                                          self.self_action,self.use_adj,
                                          self.normalize_bwd,self.k_agg,self.itype)

    def flops(self,T,F,H,W):
        print("hi.")
        return 0

        # -- unpack --
        ps,pt = self.ps,self.pt

        # -- compute search --
        nrefs_hw = ((H-1)//self.stride0+1) * ((W-1)//self.stride0+1)
        nrefs = T * HD * nrefs_hw
        nsearch = ws_h * ws_w
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
#            [Functional API]  stnls.search.paired(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(frame0, frame1, flow,
           ws, ps, k, nheads=1, batchsize=-1,
           dist_type="l2", stride0=4, stride1=1,
           dilation=1, pt=1, reflect_bounds=True,
           full_ws=True,self_action=None,
           use_adj=False, normalize_bwd=False, k_agg=-1,
           itype="float"):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = PairedSearchFunction.apply
    return fxn(frame0,frame1,flow,ws,ps,k,
               nheads,batchsize,dist_type,
               stride0,stride1,dilation,pt,reflect_bounds,
               full_ws,self_action,use_adj,normalize_bwd,k_agg,
               itype)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ws":-1,"ps":3,"k":10,
             "nheads":1,"dist_type":"l2",
             "stride0":1, "stride1":1, "dilation":1, "pt":1,
             "reflect_bounds":True, "full_ws":True,
             "self_action":None,"use_adj":False,
             "normalize_bwd": False, "k_agg":-1,
             "itype":"float",}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    cfg = extract_config(cfg,False)
    search = PairedSearch(cfg.ws, cfg.ps, cfg.k, nheads=cfg.nheads,
                          dist_type=cfg.dist_type, stride0=cfg.stride0,
                          stride1=cfg.stride1, dilation=cfg.dilation, pt=cfg.pt,
                          reflect_bounds=cfg.reflect_bounds,
                          full_ws=cfg.full_ws, self_action=cfg.self_action,
                          use_adj=cfg.use_adj,normalize_bwd=cfg.normalize_bwd,
                          k_agg=cfg.k_agg,itype=cfg.itype)
    return search

