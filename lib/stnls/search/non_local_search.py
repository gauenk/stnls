
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
from stnls.search.utils import shape_vids,dist_type_select
from stnls.search.utils import get_ctx_shell,shape_flows
from stnls.search.shared import reflect_bounds_warning

# -- implementation --
from stnls.search.impl.non_local_search import forward,backward

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Pytorch Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class NonLocalSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, flows,
                ws, wt, ps, k, nheads=1,
                stride0=1, stride1=1, strideQ=None,
                dist_type="l2", dilation=1, pt=1, topk_mode="all",
                self_action=None, ws_interior=0,
                reflect_bounds=True, full_ws=True,
                use_adj=False, normalize_bwd=False, k_agg=-1,
                off_Hq=0, off_Wq=0, itype="float"):

        """
        Run the non-local search

        vid0 = [B,T,C,H,W] or [B,HD,T,C,H,W]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """
        # print("[nls_search]: ",ws,wt,ps,k,nheads,stride0,stride1,
        #       dist_type,itype,topk_mode)

        # -- reshape with heads --
        dtype = vid0.dtype
        device = vid0.device
        in_dim = vid0.ndim
        vid0,vid1 = shape_vids(nheads,[vid0,vid1])
        B,HD,T,F,qH,qW = vid0.shape
        kH,kW = vid1.shape[-2:]
        reflect_bounds_warning(reflect_bounds)
        W_t = 2*wt+1
        assert T >= W_t,f"Num Frames [{T}] must be >= Temporal Window [{W_t}]"

        # -- manage forward shape --
        flow_ndim = flows.ndim
        # print(nheads)
        flows = shape_flows(nheads,flows)
        # print("flows.shape: ",flows.shape)
        # exit()
        B,HD,T,W_t,_,fH,fW = flows.shape

        # -- sample flow  --
        nH = (kH-1)//stride0+1
        nW = (kW-1)//stride0+1
        # print(kH,kW,nH,nW,fH,fW)
        assert (fH == nH) and (fW == nW)

        # -- run [optionally batched] forward function --
        dists,inds = forward(vid0, vid1, flows, ws, wt, ps, k,
                             stride0, stride1, strideQ, dist_type,
                             dilation, pt, topk_mode, self_action,
                             ws_interior, reflect_bounds, full_ws,
                             use_adj, off_Hq, off_Wq, itype)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        flows =  get_ctx_shell(flows,itype=="int")
        dists_ctx = get_ctx_shell(dists,itype=="int")
        ctx.save_for_backward(dists_ctx,inds,vid0,vid1,flows)
        if itype == "int":
            ctx.mark_non_differentiable(inds)
        ctx.vid_shape = vid0.shape
        ctx_vars = {"stride0":stride0,"stride1":stride1,"strideQ":strideQ,
                    "ps":ps,"pt":pt,"ws":ws,"wt":wt,"dil":dilation,
                    "reflect_bounds":reflect_bounds,
                    "normalize_bwd":normalize_bwd,
                    "k_agg":k_agg,"use_adj":use_adj,
                    "dist_type_i":dist_type_i,"itype":itype,
                    "off_Hq":off_Hq,"off_Wq":off_Wq,"flow_ndim":flow_ndim,"in_dim":in_dim}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        # dists.shape = (B,HD,T,nH,nW,K)
        # inds.shape = (B,HD,T,nH,nW,K,3)
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds):
        grad0,grad1,gfflow = backward(ctx, grad_dists, grad_inds)
        return grad0,grad1,gfflow,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Pytorch Module
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class NonLocalSearch(th.nn.Module):

    def __init__(self, ws, wt, ps=1, k=-1, nheads=1,
                 stride0=1, stride1=1, dist_type="l2",
                 dilation=1, pt=1, self_action=None, topk_mode="all",
                 ws_interior=0, reflect_bounds=True, full_ws=True,
                 use_adj=False, normalize_bwd=False, k_agg=-1,
                 off_Hq=0, off_Wq=0, strideQ=None, itype="float"):
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
        self.strideQ = strideQ
        self.dilation = dilation
        self.ws_interior = ws_interior
        self.pt = pt

        # -- shifting --
        self.off_Hq = off_Hq
        self.off_Wq = off_Wq

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
        elif len(args) == 2:
            W_t = 2*self.wt+1
            vshape = shape_vids(self.nheads,[args[0]])[0].shape
            B,HD,T,F,qH,qW = vshape
            # print("vshape: ",vshape)
            flows = th.zeros((B,HD,T,W_t,2,qH,qW),device=args[0].device)
        # if self.itype == "int": flows = flows.int()
        return NonLocalSearchFunction.apply(vid0,vid1,flows,
                                            self.ws,self.wt,self.ps,self.k,
                                            self.nheads,self.stride0,
                                            self.stride1,self.strideQ,
                                            self.dist_type,self.dilation,self.pt,
                                            self.topk_mode,self.self_action,
                                            self.ws_interior,self.reflect_bounds,
                                            self.full_ws,self.use_adj,
                                            self.normalize_bwd,self.k_agg,
                                            self.off_Hq,self.off_Wq,self.itype)

    def flops(self,T,F,H,W):
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
           ws, wt, ps=1, k=-1, nheads=1,
           stride0=1, stride1=1, dist_type="l2",
           dilation=1, pt=1, self_action=None,
           topk_mode="all",ws_interior=0,
           reflect_bounds=True,
           full_ws=True,use_adj=False,
           normalize_bwd=False, k_agg=-1,
           off_Hq=0, off_Wq=0, strideQ=None, itype="float"):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = NonLocalSearchFunction.apply
    return fxn(vid0,vid1,flows,ws,wt,ps,k,
               nheads,stride0,stride1,dist_type,
               dilation,pt,self_action,topk_mode,
               ws_interior,reflect_bounds,full_ws,
               use_adj,normalize_bwd,k_agg,
               off_Hq,off_Wq,strideQ,itype)



# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ws":-1,"wt":-1,"ps":1,"k":-1,
             "nheads":1,"dist_type":"l2",
             "stride0":1, "stride1":1, "dilation":1, "pt":1,
             "ws_interior":0,"reflect_bounds":True, "full_ws":True,
             "self_action":None,"use_adj":False,
             "normalize_bwd": False, "k_agg":-1,"topk_mode":"all",
             "off_Hq":0,"off_Wq":0,"strideQ":None,"itype":"float",}
    return extract_pairs(cfg,pairs,restrict=restrict)


def init(cfg):
    cfg = extract_config(cfg,False)
    search = NonLocalSearch(cfg.ws, cfg.wt, cfg.ps, cfg.k, nheads=cfg.nheads,
                            stride0=cfg.stride0, stride1=cfg.stride1,
                            dist_type=cfg.dist_type, dilation=cfg.dilation, pt=cfg.pt,
                            self_action=cfg.self_action, topk_mode=cfg.topk_mode,
                            ws_interior=cfg.ws_interior,
                            reflect_bounds=cfg.reflect_bounds, full_ws=cfg.full_ws,
                            use_adj=cfg.use_adj,normalize_bwd=cfg.normalize_bwd,
                            k_agg=cfg.k_agg,off_Hq=cfg.off_Hq,off_Wq=cfg.off_Wq,
                            strideQ=cfg.strideQ,itype=cfg.itype)
    return search

