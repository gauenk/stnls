
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
from .utils import get_ctx_flows,ensure_flow_shape
from .shared import manage_self
from .nls_bwd_impl import nls_backward
from .batching_utils import run_batched,batching_info

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Forward Logic
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def nls_forward(batchsize,*args):
    vid_idx = 0
    ws_idx,wt_idx = 4,5
    stride0_idx = 9
    # dists,inds = nls_forward(batchsize, vid0, vid1, fflow, bflow,
    #                          ws, wt, ps, k, dist_type,
    #                          stride0, stride1, dilation, pt,
    #                          anchor_self, remove_self, reflect_bounds,
    #                          full_ws, use_adj, off_H0, off_W0, off_H1, off_W1)
    # print("vid.shape,stride0,ws,wt: ",
    #       args[vid_idx].shape,args[stride0_idx],args[ws_idx],args[wt_idx])
    ntotal,nbatches,batchsize = batching_info(args[vid_idx],args[stride0_idx],
                                              args[ws_idx],args[wt_idx],
                                              batchsize)
    # print(nbatches,batchsize)
    if nbatches == 1: # shortcut
        qshift,nqueries = 0,-1
        return nls_fwd_main(qshift,nqueries,*args)
    else:
        return run_batched(nls_fwd_main,batchsize,vid_idx,
                           stride0_idx,ws_idx,wt_idx,*args)

def nls_fwd_main(qshift, Q, vid0, vid1, fflow, bflow,
                 ws, wt, ps, k, dist_type,
                 stride0, stride1, dilation, pt,
                 topk_mode, anchor_self, remove_self, reflect_bounds,
                 full_ws, full_ws_time, use_adj,
                 fwd_version, itype, off_H0, off_W0, off_H1, off_W1):

    # -- unpack --
    # itype = "int"
    device = vid0.device
    B,HD,T,C,H,W = vid0.shape

    # -- derived shapes --
    nH0 = (H-1)//stride0+1
    nW0 = (W-1)//stride0+1
    Q = T*nH0*nW0 if Q <= 0 else Q

    # -- search space --
    ws_h,ws_w = ws,ws
    search_abs = False#ws == -1
    if search_abs:
        ws_h,ws_w = nH0,nW0

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    st = min(2*wt+1,T)
    base_shape = (B,HD,Q,st,ws_h,ws_w)
    dists,inds = allocate_pair(base_shape,device,vid0.dtype,idist_val,itype)

    # -- forward --
    fwd_version = "v1"
    if fwd_version == "v1":
        if itype == "int":
            fwd_fxn = stnls_cuda.non_local_search_forward
            stride1 = max(1,int(stride1))
        else:
            fwd_fxn = stnls_cuda.non_local_search_bilin2d_forward
            stride1 = float(stride1)
    elif fwd_version == "v2":
        dists[...] = 0.
        fwd_fxn = stnls_cuda.non_local_search_forward_v2
    else:
        raise ValueError(f"Uknown version [{version}]")
    fflow = fflow.transpose(1,2).contiguous() # b,t,st
    bflow = bflow.transpose(1,2).contiguous()
    # print(fflow.shape)
    # print("vid0.shape: ",vid0.shape)
    fwd_fxn(vid0, vid1, fflow, bflow, dists, inds,
            wt, ps, k, dist_type_i, stride0,
            stride1, dilation, pt, qshift,
            reflect_bounds, full_ws, full_ws_time,
            search_abs, use_adj, off_H0, off_W0, off_H1, off_W1)

    # -- compress search region --
    dists=dists.view(B,HD,Q,-1)
    inds=inds.view(B,HD,Q,-1,3)

    # -- fill nan --
    fill_val = -np.inf if dist_type == "prod" else np.inf
    dists = th.nan_to_num(dists,fill_val)

    # -- manage self dists --
    dists,inds = manage_self(dists,inds,anchor_self,
                             remove_self,qshift,stride0,H,W)

    # -- topk --
    if topk_mode == "default":
        dists,inds = stnls.nn.topk(dists,inds,k,dim=3,anchor=anchor_self,
                                   descending=descending,unique=False)
    elif topk_mode == "time":
        st = 2*wt+1
        assert k % st == 0
        ke = k//st
        dists,inds = stnls.nn.topk_time(dists,inds,ke,ws,dim=3,anchor=anchor_self,
                                        descending=descending,unique=False)
    else:
        raise ValueError(f"Unknown topk_mode [{topk_mode}]")


    return dists,inds

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Pytorch Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class NonLocalSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, fflow, bflow,
                ws, wt, ps, k, nheads=1, batchsize=-1,
                dist_type="prod", stride0=4, stride1=1,
                dilation=1, pt=1, reflect_bounds=True,
                full_ws=True, full_ws_time=True,
                topk_mode="default",anchor_self=False, remove_self=False,
                use_adj=False, off_H0=0, off_W0=0, off_H1=0, off_W1=0,
                normalize_bwd=False, k_agg=-1,
                fwd_version="v1", itype_fwd="int", itype_bwd="int",
                rbwd=True, nbwd=1, exact=False, use_atomic=True,
                queries_per_thread=2, neigh_per_thread=2, channel_groups=-1):

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

        # -- manage forward shape --
        flow_ndim = fflow.ndim
        fflow = ensure_flow_shape(fflow)
        bflow = ensure_flow_shape(bflow)

        # -- run [optionally batched] forward function --
        dists,inds = nls_forward(batchsize, vid0, vid1, fflow, bflow,
                                 ws, wt, ps, k, dist_type,
                                 stride0, stride1, dilation, pt,
                                 topk_mode, anchor_self, remove_self,
                                 reflect_bounds, full_ws, full_ws_time,
                                 use_adj, fwd_version, itype_fwd,
                                 off_H0, off_W0, off_H1, off_W1)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        fflow,bflow = get_ctx_flows(itype_bwd,fflow,bflow)
        ctx.save_for_backward(inds,vid0,vid1,fflow,bflow)
        if itype_bwd == "int":
            ctx.mark_non_differentiable(inds)
        ctx.vid_shape = vid0.shape
        ctx_vars = {"batchsize":batchsize,"stride0":stride0,"stride1":stride1,
                    "ps":ps,"pt":pt,"ws":ws,"wt":wt,"dil":dilation,
                    "reflect_bounds":reflect_bounds,
                    "fwd_version":fwd_version,"normalize_bwd":normalize_bwd,
                    "k_agg":k_agg,"rbwd":rbwd,"exact":exact,"nbwd":nbwd,
                    "use_adj":use_adj,"off_H0":off_H0,"off_W0":off_W0,
                    "off_H1":off_H1,"off_W1":off_W1,
                    "dist_type_i":dist_type_i,"itype_bwd":itype_bwd,
                    "flow_ndim":flow_ndim}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds):
        grad0,grad1,gfflow,gbflow = nls_backward(ctx, grad_dists, grad_inds)
        return grad0,grad1,gfflow,gbflow,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None,None

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Pytorch Module
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class NonLocalSearch(th.nn.Module):

    def __init__(self, ws, wt, ps, k, nheads=1,
                 dist_type="prod", stride0=4, stride1=1,
                 dilation=1, pt=1, reflect_bounds=True,
                 topk_mode="default", full_ws=True, full_ws_time=True,
                 anchor_self=True, remove_self=False,
                 use_adj=False,off_H0=0,off_W0=0,off_H1=0,off_W1=0,
                 normalize_bwd=False,k_agg=-1,
                 fwd_version="v1", itype_fwd="int", itype_bwd="int",
                 rbwd=True, nbwd=1, exact=False, use_atomic=True,
                 queries_per_thread=2, neigh_per_thread=2, channel_groups=-1):
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
        self.fwd_version = fwd_version
        self.itype_fwd = itype_fwd
        self.itype_bwd = itype_bwd

        # -- manage patch and search boundaries --
        self.reflect_bounds = reflect_bounds
        self.full_ws = full_ws
        self.full_ws_time = full_ws_time

        # -- special mods to "self" search --
        self.topk_mode = topk_mode
        self.anchor_self = anchor_self
        self.remove_self = remove_self

        # -- searching offsets --
        self.use_adj = use_adj
        self.off_H0 = off_H0
        self.off_W0 = off_W0
        self.off_H1 = off_H1
        self.off_W1 = off_W1

        # -- backprop params --
        self.normalize_bwd = normalize_bwd
        self.k_agg = k_agg
        self.rbwd = rbwd
        self.nbwd = nbwd
        self.exact = exact
        self.use_atomic = use_atomic
        self.queries_per_thread = queries_per_thread
        self.neigh_per_thread = neigh_per_thread
        self.channel_groups = channel_groups


    def forward(self, vid0, vid1, fflow, bflow, batchsize=-1):
        assert self.ws > 0,"Must have nonzero spatial search window"
        assert self.wt >= 0,"Must have nonnegative time search window"
        return NonLocalSearchFunction.apply(vid0,vid1,fflow,bflow,
                                            self.ws,self.wt,self.ps,self.k,
                                            self.nheads,batchsize,
                                            self.dist_type,self.stride0,
                                            self.stride1,self.dilation,self.pt,
                                            self.reflect_bounds,
                                            self.full_ws,self.full_ws_time,
                                            self.topk_mode,self.anchor_self,
                                            self.remove_self,self.use_adj,
                                            self.off_H0,self.off_W0,
                                            self.off_H1,self.off_W1,
                                            self.normalize_bwd,self.k_agg,
                                            self.fwd_version,
                                            self.itype_fwd,self.itype_bwd,
                                            self.rbwd,self.nbwd,self.exact,
                                            self.use_atomic,
                                            self.queries_per_thread,
                                            self.neigh_per_thread,
                                            self.channel_groups)

    def flops(self,T,F,H,W):
        print("hi.")
        return 0

        # -- unpack --
        ps,pt = self.ps,self.pt

        # -- compute search --
        nrefs_hw = ((H-1)//self.stride0+1) * ((W-1)//self.stride0+1)
        nrefs = T * HD * nrefs_hw
        nsearch = ws_h * ws_w * (2*wt+1)
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
#            [Direct API]  stnls.search.nls(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid0, vid1, fflow, bflow,
           ws, wt, ps, k, nheads=1, batchsize=-1,
           dist_type="l2", stride0=4, stride1=1,
           dilation=1, pt=1, reflect_bounds=True,
           full_ws=True, full_ws_time=True,
           anchor_self=True, remove_self=False,
           use_adj=False, off_H0=0, off_W0=0, off_H1=0, off_W1=0,
           normalize_bwd=False, k_agg=-1,
           fwd_version="v1", itype_fwd="int",itype_bwd="int",
           rbwd=False, nbwd=1, exact=False, use_atomic=True,
           queries_per_thread=2, neigh_per_thread=2, channel_groups=-1):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = NonLocalSearchFunction.apply
    return fxn(vid0,vid1,fflow,bflow,ws,wt,ps,k,
               nheads,batchsize,dist_type,
               stride0,stride1,dilation,pt,reflect_bounds,
               full_ws,full_ws_time,anchor_self,remove_self,
               use_adj,off_H0,off_W0,off_H1,off_W1,
               normalize_bwd,k_agg,fwd_version,
               itype_fwd,itype_bwd,
               rbwd,nbwd,exact,use_atomic,
               queries_per_thread,neigh_per_thread,channel_groups)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ws":-1,"wt":-1,"ps":7,"k":10,
             "nheads":1,"dist_type":"l2",
             "stride0":4, "stride1":1, "dilation":1, "pt":1,
             "reflect_bounds":True, "full_ws":True, "full_ws_time":True,
             "anchor_self":True, "remove_self":False,"fwd_version":"v1",
             "use_adj":False, "off_H0":0,"off_W0":0,"off_H1":0,"off_W1":0,
             "normalize_bwd": False, "k_agg":-1,"rbwd":False, "nbwd":1,
             "itype_fwd":"int","itype_bwd":"int",
             "exact":False, "use_atomic": True, "topk_mode":"default",
             "queries_per_thread":2,"neigh_per_thread":2,"channel_groups":-1}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    cfg = extract_config(cfg)
    search = NonLocalSearch(cfg.ws, cfg.wt, cfg.ps, cfg.k, nheads=cfg.nheads,
                            dist_type=cfg.dist_type, stride0=cfg.stride0,
                            stride1=cfg.stride1, dilation=cfg.dilation, pt=cfg.pt,
                            reflect_bounds=cfg.reflect_bounds, topk_mode=cfg.topk_mode,
                            full_ws=cfg.full_ws, full_ws_time=cfg.full_ws_time,
                            anchor_self=cfg.anchor_self, remove_self=cfg.remove_self,
                            use_adj=cfg.use_adj,off_H0=cfg.off_H0,off_W0=cfg.off_W0,
                            off_H1=cfg.off_H1,off_W1=cfg.off_W1,
                            normalize_bwd=cfg.normalize_bwd,k_agg=cfg.k_agg,
                            fwd_version=cfg.fwd_version,
                            itype_fwd=cfg.itype_fwd,itype_bwd=cfg.itype_bwd,
                            rbwd=cfg.rbwd, nbwd=cfg.nbwd, exact=cfg.exact,
                            use_atomic=cfg.use_atomic,
                            queries_per_thread=cfg.queries_per_thread,
                            neigh_per_thread=cfg.neigh_per_thread,
                            channel_groups=cfg.channel_groups)
    return search

