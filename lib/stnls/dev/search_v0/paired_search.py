
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
from .utils import get_ctx_flows,ensure_flow_shape
from .shared import manage_self
from .paired_bwd_impl import paired_backward
from .batching_utils import run_batched,batching_info

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Forward Logic
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def paired_forward(batchsize,*args):
    qshift,nqueries = 0,-1
    return paired_fwd_main(qshift,nqueries,*args)

def ensure_flow_dim(flow):
    if flow.ndim == 4:
        flow = flow[:,None] # add nheads
    return flow

def paired_fwd_main(qshift, Q, frame0, frame1, flow,
                    ws, ps, k, dist_type,
                    stride0, stride1, dilation, pt,
                    topk_mode, anchor_self, remove_self, reflect_bounds,
                    full_ws, full_ws_time, use_adj,
                    fwd_version, itype, off_H0, off_W0, off_H1, off_W1):

    # -- unpack --
    # itype = "int"
    device = frame0.device
    B,HD_fr,C,H,W = frame0.shape
    HD_flow = flow.shape[1]
    # print(frame0.shape,flow.shape)
    assert flow.ndim == 5
    HD = max(HD_flow,HD_fr)

    # -- derived shapes --
    nH0 = (H-1)//stride0+1
    nW0 = (W-1)//stride0+1
    Q = nH0*nW0 if Q <= 0 else Q

    # -- search space --
    ws_h,ws_w = ws,ws

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    base_shape = (B,HD,Q,ws_h,ws_w)
    dists,inds = allocate_pair_2d(base_shape,device,frame0.dtype,idist_val,itype)
    # print("inds.shape: ",inds.shape)

    # -- forward --
    fwd_version = "v1"
    if fwd_version == "v1":
        if itype == "int":
            fwd_fxn = stnls_cuda.paired_search_forward
        else:
            fwd_fxn = stnls_cuda.paired_search_bilin2d_forward
            stride1 = float(stride1)
    else:
        raise ValueError(f"Uknown version [{version}]")
    # print(frame0.shape,flow.shape,dists.shape,inds.shape)
    fwd_fxn(frame0, frame1, flow, dists, inds,
            ps, k, dist_type_i, stride0,
            stride1, dilation, qshift,
            reflect_bounds, full_ws, full_ws_time,
            use_adj, off_H0, off_W0, off_H1, off_W1)

    # print(frame0.shape,frame1.shape,flow.shape,inds.shape)
    # -- compress search region --
    dists=dists.view(B,HD,Q,-1)
    inds=inds.view(B,HD,Q,-1,2)
    # th.cuda.synchronize()

    # -- fill nan --
    fill_val = -np.inf if dist_type == "prod" else np.inf
    dists = th.nan_to_num(dists,fill_val)

    # -- manage self dists --
    inds = th.cat([th.zeros_like(inds[...,[0]]),inds],-1)
    dists,inds = manage_self(dists,inds,anchor_self,
                             remove_self,qshift,stride0,H,W)
    inds = inds[...,1:]
    # print(inds.shape)
    # print(inds[0,0,:,0])
    # exit()

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

class PairedSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, frame0, frame1, flow,
                ws, ps, k, nheads=1, batchsize=-1,
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

        # -- run [optionally batched] forward function --
        dists,inds = paired_forward(batchsize, frame0, frame1, flow,
                                    ws, ps, k, dist_type,
                                    stride0, stride1, dilation, pt,
                                    topk_mode, anchor_self, remove_self,
                                    reflect_bounds, full_ws, full_ws_time,
                                    use_adj, fwd_version, itype_fwd,
                                    off_H0, off_W0, off_H1, off_W1)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        flow,_ = get_ctx_flows(itype_bwd,flow,flow)
        ctx.save_for_backward(inds,frame0,frame1,flow)
        if itype_bwd == "int":
            ctx.mark_non_differentiable(inds)
        ctx.vid_shape = frame0.shape
        ctx_vars = {"batchsize":batchsize,"stride0":stride0,"stride1":stride1,
                    "ps":ps,"pt":pt,"ws":ws,"dil":dilation,
                    "reflect_bounds":reflect_bounds,
                    "fwd_version":fwd_version,"normalize_bwd":normalize_bwd,
                    "k_agg":k_agg,"rbwd":rbwd,"exact":exact,"nbwd":nbwd,
                    "use_adj":use_adj,"off_H0":off_H0,"off_W0":off_W0,
                    "off_H1":off_H1,"off_W1":off_W1,
                    "dist_type_i":dist_type_i,"itype_bwd":itype_bwd}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds):
        grad0,grad1,gflow = paired_backward(ctx, grad_dists, grad_inds)
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


    def paired_vids(self, vid0, vid1, acc_flows, wt, skip_self=False):
        dists,inds = [],[]
        T = vid0.shape[1]
        zflow = th.zeros_like(acc_flows.fflow[:,0,0])
        for ti in range(T):
            # if ti != 1: continue

            swap = False
            t_inc = 0
            prev_t = ti
            t_shift = min(0,ti-wt) + max(0,ti + wt - (T-1))
            t_max = min(T-1,ti + wt - t_shift);
            # print(t_shift,t_max)
            tj = ti

            dists_i,inds_i = [],[]
            for _tj in range(2*wt+1):

                # -- update search frame --
                prev_t = tj
                tj = prev_t + t_inc
                swap = tj > t_max
                t_inc = 1 if (t_inc == 0) else t_inc
                t_inc = -1 if swap else t_inc
                tj = ti-1 if swap else tj
                prev_t = ti if swap else prev_t
                # print(ti,tj,t_inc,swap)

                frame0 = vid0[:,ti]
                frame1 = vid1[:,tj]
                if (ti == tj) and skip_self: continue
                if ti == tj:
                    flow = zflow
                elif ti < tj:
                    # print("fwd: ",ti,tj,tj-ti-1)
                    # flow = acc_flows.fflow[:,tj - ti - 1]
                    flow = acc_flows.fflow[:,ti,tj-ti-1]
                elif ti > tj:
                    # print("bwd: ",ti,tj,ti-tj-1)
                    # flow = acc_flows.bflow[:,ti - tj - 1]
                    flow = acc_flows.bflow[:,ti,ti-tj-1]
                flow = flow.float()
                dists_ij,inds_ij = self.forward(frame0,frame1,flow)
                inds_t = tj*th.ones_like(inds_ij[...,[0]])
                inds_ij = th.cat([inds_t,inds_ij],-1)
                # print("inds_ij.shape: ",inds_ij.shape,inds_t.shape)
                dists_i.append(dists_ij)
                inds_i.append(inds_ij)
            dists_i = th.cat(dists_i,-1)
            inds_i = th.cat(inds_i,-2)
            dists.append(dists_i)
            inds.append(inds_i)
        dists = th.cat(dists,-2)
        inds = th.cat(inds,-3)
        # print("inds.shape: ",inds.shape)
        return dists,inds

    def paired_stacking(self, vid0, vid1, acc_flows, wt, stack_fxn):
        dists,inds = [],[]
        T = vid0.shape[1]
        zflow = th.zeros_like(acc_flows.fflow[:,0,0])
        for ti in range(T):
            # if ti != 1: continue

            swap = False
            t_inc = 0
            prev_t = ti
            t_shift = min(0,ti-wt) + max(0,ti + wt - (T-1))
            t_max = min(T-1,ti + wt - t_shift);
            # print(t_shift,t_max)
            tj = ti

            dists_i,inds_i = [],[]
            for _tj in range(2*wt+1):

                # -- update search frame --
                prev_t = tj
                tj = prev_t + t_inc
                swap = tj > t_max
                t_inc = 1 if (t_inc == 0) else t_inc
                t_inc = -1 if swap else t_inc
                tj = ti-1 if swap else tj
                prev_t = ti if swap else prev_t
                # print(ti,tj,t_inc,swap)

                frame0 = vid0[:,ti]
                frame1 = vid1[:,tj]
                if ti == tj:
                    flow = zflow
                elif ti < tj:
                    # print("fwd: ",ti,tj,tj-ti-1)
                    # flow = acc_flows.fflow[:,tj - ti - 1]
                    flow = acc_flows.fflow[:,ti,tj-ti-1]
                elif ti > tj:
                    # print("bwd: ",ti,tj,ti-tj-1)
                    # flow = acc_flows.bflow[:,ti - tj - 1]
                    flow = acc_flows.bflow[:,ti,ti-tj-1]
                flow = flow.float()
                dists_ij,inds_ij = self.forward(frame0,frame1,flow)
                inds_t = tj*th.ones_like(inds_ij[...,[0]])
                inds_ij = th.cat([inds_t,inds_ij],-1)
                dists_i.append(dists_ij)
                inds_i.append(inds_ij)
            dists_i = th.cat(dists_i,-1)
            inds_i = th.cat(inds_i,-2)
            dists.append(dists_i)
            inds.append(inds_i)
        dists = th.cat(dists,-2)
        inds = th.cat(inds,-3)
        return dists,inds

    def forward(self, frame0, frame1, flow, batchsize=-1):
        assert self.ws > 0,"Must have nonzero spatial search window"
        return PairedSearchFunction.apply(frame0,frame1,flow,
                                          self.ws,self.ps,self.k,
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
#            [Direct API]  stnls.search.paired(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(frame0, frame1, flow,
           ws, ps, k, nheads=1, batchsize=-1,
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
    fxn = PairedSearchFunction.apply
    return fxn(frame0,frame1,flow,ws,ps,k,
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
    pairs = {"ws":-1,"ps":7,"k":10,
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
    search = PairedSearch(cfg.ws, cfg.ps, cfg.k, nheads=cfg.nheads,
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

