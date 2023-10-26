
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
from .shared import manage_self,reflect_bounds_warning
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

def paired_forward(frame0, frame1, flow,
                    ws, ps, k, dist_type,
                    stride0, stride1, dilation, pt,
                    self_action, reflect_bounds,
                    full_ws, use_adj, itype):

    # -- unpack --
    device = frame0.device
    B,HD_fr,C,H,W = frame0.shape
    HD_flow = flow.shape[1]
    # print(frame0.shape,flow.shape)
    assert flow.ndim == 5
    HD = max(HD_flow,HD_fr)
    patch_offset = 0 if use_adj else -(ps//2)

    # -- derived shapes --
    nH0 = (H-1)//stride0+1
    nW0 = (W-1)//stride0+1
    Q = nH0*nW0

    # -- search space --
    ws_h,ws_w = ws,ws

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    base_shape = (B,HD,Q,ws_h,ws_w)
    dists,inds = allocate_pair_2d(base_shape,device,frame0.dtype,idist_val,itype)
    # print("inds.shape: ",inds.shape)

    # -- forward --
    if itype == "int":
        flow = flow.round()
        inds = inds.int()
        stride1 = max(1,int(stride1))
        fwd_fxn = stnls_cuda.paired_search_int_forward
    else:
        fwd_fxn = stnls_cuda.paired_search_bilin2d_forward
        stride1 = float(stride1)
    # print(frame0.shape,flow.shape,dists.shape,inds.shape)
    fwd_fxn(frame0, frame1, flow, dists, inds,
            ps, k, stride0, stride1, dilation,
            reflect_bounds, full_ws, patch_offset, dist_type_i)

    # print(frame0.shape,frame1.shape,flow.shape,inds.shape)
    # -- compress search region --
    dists=dists.view(B,HD,Q,-1)
    inds=inds.view(B,HD,Q,-1,2)
    # th.cuda.synchronize()

    # -- anchor --
    assert self_action in [None,"anchor","anchor_each"]
    anchor_self = False if self_action is None else "anchor" in self_action
    if self_action is None: pass
    elif "anchor" in self_action:
        stnls.nn.anchor_self(dists,inds,stride0,H,W)
    else:
        raise ValueError(f"Uknown option for self_action [{self_action}]")

    # # # -- manage self dists --
    # # anchor_self = self_action == "anchor"
    # # remove_self = self_action == "remove"
    # # inds = th.cat([th.zeros_like(inds[...,[0]]),inds],-1)
    # # dists,inds = manage_self(dists,inds,anchor_self,
    # #                          remove_self,0,stride0,H,W)
    # inds = inds[...,1:]
    # # print(inds.shape)
    # # print(inds[0,0,:,0])
    # # exit()

    # -- topk --
    if k > 0:
        dim = 3
        dists=dists.view(B,HD,Q,ws*ws)
        inds=inds.view(B,HD,Q,ws*ws,2)
        dists,inds = stnls.nn.topk(dists,inds,k,dim=dim,anchor=anchor_self,
                                   descending=descending)

    # -- reshape --
    dists=dists.reshape(B,HD,1,nH0,nW0,-1)
    inds=inds.reshape(B,HD,1,nH0,nW0,-1,2)

    return dists,inds

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Pytorch Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PairedSearchFunction(th.autograd.Function):


    @staticmethod
    def forward(ctx, frame0, frame1, flow,
                ws, ps, k, nheads=1,
                dist_type="prod", stride0=4, stride1=1,
                dilation=1, pt=1, reflect_bounds=True,
                full_ws=True, self_action=None,
                use_adj=False, normalize_bwd=False, k_agg=-1, itype="int"):

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
        dists,inds = paired_forward(frame0, frame1, flow,
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
                 full_ws=True, self_action=None, use_adj=False,
                 normalize_bwd=False,k_agg=-1,
                 itype="int"):
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
        dists,inds = [],[]
        T = vid0.shape[1]
        zflow = th.zeros_like(flows[:,:,0,0])
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

                if (ti == tj) and skip_self: continue
                frame0 = vid0[:,ti]
                frame1 = vid1[:,tj]
                if _tj > 0: flow = flows[:,:,ti,_tj-1]
                else: flow = zflow
                flow = flow.float()
                dists_ij,inds_ij = self.forward(frame0,frame1,flow)
                inds_t = (tj-ti)*th.ones_like(inds_ij[...,[0]])
                inds_ij = th.cat([inds_t,inds_ij],-1)
                dists_i.append(dists_ij)
                inds_i.append(inds_ij)
            # -- stack across K --
            dists_i = th.cat(dists_i,-1)
            inds_i = th.cat(inds_i,-2)
            dists.append(dists_i)
            inds.append(inds_i)
        # -- stack across time --
        dists = th.cat(dists,-4)
        inds = th.cat(inds,-5)
        # print("inds.shape: ",inds.shape)
        return dists,inds

    # def paired_stacking(self, vid0, vid1, acc_flows, wt, stack_fxn):
    #     dists,inds = [],[]
    #     T = vid0.shape[1]
    #     zflow = th.zeros_like(acc_flows.fflow[:,0,0])
    #     for ti in range(T):
    #         # if ti != 1: continue

    #         swap = False
    #         t_inc = 0
    #         prev_t = ti
    #         t_shift = min(0,ti-wt) + max(0,ti + wt - (T-1))
    #         t_max = min(T-1,ti + wt - t_shift);
    #         # print(t_shift,t_max)
    #         tj = ti

    #         dists_i,inds_i = [],[]
    #         for _tj in range(2*wt+1):

    #             # -- update search frame --
    #             prev_t = tj
    #             tj = prev_t + t_inc
    #             swap = tj > t_max
    #             t_inc = 1 if (t_inc == 0) else t_inc
    #             t_inc = -1 if swap else t_inc
    #             tj = ti-1 if swap else tj
    #             prev_t = ti if swap else prev_t
    #             # print(ti,tj,t_inc,swap)

    #             frame0 = vid0[:,ti]
    #             frame1 = vid1[:,tj]
    #             if ti == tj:
    #                 flow = zflow
    #             elif ti < tj:
    #                 # print("fwd: ",ti,tj,tj-ti-1)
    #                 # flow = acc_flows.fflow[:,tj - ti - 1]
    #                 flow = acc_flows.fflow[:,ti,tj-ti-1]
    #             elif ti > tj:
    #                 # print("bwd: ",ti,tj,ti-tj-1)
    #                 # flow = acc_flows.bflow[:,ti - tj - 1]
    #                 flow = acc_flows.bflow[:,ti,ti-tj-1]
    #             flow = flow.float()
    #             dists_ij,inds_ij = self.forward(frame0,frame1,flow)
    #             inds_t = (tj-ti)*th.ones_like(inds_ij[...,[0]])
    #             inds_ij = th.cat([inds_t,inds_ij],-1)
    #             dists_i.append(dists_ij)
    #             inds_i.append(inds_ij)
    #         dists_i = th.cat(dists_i,-1)
    #         inds_i = th.cat(inds_i,-2)
    #         dists.append(dists_i)
    #         inds.append(inds_i)
    #     dists = th.cat(dists,-2)
    #     inds = th.cat(inds,-3)
    #     return dists,inds

    def forward(self, frame0, frame1, flow):
        assert self.ws > 0,"Must have nonzero spatial search window"
        return PairedSearchFunction.apply(frame0,frame1,flow,
                                          self.ws,self.ps,self.k,
                                          self.nheads,self.dist_type,self.stride0,
                                          self.stride1,self.dilation,self.pt,
                                          self.reflect_bounds,self.full_ws,
                                          self.self_action,self.use_adj,
                                          self.normalize_bwd,
                                          self.k_agg,self.itype)

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
    pairs = {"ws":-1,"ps":7,"k":10,
             "nheads":1,"dist_type":"l2",
             "stride0":4, "stride1":1, "dilation":1, "pt":1,
             "reflect_bounds":True, "full_ws":True,
             "self_action":None,"use_adj":False,
             "normalize_bwd": False, "k_agg":-1,
             "itype":"float",}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    cfg = extract_config(cfg)
    search = PairedSearch(cfg.ws, cfg.ps, cfg.k, nheads=cfg.nheads,
                          dist_type=cfg.dist_type, stride0=cfg.stride0,
                          stride1=cfg.stride1, dilation=cfg.dilation, pt=cfg.pt,
                          reflect_bounds=cfg.reflect_bounds,
                          full_ws=cfg.full_ws, self_action=cfg.self_action,
                          use_adj=cfg.use_adj,normalize_bwd=cfg.normalize_bwd,
                          k_agg=cfg.k_agg,itype=cfg.itype)
    return search

