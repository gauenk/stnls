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
from .batching_utils import run_batched,batching_info
# from .nls_bwd_impl import nls_backward
from .ref_bwd_impl import ref_backward

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Forward Logic
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def refine_forward(batchsize,*args):
    vid_idx = 0
    ws_idx,wt_idx = 3,4
    stride0_idx = 8
    ntotal,nbatches,batchsize = batching_info(args[vid_idx],args[stride0_idx],
                                              args[ws_idx],args[wt_idx],
                                              batchsize)
    if nbatches == 1: # shortcut
        qshift,nqueries = 0,-1
        return refine_fwd_main(qshift,nqueries,*args)
    else:
        return run_batched(refine_fwd_main,batchsize,vid_idx,
                           stride0_idx,ws_idx,wt_idx,*args)

def refine_fwd_main(qshift, Q, vid0, vid1, qinds,
                    ws, wr, ps, k, dist_type,
                    stride0, stride1, dilation, pt,
                    anchor_self, remove_self, reflect_bounds,
                    full_ws, use_adj, itype_fwd, off_H0, off_W0, off_H1, off_W1):

    # -- fix negative Q --
    if Q > 0:
        qinds = qinds[:,:,qshift:qshift+Q].contiguous()

    # -- search space --
    wr_h,wr_w = wr,wr
    ws_h,ws_w = ws,ws
    search_abs = ws == -1
    if search_abs:
        nH0 = (H-1)//stride0+1
        nW0 = (W-1)//stride0+1
        ws_h,ws_w = nH0,nW0

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    device = qinds.device
    B,HD,Q,K = qinds.shape[:-1]
    base_shape = (B,HD,Q,K,wr_h,wr_w)
    dists,inds = allocate_pair(base_shape,device,vid0.dtype,idist_val,itype_fwd)
    imode = 0 if itype_fwd == "int" else 1

    # -- run --
    # print(vid0.shape,qinds.shape)
    stnls_cuda.refinement_forward(vid0, vid1, qinds, dists, inds,
                                  ws_h, ws_w, ps, k, dist_type_i,
                                  stride0, stride1, dilation, pt, qshift,
                                  reflect_bounds, full_ws, use_adj,
                                  off_H0, off_W0, off_H1, off_W1, imode)
    # print("dists [max,min]: ",th.max(dists).item(),th.min(dists).item())

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
    dists,inds = manage_self(dists,inds,anchor_self,
                             remove_self,qshift,stride0,H,W)

    # -- topk --
    qinds = rearrange(qinds,'b hd q k tr -> (b hd q) k tr')
    k = min(qinds.shape[1],k)
    # print(dists[0,0,0],inds[0,0,0],len(dists[0,0,0]),k)
    dists,inds = stnls.nn.topk(dists,inds,k,dim=3,anchor=anchor_self,
                              descending=descending,unique=True,qinds=qinds)
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
                ws, ps, k, wr, kr, nheads=1, batchsize=-1,
                dist_type="prod", stride0=4, stride1=1,
                dilation=1, pt=1, reflect_bounds=True, full_ws=False,
                anchor_self=True, remove_self=False,
                use_adj=False, off_H0=0, off_W0=0, off_H1=0, off_W1=0,
                normalize_bwd=False, k_agg=-1,
                itype_fwd="int",itype_bwd="int",
                rbwd=False, nbwd=1, exact=False,
                use_atomic=True, queries_per_thread=4,
                neigh_per_thread=4, channel_groups=-1):
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
        # print("vid0.shape: ",vid0.shape,qinds.shape,nheads)
        B,HD,T,F,H,W = vid0.shape
        assert qinds.shape[1] == HD

        # -- filter only to kr --
        qinds = filter_k(qinds,kr)
        qinds = qinds.contiguous()

        # -- run fwd pass --
        dists,inds = refine_forward(batchsize, vid0, vid1, qinds,
                                    ws, wr, ps, k, dist_type,
                                    stride0, stride1, dilation, pt,
                                    anchor_self,remove_self, reflect_bounds,
                                    full_ws, use_adj, itype_fwd,
                                    off_H0, off_W0, off_H1, off_W1)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        qinds = get_ctx_qinds(itype_bwd,qinds)
        ctx.save_for_backward(inds,vid0,vid1,qinds)
        if itype_bwd == "int":
            ctx.mark_non_differentiable(inds)
        ctx.vid_shape = vid0.shape
        ctx_vars = {"batchsize":batchsize,"stride0":stride0,"stride1":stride1,
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
    def backward(ctx, grad_dists, grad_inds_is_none):
        # print("refinement: ",grad_dists.shape,grad_inds_is_none)
        grad0,grad1,grad_qinds = ref_backward(ctx, grad_dists, grad_inds_is_none)
        return grad0,grad1,grad_qinds,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,None

class RefineSearch(th.nn.Module):

    def __init__(self, ws, ps, k, wr, kr, nheads=1,
                 dist_type="l2", stride0=4, stride1=1, dilation=1, pt=1,
                 reflect_bounds=True, full_ws=False,
                 anchor_self=False, remove_self=False,
                 use_adj=False,off_H0=0,off_W0=0,off_H1=0,off_W1=0,
                 normalize_bwd=False, k_agg=-1,
                 itype_fwd="int",itype_bwd="int",
                 rbwd=True, nbwd=1, exact=False, use_atomic=True,
                 queries_per_thread=4, neigh_per_thread=4, channel_groups=-1):
        super().__init__()

        # -- core search params --
        self.ws = ws
        self.ps = ps
        self.k = k
        self.wr = wr
        self.kr = kr
        self.nheads = nheads
        self.dist_type = dist_type
        self.stride0 = stride0
        self.stride1 = stride1
        self.dilation = dilation
        self.pt = pt

        # -- manage patch and search boundaries --
        self.reflect_bounds = reflect_bounds
        self.full_ws = full_ws

        # -- special mods to "self" search --
        self.anchor_self = anchor_self
        self.remove_self = remove_self

        # -- with/without grads --
        self.itype_fwd = itype_fwd
        self.itype_bwd = itype_bwd

        # -- searching offsets --
        self.use_adj = use_adj
        self.off_H0 = off_H0
        self.off_W0 = off_W0
        self.off_H1 = off_H1
        self.off_W1 = off_W1

        # -- backprop params --
        self.normalize_bwd = normalize_bwd
        self.k_agg = k_agg
        self.nbwd = nbwd
        self.exact = exact
        self.use_atomic = use_atomic
        self.rbwd = rbwd
        self.queries_per_thread = queries_per_thread
        self.neigh_per_thread = neigh_per_thread
        self.channel_groups = channel_groups


    def forward(self,vid0,vid1,qinds,batchsize=-1):
        return RefineSearchFunction.apply(vid0,vid1,qinds,
                                          self.ws,self.ps,self.k,
                                          self.wr,self.kr,self.nheads,batchsize,
                                          self.dist_type,self.stride0,self.stride1,
                                          self.dilation,self.pt,
                                          self.reflect_bounds,self.full_ws,
                                          self.anchor_self,self.remove_self,
                                          self.use_adj,self.off_H0,self.off_W0,
                                          self.off_H1,self.off_W1,
                                          self.normalize_bwd,self.k_agg,
                                          self.itype_fwd,self.itype_bwd,
                                          self.rbwd,self.nbwd,
                                          self.exact,self.use_atomic,
                                          self.queries_per_thread,
                                          self.neigh_per_thread,
                                          self.channel_groups)

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
           ws, ps, k, wr, kr=-1, nheads=1, batchsize=-1,
           dist_type="l2", stride0=4, stride1=1,
           dilation=1, pt=1, reflect_bounds=True, full_ws=False,
           anchor_self=True, remove_self=False,
           use_adj=False, off_H0=0, off_W0=0, off_H1=0, off_W1=0,
           normalize_bwd=False, k_agg=-1,rbwd=False, nbwd=1, exact=False, use_atomic=True,
           queries_per_thread=4, neigh_per_thread=4, channel_groups=-1):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = RefineSearchFunction.apply
    return fxn(vid0, vid1, qinds,
               ws, ps, k, wr, kr, nheads, batchsize,
               dist_type, stride0, stride1,
               dilation, pt, reflect_bounds,
               full_ws, anchor_self, remove_self,
               use_adj, off_H0, off_W0, off_H1, off_W1,
               normalize_bwd, k_agg, rbwd, nbwd, exact, use_atomic,
               queries_per_thread, neigh_per_thread, channel_groups)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    pairs = {"ws":-1,"wt":-1,"ps":7,"k":10,"wr":1,"kr":-1,
             "nheads":1,"dist_type":"l2",
             "stride0":4, "stride1":1, "dilation":1, "pt":1,
             "reflect_bounds":True, "full_ws":False,
             "anchor_self":True, "remove_self":False,
             "use_adj":False,"off_H0":0,"off_W0":0,"off_H1":0,"off_W1":0,
             "normalize_bwd": False, "k_agg":-1,
             "itype_fwd":"int", "itype_bwd":"int",
             "rbwd":False, "nbwd":1, "exact":False, "use_atomic": True,
             "queries_per_thread":2,"neigh_per_thread":2,"channel_groups":-1}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    search = RefineSearch(cfg.ws, cfg.ps, cfg.k, cfg.wr, cfg.kr,
                          nheads=cfg.nheads, dist_type=cfg.dist_type,
                          stride0=cfg.stride0, stride1=cfg.stride1,
                          dilation=cfg.dilation, pt=cfg.pt,
                          reflect_bounds=cfg.reflect_bounds, full_ws=cfg.full_ws,
                          anchor_self=cfg.anchor_self, remove_self=cfg.remove_self,
                          use_adj=cfg.use_adj,off_H0=cfg.off_H0,off_W0=cfg.off_W0,
                          off_H1=cfg.off_H1,off_W1=cfg.off_W1,
                          normalize_bwd=cfg.normalize_bwd, k_agg=cfg.k_agg,
                          itype_fwd=cfg.itype_fwd,itype_bwd=cfg.itype_bwd,
                          rbwd=cfg.rbwd, nbwd=cfg.nbwd,
                          exact=cfg.exact, use_atomic=cfg.use_atomic,
                          queries_per_thread=cfg.neigh_per_thread,
                          neigh_per_thread=cfg.neigh_per_thread,
                          channel_groups=cfg.channel_groups)

    return search

