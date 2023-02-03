# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import dnls_cuda

# -- package --
import dnls

# -- api --
from .utils import extract_pairs

# -- local --
from .utils import dist_type_select,shape_vids,filter_k
from .nls_bwd_impl import nls_backward
from .non_local_search import _apply as nls_apply
from .refinement import _apply as refine_apply

class ApproxSpaceSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, fflow, bflow,
                ws, wt, ps, k, wr, kr, scale, nheads=1, batchsize=-1,
                dist_type="prod", stride0=4, stride1=1,
                dilation=1, pt=1, reflect_bounds=True, full_ws=False,
                anchor_self=False, remove_self=False,
                use_adj=True, off_H0=0, off_W0=0, off_H1=0, off_W1=0,
                rbwd=True, nbwd=1, exact=False):

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #     1.) Run Exact Search With Coarse Grid
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- exact search -
        assert isinstance(scale,int),"Must be int."
        stride0_c = scale*stride0
        vid0,vid1 = shape_vids(nheads,[vid0,vid1])
        anchor_self_e = True
        k_exact = k
        dists,inds = nls_apply(vid0,vid1,fflow,bflow,
                               ws,wt,ps,k_exact,nheads,batchsize,
                               dist_type,stride0_c,stride1,
                               dilation,pt,reflect_bounds,full_ws,
                               anchor_self_e,remove_self,use_adj,
                               off_H0,off_W0,off_H1,off_W1,
                               rbwd,nbwd,exact)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #     2.) Upsampling Indices to Full-Grid
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- upsampling --
        T,_,H,W = vid0.shape[-4:]
        inds = dnls.nn.interpolate_inds(filter_k(inds,kr,k),scale,stride0,T,H,W)

        # -- jittering to get uniq --
        inds = dnls.nn.jitter_unique_inds(inds,3,k,H,W)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #     3.) Refine/Evaluate On New Grid
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- refinement --
        dists,inds = refine_apply(vid0, vid1, inds,
                                  ws,ps,k,wr,-1,nheads,batchsize,
                                  dist_type,stride0,stride1,
                                  dilation,pt,reflect_bounds,full_ws,
                                  anchor_self,remove_self,use_adj,
                                  off_H0,off_W0,off_H1,off_W1,
                                  rbwd,nbwd,exact)

        # -- setup ctx --
        dist_type_i,descending,dval = dist_type_select(dist_type)
        ctx.save_for_backward(inds,vid0,vid1)
        ctx.mark_non_differentiable(inds)
        ctx.vid_shape = vid0.shape
        ctx_vars = {"batchsize":batchsize,"stride0":stride0,"ps":ps,"pt":pt,
                    "dil":dilation,"reflect_bounds":reflect_bounds,
                    "rbwd":rbwd,"exact":exact,"nbwd":nbwd,
                    "use_adj":use_adj,"off_H0":off_H0,"off_W0":off_W0,
                    "off_H1":off_H1,"off_W1":off_W1,"dist_type_i":dist_type_i}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds_is_none):
        grad0,grad1 = nls_backward(ctx, grad_dists, grad_inds_is_none)
        return grad0,grad1,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,None,\
            None,None


class ApproxSpaceSearch(th.nn.Module):

    def __init__(self, ws, wt, ps, k, wr, kr, scale, nheads=1,
                 dist_type="prod", stride0=4, stride1=1, dilation=1, pt=1,
                 reflect_bounds=True, full_ws=False,
                 anchor_self=False, remove_self=False,
                 use_adj=True,off_H0=0,off_W0=0,off_H1=0,off_W1=0,
                 rbwd=True, nbwd=1, exact=False):
        super().__init__()

        # -- core search params --
        self.ws = ws
        self.wt = wt
        self.ps = ps
        self.k = k
        self.wr = wr
        self.kr = kr
        self.scale = scale
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

        # -- searching offsets --
        self.use_adj = use_adj
        self.off_H0 = off_H0
        self.off_W0 = off_W0
        self.off_H1 = off_H1
        self.off_W1 = off_W1

        # -- backprop params --
        self.nbwd = nbwd
        self.exact = exact
        self.rbwd = rbwd

    def forward(self, vid0, vid1, fflow, bflow, batchsize=-1):
        fxn = ApproxSpaceSearchFunction.apply
        return fxn(vid0,vid1,fflow,bflow,
                   self.ws,self.wt,self.ps,self.k,
                   self.wr,self.kr,self.scale,
                   self.nheads,batchsize,self.dist_type,
                   self.stride0,self.stride1,
                   self.dilation,self.pt,
                   self.reflect_bounds,self.full_ws,
                   self.anchor_self,self.remove_self,
                   self.use_adj,self.off_H0,self.off_W0,
                   self.off_H1,self.off_W1,
                   self.rbwd,self.nbwd,self.exact)

    def flops(self,B,HD,T,F,H,W):
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
        return 0


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       [Direct API] dnls.search.approx_space(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid0, vid1, fflow, bflow,
           ws, wt, ps, k, wr, kr, scale, nheads=1, batchsize=-1,
           dist_type="prod", stride0=4, stride1=1,
           dilation=1, pt=1, reflect_bounds=True, full_ws=False,
           anchor_self=True, remove_self=False,
           use_adj=True, off_H0=0, off_W0=0, off_H1=0, off_W1=0,
           rbwd=True, nbwd=1, exact=False):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = ApproxSpaceSearchFunction.apply
    return fxn(vid0, vid1, fflow, bflow,
               ws, wt, ps, k, wr, kr, scale,
               nheads, batchsize, dist_type, stride0, stride1,
               dilation, pt, reflect_bounds,
               full_ws, anchor_self, remove_self,
               use_adj, off_H0, off_W0, off_H1, off_W1,
               rbwd, nbwd, exact)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       [Python Dict API] dnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def extract_config(cfg):
    pairs = {"ws":-1,"wt":-1,"ps":7,"k":10,
             "wr_s":1,"kr_s":-1,"scale":2,
             "nheads":1,"dist_type":"prod",
             "stride0":4, "stride1":1, "dilation":1, "pt":1,
             "reflect_bounds":True, "full_ws":False,
             "anchor_self":True, "remove_self":False,
             "use_adj":True,"off_H0":0,"off_W0":0,"off_H1":0,"off_W1":0,
             "rbwd":True, "nbwd":1, "exact":False}
    return extract_pairs(pairs,cfg)

def init(cfg):
    search = ApproxSpaceSearch(cfg.ws, cfg.wt, cfg.ps, cfg.k,
                               cfg.wr_s, cfg.kr_s, cfg.scale,
                          nheads=cfg.nheads, dist_type=cfg.dist_type,
                          stride0=cfg.stride0, stride1=cfg.stride1,
                          dilation=cfg.dilation, pt=cfg.pt,
                          reflect_bounds=cfg.reflect_bounds, full_ws=cfg.full_ws,
                          anchor_self=cfg.anchor_self, remove_self=cfg.remove_self,
                          use_adj=cfg.use_adj,off_H0=cfg.off_H0,off_W0=cfg.off_W0,
                          off_H1=cfg.off_H1,off_W1=cfg.off_W1,
                          rbwd=cfg.rbwd, nbwd=cfg.nbwd, exact=cfg.exact)
    return search

