# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import dnls_cuda

# -- package --
import dnls

# -- local --
from .utils import dist_type_select,shape_vids,filter_k
from .nls_bwd_impl import nls_backward
from .non_local_search import _apply as nls_apply
from .refinement import _apply as refine_apply

class ApproxSpaceSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, fflow, bflow,
                ws, wt, ps, k, wr, kr, scale, nheads=1, qshift=0, Q=-1,
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

        assert isinstance(scale,int),"Must be int."
        stride0_c = scale*stride0
        vid0,vid1 = shape_vids(nheads,[vid0,vid1])
        anchor_self_e = True
        dists,inds = nls_apply(vid0,vid1,fflow,bflow,
                               ws,wt,ps,-1,nheads,qshift,Q,
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

        T,_,H,W = vid0.shape[-4:]
        inds = dnls.nn.interpolate_inds(filter_k(inds,kr),scale,stride0,T,H,W)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        #     3.) Refine/Evaluate On New Grid
        #
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        dists,inds = refine_apply(vid0, vid1, inds,
                                  ws,ps,k,wr,-1,nheads,qshift,
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
        ctx_vars = {"qshift":qshift,"stride0":stride0,"ps":ps,"pt":pt,
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

    def forward(self, vid0, vid1, fflow, bflow, qshift=0, nqueries=-1):
        fxn = ApproxSpaceSearchFunction.apply
        return fxn(vid0,vid1,fflow,bflow,
                   self.ws,self.wt,self.ps,self.k,
                   self.wr,self.kr,self.scale,
                   self.nheads,qshift,nqueries,
                   self.dist_type,self.stride0,self.stride1,
                   self.dilation,self.pt,
                   self.reflect_bounds,self.full_ws,
                   self.anchor_self,self.remove_self,
                   self.use_adj,self.off_H0,self.off_W0,
                   self.off_H1,self.off_W1,
                   self.rbwd,self.nbwd,self.exact)

    def flops(self,HD,T,F,H,W):

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

_apply = ApproxSpaceSearchFunction.apply # api
