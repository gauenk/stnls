
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

# -- package --
import stnls

# -- api --
from .utils import extract_pairs

# -- local --
from .utils import shape_vids,allocate_inds,dist_type_select,allocate_vid
from .shared import manage_self
# from .nls_bwd_impl import nls_backward
# from .batching_utils import run_batched,batching_info
# from .n3mm_utils import IndexedMatmul1Efficient
from .n3mm_utils import matmult_fwd,matmult_bwd,vid_index_neighbours

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Forward Logic
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def n3mm_fwd_main(vid0, vid1, fflow, bflow,
                  ws, wt, ps, dist_type,
                  stride0, stride1, dilation, pt,
                  reflect_bounds, use_adj,
                  off_H0, off_W0, off_H1, off_W1):

    # -- unpack --
    device = vid0.device
    B,HD,T,C,H,W = vid0.shape

    # -- derived shapes --
    nH0 = (H-1)//stride0+1
    nW0 = (W-1)//stride0+1
    Q = T*nH0*nW0

    # -- settings from distance type --
    # dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    st = min(2*wt+1,T)
    base_shape = (B,HD,Q,st,ws,ws)
    inds = allocate_inds(base_shape,device)

    # -- compute indices --
    stnls_cuda.n3mm_fill_inds(inds,fflow,bflow,ws,wt)
    
    vid_index_neighbours(b,t,n1,n2,m1,m2,s,dev,exclude_self=True)

    # -- compute database --
    pat0 = create_patch_database(vid0,stride0,ps,reflect_bounds)
    pat1 = create_patch_database(vid1,stride1,ps,reflect_bounds)

    # -- forward --
    prods = matmult_fwd(pat0,pat1,inds)
    if dist_type == "prod":
        dists = prods
    else:
        dists = -2*prods
        pat1_norm = (xe**2).sum(dim=-1, keepdim=True)
        pat1_norm = pat1_norm.gather(dim=1, index=If[:,:,0:1]).view(b,m,o,1)
        pat0_norm = pat0_norm.sum(-1)
        dists += pat0_norm + pat1_norm

    return dists,inds

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Pytorch Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class N3MatMultSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, fflow, bflow,
                ws, wt, ps, k, nheads=1, batchsize=-1,
                dist_type="prod", stride0=4, stride1=1,
                dilation=1, pt=1, reflect_bounds=True, full_ws=False,
                anchor_self=False, remove_self=False,
                use_adj=True, off_H0=0, off_W0=0, off_H1=0, off_W1=0):

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

        # -- run, optionally batched, forward function --
        dists,inds = n3mm_fwd_main(vid0, vid1, fflow, bflow,
                                   ws, wt, ps, dist_type,
                                   stride0, stride1, dilation, pt,
                                   reflect_bounds, use_adj,
                                   off_H0, off_W0, off_H1, off_W1)


        # -- compress search region --
        dists=dists.view(B,HD,Q,-1)
        inds=inds.view(B,HD,Q,-1,3)

        # -- manage self dists --
        dists,inds = manage_self(dists,inds,anchor_self,
                                 remove_self,qshift,stride0,H,W)

        # -- topk --
        dists,inds = stnls.nn.topk(dists,inds,k,dim=3,anchor=anchor_self,
                                   descending=descending,unique=False)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        ctx.save_for_backward(inds,vid0,vid1)
        ctx.mark_non_differentiable(inds)
        ctx.vid_shape = vid0.shape
        ctx_vars = {"batchsize":batchsize,"ps":ps,"pt":pt,
                    "stride0":stride0,"stride1":stride1,                    
                    "dil":dilation,"reflect_bounds":reflect_bounds,
                    "use_adj":use_adj,"off_H0":off_H0,"off_W0":off_W0,
                    "off_H1":off_H1,"off_W1":off_W1,"dist_type_i":dist_type_i}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds_is_none):

        # -- unpacking --
        inds,vid0,vid1 = ctx.saved_tensors
        ps,pt = ctx.ps,ctx.pt
        stride0,stride1 = ctx.stride0,ctx.stride1
        dil,reflect_bounds = ctx.dil,ctx.reflect_bounds

        # -- compute database --
        pat0 = create_patch_database(vid0,stride0,ps,reflect_bounds)
        pat1 = create_patch_database(vid1,stride1,ps,reflect_bounds)

        # -- backward step --
        grad0,grad1 = matmult_bwd(pat0,pat1,inds,grad_dists)
    
        return grad0,grad1,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Pytorch Module
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class N3MatMultSearch(th.nn.Module):

    def __init__(self, ws, wt, ps, k, nheads=1,
                 dist_type="prod", stride0=4, stride1=1,
                 dilation=1, pt=1, reflect_bounds=True,
                 full_ws=True, anchor_self=False, remove_self=False,
                 use_adj=True,off_H0=0,off_W0=0,off_H1=0,off_W1=0):
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


    def forward(self, vid0, vid1, fflow, bflow, batchsize=-1):
        return N3MatMultSearchFunction.apply(vid0,vid1,fflow,bflow,
                                        self.ws,self.wt,self.ps,self.k,
                                        self.nheads,batchsize,
                                        self.dist_type,self.stride0,
                                        self.stride1,self.dilation,self.pt,
                                        self.reflect_bounds,self.full_ws,
                                        self.anchor_self,self.remove_self,
                                        self.use_adj,self.off_H0,self.off_W0,
                                        self.off_H1,self.off_W1)

    def flops(self,T,F,H,W):
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
           dist_type="prod", stride0=4, stride1=1,
           dilation=1, pt=1, reflect_bounds=True, full_ws=True,
           anchor_self=True, remove_self=False,
           use_adj=True, off_H0=0, off_W0=0, off_H1=0, off_W1=0):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = N3MatMultSearchFunction.apply
    return fxn(vid0,vid1,fflow,bflow,ws,wt,ps,k,
               nheads,batchsize,dist_type,
               stride0,stride1,dilation,pt,reflect_bounds,
               full_ws,anchor_self,remove_self,
               use_adj,off_H0,off_W0,off_H1,off_W1)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg):
    pairs = {"ws":-1,"wt":-1,"ps":7,"k":10,
             "nheads":1,"dist_type":"prod",
             "stride0":4, "stride1":1, "dilation":1, "pt":1,
             "reflect_bounds":True, "full_ws":True,
             "anchor_self":True, "remove_self":False,
             "use_adj":True,"off_H0":0,"off_W0":0,"off_H1":0,"off_W1":0}
    return extract_pairs(pairs,cfg)

def init(cfg):
    cfg = extract_config(cfg)
    search = N3MatMultSearch(cfg.ws, cfg.wt, cfg.ps, cfg.k, nheads=cfg.nheads,
                        dist_type=cfg.dist_type, stride0=cfg.stride0,
                        stride1=cfg.stride1, dilation=cfg.dilation, pt=cfg.pt,
                        reflect_bounds=cfg.reflect_bounds, full_ws=cfg.full_ws,
                        anchor_self=cfg.anchor_self, remove_self=cfg.remove_self,
                        use_adj=cfg.use_adj,off_H0=cfg.off_H0,off_W0=cfg.off_W0,
                        off_H1=cfg.off_H1,off_W1=cfg.off_W1)
    return search