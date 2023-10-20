
# -- python --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- cpp cuda kernel --
import stnls_cuda

# -- package --
import stnls

# -- api --
from .utils import extract_pairs

# -- local --
from .utils import shape_vids,allocate_inds,dist_type_select,allocate_vid
from .utils import descending_menu
from .shared import manage_self,run_fold
# from .nls_bwd_impl import nls_backward
# from .batching_utils import run_batched,batching_info
# from .n3mm_utils import IndexedMatmul1Efficient
from .n3mm_utils import matmult_fwd,matmult_bwd,raster_indices,vid2patches

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Forward Logic
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def n3mm_fwd_main(vid0, vid1, fflow, bflow,
                  nheads, ws, wt, ps, dist_type,
                  stride0, stride1, dilation, pt,
                  reflect_bounds, use_adj):

    # -- unpack --
    device = vid0.device
    B,T,C,H,W = vid0.shape

    # -- derived shapes --
    nH0 = (H-1)//stride0+1
    nW0 = (W-1)//stride0+1
    Q = T*nH0*nW0

    # -- settings from distance type --
    # dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- compute indices --
    inds = stnls.nn.non_local_inds(fflow,bflow,ws,wt,stride0,stride1).int()

    # -- boundary --
    # inds = th.where(inds<0,-inds,inds)
    # inds[...,1] = th.where(inds[...,1]>=H,2*(H-1)-inds[...,1],inds[...,1])
    # inds[...,2] = th.where(inds[...,2]>=W,2*(W-1)-inds[...,2],inds[...,2])
    assert th.all(inds>=0)
    assert th.all(inds[...,1]<H)
    assert th.all(inds[...,2]<W)

    # -- prepare shaping --
    assert inds.shape[1] == Q
    inds = inds.view(B,Q,-1,3)
    inds = repeat(inds,'b q l tr -> (b HD) q l tr',HD=nheads)

    # -- create patch database --
    pat0 = vid2patches(vid0,nheads,stride0,ps,dilation,reflect_bounds)
    pat1 = vid2patches(vid1,nheads,stride1,ps,dilation,reflect_bounds)

    # -- forward --
    inds_r = raster_indices(inds,H,W,stride1)
    prods = matmult_fwd(pat1,pat0,inds_r)

    # th.cuda.synchronize()
    if dist_type == "prod":
        dists = prods
    else:
        b,n,e = pat1.shape
        b,m,o = inds_r.shape
        If = inds_r.view(b, m*o,1).expand(b,m*o,e)
        pat1_norm = (pat1**2).sum(dim=-1, keepdim=True)
        pat1_norm = pat1_norm.gather(dim=1, index=If[:,:,0:1]).view(b,m,o,1)
        pat0_norm = (pat0**2).sum(-1,keepdim=True)[...,None]
        dists = pat0_norm + pat1_norm- 2*prods.unsqueeze(3)

    # -- reshape with heads --
    dists = dists.view(B,nheads,Q,-1)
    inds = inds.view(B,nheads,Q,-1,3)

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
                dilation=1, pt=1, reflect_bounds=True,
                self_action=None, use_adj=False, normalize_bwd=True):

        """
        Run the non-local search

        vid0 = [B,T,C,H,W] or [B,HD,T,C,H,W]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """

        # -- reshape with heads --
        dtype = vid0.dtype
        device = vid0.device
        # vid0,vid1 = shape_vids(nheads,[vid0,vid1])
        B,T,F,H,W = vid0.shape
        HD = nheads

        # -- run, optionally batched, forward function --
        dists,inds = n3mm_fwd_main(vid0, vid1, fflow, bflow,
                                   nheads, ws, wt, ps, dist_type,
                                   stride0, stride1, dilation, pt,
                                   reflect_bounds, use_adj)


        # -- compress search region --
        B,HD,Q,*_ = dists.shape
        nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
        dists=dists.view(B,HD,T,nH,nW,-1)
        inds=inds.view(B,HD,T,nH,nW,-1,3)

        # -- manage self dists --
        qshift = 0
        anchor_self = not(self_action is None) and "anchor" in self_action
        remove_self = not(self_action is None) and "remove" in self_action
        dists,inds = manage_self(dists,inds,anchor_self,
                                 remove_self,qshift,stride0,H,W)
        # -- topk --
        descending = descending_menu(dist_type)
        dists,inds = stnls.nn.topk(dists,inds,k,dim=3,anchor=anchor_self,
                                   descending=descending,unique=False)

        # -- setup ctx --
        dist_type_i = dist_type_select(dist_type)[0]
        ctx.save_for_backward(inds,vid0,vid1)
        ctx.mark_non_differentiable(inds)
        ctx_vars = {"batchsize":batchsize,"ps":ps,"pt":pt,
                    "dist_type":dist_type,"nheads":nheads,
                    "stride0":stride0,"stride1":stride1,
                    "dil":dilation,"reflect_bounds":reflect_bounds,
                    "use_adj":use_adj,"dist_type_i":dist_type_i,
                    "normalize_bwd":normalize_bwd}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- return --
        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds_is_none):

        # -- unpacking --
        inds,vid0,vid1 = ctx.saved_tensors
        ps,pt = ctx.ps,ctx.pt
        nheads = ctx.nheads
        stride0,stride1 = ctx.stride0,ctx.stride1
        dilation,reflect_bounds = ctx.dil,ctx.reflect_bounds

        # -- compute database --
        pat0 = vid2patches(vid0,nheads,stride0,ps,dilation,reflect_bounds)
        pat1 = vid2patches(vid1,nheads,stride1,ps,dilation,reflect_bounds)

        # -- backward step --
        H,W = vid0.shape[-2:]
        inds = rearrange(inds,'b hd t nh nw l tr -> (b hd) (t nh nw) l tr')
        grad_dists = rearrange(grad_dists,'b hd t nh nw l -> (b hd) (t nh nw) l')
        inds_r = raster_indices(inds,H,W,stride1).contiguous()
        BHD = pat1.shape[0]
        pgrad1,pgrad0 = [],[]
        for b in range(BHD):
            pgrad1_,pgrad0_ = matmult_bwd(pat1[[b]],pat0[[b]],inds_r[[b]],grad_dists[[b]])
            pgrad1.append(pgrad1_)
            pgrad0.append(pgrad0_)
        pgrad1 = th.cat(pgrad1)
        pgrad0 = th.cat(pgrad0)

        # -- reshape --
        B,T = vid0.shape[:2]
        shape_str = '(b hd) (t l) (c ph pw) -> (b t) (hd c ph pw) l'
        pgrad0 = rearrange(pgrad0,shape_str,b=B,t=T,ph=ps,pw=ps)
        pgrad1 = rearrange(pgrad1,shape_str,b=B,t=T,ph=ps,pw=ps)

        # -- fold into video --
        grad0,grad0z = run_fold(pgrad0,H,W,ps,stride0,dilation)
        if ctx.normalize_bwd: grad0 = grad0/grad0z
        grad0 = rearrange(grad0,'(b t) c h w -> b t c h w',b=B)

        grad1,grad1z = run_fold(pgrad1,H,W,ps,stride1,dilation)
        if ctx.normalize_bwd: grad1 = grad1/grad1z
        grad1 = rearrange(grad1,'(b t) c h w -> b t c h w',b=B)

        return grad0,grad1,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,\
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
                 self_action=None, use_adj=False, normalize_bwd=True):
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
        self.normalize_bwd = normalize_bwd

        # -- manage patch and search boundaries --
        self.reflect_bounds = reflect_bounds
        self.use_adj = use_adj

        # -- special mods to "self" search --
        self.self_action = self_action


    def forward(self, vid0, vid1, fflow, bflow, batchsize=-1):
        return N3MatMultSearchFunction.apply(vid0,vid1,fflow,bflow,
                                             self.ws,self.wt,self.ps,self.k,
                                             self.nheads,batchsize,
                                             self.dist_type,self.stride0,
                                             self.stride1,self.dilation,self.pt,
                                             self.reflect_bounds,
                                             self.self_action,self.use_adj,
                                             self.normalize_bwd)

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
#            [Functional API]  stnls.search.n3mm_search(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid0, vid1, fflow, bflow,
           ws, wt, ps, k, nheads=1, batchsize=-1,
           dist_type="prod", stride0=4, stride1=1,
           dilation=1, pt=1, reflect_bounds=True,
           self_action=None, use_adj=False, normalize_bwd=True):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = N3MatMultSearchFunction.apply
    return fxn(vid0,vid1,fflow,bflow,ws,wt,ps,k,
               nheads,batchsize,dist_type,
               stride0,stride1,dilation,pt,reflect_bounds,
               self_action,use_adj,normalize_bwd)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def extract_config(cfg,restrict=True):
    pairs = {"ws":-1,"wt":-1,"ps":7,"k":10,
             "nheads":1,"dist_type":"prod",
             "stride0":4, "stride1":1, "dilation":1, "pt":1,
             "reflect_bounds":True,"self_action":None,
             "use_adj":False,"normalize_bwd":True}
    return extract_pairs(cfg,pairs,restrict=restrict)

def init(cfg):
    cfg = extract_config(cfg)
    search = N3MatMultSearch(cfg.ws, cfg.wt, cfg.ps, cfg.k, nheads=cfg.nheads,
                             dist_type=cfg.dist_type, stride0=cfg.stride0,
                             stride1=cfg.stride1, dilation=cfg.dilation, pt=cfg.pt,
                             reflect_bounds=cfg.reflect_bounds,
                             self_action=cfg.self_action,use_adj=cfg.use_adj,
                             normalize_bwd=cfg.normalize_bwd)
    return search
