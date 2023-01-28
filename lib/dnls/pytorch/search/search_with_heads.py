
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- softmax --
import torch.nn.functional as nnf

# -- cpp cuda kernel --
import dnls_cuda

# -- package --
import dnls

# -- local --
from .search_utils import *

def dist_menu(dist_type):
    menu = {"prod":0,"l2":1}
    return menu[dist_type]

def descending_menu(dist_type):
    menu = {"prod":True,"l2":False}
    return menu[dist_type]

def init_dist_val_menu(dist_type):
    menu = {"prod":-np.inf,"l2":np.inf}
    return menu[dist_type]

def allocate_pair(base_shape,device,dtype,idist_val):
    dists = idist_val * th.ones(base_shape,device=device,dtype=dtype)
    inds =  -th.ones(base_shape+(3,),device=device,dtype=th.int32)
    return dists,inds

def shape_vids(nheads,*vids):
    _vids = []
    for vid in vids:
        # -- reshape with heads --
        dtype = vid.dtype
        device = vid.device
        assert vid.ndim in [5], "Must be 5 dims."
        if vid.ndim == 5:
            c = vid.shape[2]
            assert c % nheads == 0,"must be multiple of each other."
            shape_str = 'b t (HD c) h w -> b HD t c h w'
            vid = rearrange(vid,shape_str,HD=nheads).contiguous()
        assert vid.shape[1] == nheads
        _vids.append(vid)
    return _vids

class NonLocalSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, fflow, bflow,
                ws, wt, ps, k, nheads=1, qshift=0, Q=-1,
                dist_type="prod", stride0=4, stride1=1,
                dilation=1, pt=1, reflect_bounds=True, full_ws=False,
                anchor_self=False,remove_self=False,
                use_adj=True,h0_off=0, w0_off=0, h1_off=0, w1_off=0,
                rbwd=True,nbwd=1,exact=False):

        """
        vid0 = [B,T,C,H,W] or [B,H,T,C,H,W]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """

        # -- reshape with heads --
        device = vid0.device
        vid0,vid1 = shape_vids(nheads,vid0,vid1)
        B,HD,T,F,H,W = vid0.shape

        # -- derived shapes --
        nH0 = (H-1)//stride0+1
        nW0 = (W-1)//stride0+1
        Q = T*nH0*nW0 if Q <= 0 else Q

        # -- search space --
        ws_h,ws_w = ws,ws
        search_abs = ws == -1
        if search_abs:
            ws_h,ws_w = nH0,nW0

        # -- optionally set device --
        # th.cuda.set_device(device)

        # -- settings from distance type --
        dist_type_i = dist_menu(dist_type)
        descending = descending_menu(dist_type)
        idist_val = init_dist_val_menu(dist_type)

        # -- allocate results --
        st = min(2*wt+1,T)
        base_shape = (B,HD,Q,st,ws_h,ws_w)
        dists_exh,inds_exh = allocate_pair(base_shape,device,vid0.dtype,idist_val)

        # -- forward --
        dnls_cuda.search_with_heads_forward(vid0, vid1, fflow, bflow,
                                            dists_exh, inds_exh,
                                            qshift, stride0, nH0, nW0,
                                            h0_off, w0_off, h1_off, w1_off,
                                            ps, pt, ws_h, ws_w,
                                            wt, dilation, stride1,
                                            use_adj, reflect_bounds,
                                            search_abs, full_ws, dist_type_i)

        # -- compress search region --
        dists_exh=dists_exh.view(B,HD,Q,-1)
        inds_exh=inds_exh.view(B,HD,Q,-1,3)

        # -- manage self dists --
        assert not(remove_self and anchor_self)
        if remove_self:
            dists_exh,inds_exh = run_remove_self_cuda(dists_exh,inds_exh,qshift,
                                                      stride0,nH0,nW0)
        if anchor_self:
            dnls.nn.anchor_self(dists_exh,inds_exh,stride0,H,W)

        # -- topk --
        if k > 0:
            dists,inds = dnls.nn.topk(dists_exh,inds_exh,k,dim=3,anchor=anchor_self,
                                      descending=descending,unique=False)
        else:
            dists,inds = dists_exh,inds_exh

        # -- contiguous --
        # dists = dists.contiguous()
        # inds = inds.contiguous()

        # -- for backward --
        ctx.save_for_backward(inds,vid0,vid1)
        ctx.vid_shape = vid0.shape
        ctx_vars = {"qshift":qshift,"stride0":stride0,"ps":ps,"pt":pt,
                    "dil":dilation,"reflect_bounds":reflect_bounds,
                    "rbwd":rbwd,"exact":exact,"nbwd":nbwd,
                    "use_adj":use_adj,"h0_off":h0_off,"w0_off":w0_off,
                    "h1_off":h1_off,"w1_off":w1_off,"dist_type_i":dist_type_i}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        return dists,inds

    @staticmethod
    def backward(ctx, grad_dists, grad_inds_is_none):

        # -- populate names --
        inds,vid0,vid1 = ctx.saved_tensors

        # -- allocate grads --
        grad_vid0 = allocate_vid(ctx.vid_shape,grad_dists.device)
        grad_vid1 = allocate_vid(ctx.vid_shape,grad_dists.device)

        # -- ensure contiguous --
        grad_dists = grad_dists.contiguous()
        inds = inds.contiguous()

        # -- derived shapes --
        H,W = ctx.vid_shape[-2:]
        nH0 = (H-1)//ctx.stride0+1
        nW0 = (W-1)//ctx.stride0+1

        # -- allow for repeated exec --
        bwd_fxn = dnls_cuda.search_with_heads_backward
        if ctx.nbwd == 1:
            bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,
                    grad_dists,inds,ctx.qshift,ctx.stride0,nH0,nW0,
                    ctx.h0_off, ctx.w0_off,ctx.h1_off, ctx.w1_off,
                    ctx.ps,ctx.pt,ctx.dil,ctx.use_adj,
                    ctx.reflect_bounds,ctx.rbwd,ctx.exact,ctx.dist_type_i)
        else:
            for _ in range(ctx.nbwd):
                grad_vid0_i = allocate_vid(vid_shape,grad_dists.device)
                grad_vid1_i = allocate_vid(vid_shape,grad_dists.device)
                bwd_fxn(grad_vid0_i,grad_vid1_i,vid0,vid1,
                        grad_dists,inds,ctx.qshift,ctx.stride0,nH0,nW0,
                        ctx.h0_off, ctx.w0_off,ctx.h1_off, ctx.w1_off,
                        ctx.ps,ctx.pt,ctx.dil,ctx.use_adj,
                        ctx.reflect_bounds,ctx.rbwd,ctx.exact,ctx.dist_type_i)
                grad_vid0 += grad_vid0_i
                grad_vid1 += grad_vid1_i
            grad_vid0 /= ctx.nbwd
            grad_vid1 /= ctx.nbwd

        # -- finalize shape --
        grad_vid0 = rearrange(grad_vid0,'B H t c h w -> B t (H c) h w')
        grad_vid1 = rearrange(grad_vid1,'B H t c h w -> B t (H c) h w')

        return grad_vid0,grad_vid1,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None,\
            None,None,None,None,None,None,None,None,None,None,None

nls = NonLocalSearchFunction.apply # api

class NonLocalSearch(th.nn.Module):


    def __init__(self, ws, wt, ps, k, nheads,
                 dist_type="prod", stride0=4, stride1=1, dilation=1, pt=1,
                 reflect_bounds=True, full_ws = False,
                 anchor_self=False, remove_self=False,
                 use_adj=True,h0_off=0,w0_off=0,h1_off=0,w1_off=0,
                 rbwd=True, nbwd=1, exact=False):
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
        self.h0_off = h0_off
        self.w0_off = w0_off
        self.h1_off = h1_off
        self.w1_off = w1_off

        # -- backprop params --
        self.nbwd = nbwd
        self.exact = exact
        self.rbwd = rbwd

    def empty_flow(self,vshape,dtype,device):
        b,t,c,h,w = vshape
        zflow = th.zeros((b,t,2,h,w),dtype=dtype,device=device)
        return zflow

    def forward(self, vid0, vid1, fflow, bflow, qshift=0, nqueries=-1):
        return NonLocalSearchFunction.apply(vid0,vid1,fflow,bflow,
                                            self.ws,self.wt,self.ps,self.k,
                                            self.nheads,qshift,nqueries,
                                            self.dist_type,self.stride0,self.stride1,
                                            self.dilation,self.pt,
                                            self.reflect_bounds,self.full_ws,
                                            self.anchor_self,self.remove_self,
                                            self.use_adj,self.h0_off,self.w0_off,
                                            self.h1_off,self.w1_off,
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
