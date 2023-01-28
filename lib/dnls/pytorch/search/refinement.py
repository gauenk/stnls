import torch as th

class RefineSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx,vid0,vid1,qshift=0,Q=-1):
        # -- reshape with heads --
        dtype = vid0.dtype
        device = vid0.device
        assert vid0.ndim in [5], "Must be 5 dims."
        if vid0.ndim == 5:
            # c = vid0.shape[-1]
            c = vid0.shape[2]
            assert c % nheads == 0,"must be multiple of each other."
            # shape_str = 'b t h w (H c) -> b H t h w c'
            shape_str = 'b t (H c) h w -> b H t c h w'
            vid0 = rearrange(vid0,shape_str,H=nheads).contiguous()
            vid1 = rearrange(vid1,shape_str,H=nheads).contiguous()
        assert vid0.shape[1] == nheads
        assert vid1.shape[1] == nheads
        # vid0 = vid0.contiguous()
        # vid1 = vid1.contiguous()
        # B,H,t,h,w,c = vid0.shape
        B,H,t,c,h,w = vid0.shape
        vshape = (t,c,h,w)
        n_h0,n_w0 = get_num_img(vshape,stride0,ps,dilation)
        _,_,Q,K_exh = qinds.shape[:4] # B H Q K_exh 3

        # -- allocs --
        BHQ = B*H*Q
        dists_exh = -th.inf*th.ones((B,H,Q,K_exh,ws_h,ws_w),device=device,dtype=dtype)
        inds_exh = -th.ones((B,H,Q,K_exh,ws_h,ws_w,3),device=device,dtype=th.int32)

        # -- allocates self --
        assert use_self == anchor_self
        if anchor_self:
            self_dists = -th.inf * th.ones((B,H,Q),device=device,dtype=dtype)
        else:
            self_dists = -th.inf * th.ones((1,1,1),device=device,dtype=dtype)

        # -- viz --
        # print("vid0.shape: " ,vid0.shape)
        # print("vid1.shape: " ,vid1.shape)
        # print("dists_exh.shape: " ,dists_exh.shape)
        # print("inds_exh.shape: " ,inds_exh.shape)
        # print("qinds.shape: " ,qinds.shape)
        # print(qstart,stride0,stride1,ps,pt,ws_h,ws_w)
        # print(n_h0,n_w0,h0_off, w0_off, h1_off, w1_off)
        # print(chnls,dilation,use_adj,reflect_bounds)
        # print(search_abs, full_ws, anchor_self, use_self)

        # -- setup flows --
        gpuid = th.cuda.current_device()

        # -- forward --
        th.cuda.set_device(device)
        dnls_cuda.prod_refine_forward(vid0, vid1, dists_exh, inds_exh,
                                      self_dists, qinds,
                                      qstart, stride0, n_h0, n_w0,
                                      h0_off, w0_off, h1_off, w1_off,
                                      ps, pt, ws_h, ws_w, ws_h_og, ws_w_og,
                                      chnls, dilation, stride1, use_adj,
                                      reflect_bounds, search_abs, full_ws,
                                      anchor_self, use_self)
        th.cuda.synchronize()


        # -- shape for next step --
        B,H,Q = dists_exh.shape[:3]
        dists_exh=dists_exh.view(B*H,Q,-1)#.contiguous()
        inds_exh=inds_exh.view(B*H,Q,-1,3)#.contiguous()
        # print(qinds[0,0,:10,0])
        # print("inds exh")
        # for i in range(inds_exh.shape[2]):
        #     print(i,inds_exh[0,0,i])
        # print(inds_exh[0,1,:5])

        # -- remove self --
        if remove_self:
            dists_exh,inds_exh = run_remove_self_cuda(dists_exh,inds_exh,qstart,
                                                      stride0,n_h0,n_w0)

        # -- topk [with uniques] --
        assert use_k is True,"Must topk to efficiently remove duplicates"
        if use_k:

            if anchor_self:
                dnls.nn.anchor_self(dists,inds,qstart,stride0,H,W)
            dists_k,inds_k = topk(dists,inds,k,dim=3,anchor=anchor_self,
                                  descending=True,unique=True)
            # # print("inds_exh.shape: ",inds_exh.shape)
            # K_exh = inds_exh.shape[1]
            # dists,inds = allocate_rtn(B*H*Q,K_exh,device,dtype)

            # # -- sort --
            # topk_with_anchor(dists_exh,inds_exh,dists,inds,self_dists,anchor_self)
            # # get_topk_prod(dists_exh,inds_exh,dists,inds)

            # # -- only unique --
            # dists,inds = only_unique(dists,inds,k)

        else:
            dists,inds = dists_exh,inds_exh

        # -- fill nans --
        args = th.where(th.isnan(dists))
        dists[args] = -th.inf # fix nan

        # -- final shape with heads -
        dists = dists.view(B,H,Q,-1)
        inds = inds.view(B,H,Q,-1,3)
        # print("inds.shape: ",inds.shape)

        # -- for backward --
        ctx.save_for_backward(inds,vid0,vid1)
        ctx.vid_shape = vid0.shape
        ctx.nheads = nheads
        ctx.qstart,ctx.stride0 = qstart,stride0
        ctx.ps,ctx.pt,ctx.dil = ps,pt,dilation
        ctx.reflect_bounds = reflect_bounds
        ctx.rbwd,ctx.exact = rbwd,exact
        ctx.use_adj,ctx.nbwd = use_adj,nbwd
        ctx.n_h0,ctx.n_w0 = n_h0,n_w0
        ctx.h0_off,ctx.w0_off = h0_off, w0_off
        ctx.h1_off,ctx.w1_off = h1_off, w1_off
        return dists,inds


    @staticmethod
    def backward(ctx):
        pass


class RefineSearch(th.nn.Module):

    def __init__(self, ws, ps, k, nheads,
                 dist_type="prod", stride0=4, stride1=1, dilation=1, pt=1,
                 reflect_bounds=True, full_ws = False,
                 anchor_self=False, remove_self=False,
                 use_adj=True,h0_off=0,w0_off=0,h1_off=0,w1_off=0,
                 rbwd=True, nbwd=1, exact=False):
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

    def forward(self,vid0,vid1,qshift=0,nqueries=-1):
        return RefineSearchFunction.apply(vid0,vid1,fflow,bflow,
                                          self.ws,self.ps,self.k,
                                          self.nheads,qshift,nqueries,
                                          self.dist_type,self.stride0,self.stride1,
                                          self.dilation,self.pt,
                                          self.reflect_bounds,self.full_ws,
                                          self.anchor_self,self.remove_self,
                                          self.use_adj,self.h0_off,self.w0_off,
                                          self.h1_off,self.w1_off,
                                          self.rbwd,self.nbwd,self.exact)

_apply = RefineSearchFunction.apply # api

