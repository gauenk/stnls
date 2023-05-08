
# -- python --
import torch as th

# -- cpp cuda kernel --
import stnls_cuda


def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(inds,ps,pt,c):
    device = inds.device
    b,nq,k = inds.shape[:3]
    patches = th.zeros((b,nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches


class unfold_k(th.autograd.Function):
    # [video -> patches] @ inds

    @staticmethod
    def forward(ctx, vid, inds, ps, pt=1, dilation=1,
                btype="default", exact=False,
                use_adj = False, reflect_bounds=True,
                use_atomic=True):
        """
        vid = [B,T,C,H,W]
        inds = [B,NumQueries,K,3]
        ps = patchsize
        pt = patchsize_time (forward only)
        patches = [B,Q,K,pt,C,ps,ps]

        """

        # -- init --
        patches = allocate_patches(inds,ps,pt,vid.shape[-3])
        inds = inds.contiguous()
        adj = ps//2 if use_adj else 0

        # -- fwd --
        stnls_cuda.unfoldk_forward(vid, patches, inds, dilation, adj, reflect_bounds)

        # -- save --
        ctx.save_for_backward(inds)
        ctx.ps,ctx.pt = ps,pt
        ctx.vid_shape = vid.shape
        ctx.dilation = dilation
        ctx.exact = exact
        ctx.btype = btype
        ctx.use_adj = use_adj
        ctx.reflect_bounds = reflect_bounds
        ctx.use_atomic = use_atomic
        return patches

    @staticmethod
    def backward(ctx, grad_patches):
        inds = ctx.saved_tensors[0]
        ps,pt = ctx.ps,ctx.pt
        vid_shape = ctx.vid_shape
        dilation = ctx.dilation
        exact = ctx.exact
        btype = ctx.btype
        use_adj = ctx.use_adj
        use_atomic = ctx.use_atomic
        reflect_bounds = ctx.reflect_bounds
        grad_vid = allocate_vid(vid_shape,grad_patches.device)
        grad_patches = grad_patches.contiguous()
        adj = ps//2 if use_adj else 0
        if btype in "default" or btype in "simple":
            stnls_cuda.unfoldk_backward(grad_vid,grad_patches,inds,
                                        dilation,exact,adj,reflect_bounds,
                                        use_atomic)
        elif btype in "efficient":
            raise NotImplementedError("")
            # stnls_cuda.unfoldk_backward_eff(grad_vid,grad_patches,inds,
            #                                dilation,exact,adj,reflect_bounds)
        else:
            raise ValueError(f"Uknown backward type for unfoldk [{btype}]")
        return grad_vid,None,None,None,None,None,None,None,None,None

class UnfoldK(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, pt=1, dilation=1, btype="default", exact=False,
                 use_adj=False, reflect_bounds = True, use_atomic=True):
        super().__init__()
        self.ps = ps
        self.pt = pt
        self.dilation = dilation
        self.exact = exact
        self.btype = btype
        self.use_adj = use_adj
        self.reflect_bounds = reflect_bounds
        self.use_atomic = use_atomic
        # self.device = device

    def forward(self, vid, inds):
        return unfold_k.apply(vid,inds,self.ps,self.pt,
                              self.dilation,self.btype,self.exact,
                              self.use_adj, self.reflect_bounds,
                              self.use_atomic)

def _apply(vid,inds,ps,pt=1,dilation=1,btype="default",
           exact=False,use_adj=False,reflect_bounds=True,
           use_atomic=True):
    return unfold_k.apply(vid,inds,ps,pt,dilation,btype,exact,
                          use_adj, reflect_bounds,use_atomic)

