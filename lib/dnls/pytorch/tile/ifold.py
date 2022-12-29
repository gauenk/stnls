"""

Fold but with an inset rectangle of any size

"""


# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


def allocate_patches(b,nq,k,ps,pt,c,device):
    patches = th.zeros((b,nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class ifold(th.autograd.Function):
    """
    [patches -> video] @ nlInds

    nlInds.shape = [NumQueries,K,3]
    patches.shape = [NumQueries,K,pt,c,ps,ps]
    """

    @staticmethod
    def forward(ctx, patches, vid, coords, qStart, stride, dilation, adj,
                only_full,reflect_bounds):
        top,left,btm,right = coords
        # print(top,left,btm,right,qStart)
        dnls_cuda.ifold_forward(vid, patches, top, left, btm, right,
                                qStart, stride, dilation, adj, only_full, reflect_bounds)
        # print("ifold [vid>0]: ",th.any(vid.abs()>0).item())
        # print(vid)
        ctx.coords = coords
        ctx.qStart = qStart
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.adj = adj
        ctx.only_full = only_full
        ctx.reflect_bounds = reflect_bounds
        ctx.qNum = patches.shape[1]
        ctx.pt = patches.shape[3]
        ctx.ps = patches.shape[6]
        return vid

    @staticmethod
    def backward(ctx, grad_vid):

        # -- unpack --
        grad_vid = grad_vid.contiguous()
        ps,pt = ctx.ps,ctx.pt
        dilation = ctx.dilation
        qStart = ctx.qStart
        stride = ctx.stride
        qNum = ctx.qNum
        adj = ctx.adj
        only_full = ctx.only_full
        reflect_bounds = ctx.reflect_bounds
        top,left,btm,right = ctx.coords

        # -- alloc --
        b,t,c,h,w  = grad_vid.shape
        npix,colors = t*h*w,c
        device = grad_vid.device
        grad_patches = allocate_patches(b,qNum,1,ps,pt,colors,device)

        # -- backward --
        dnls_cuda.ifold_backward(grad_vid,grad_patches,
                                 top, left, btm, right,
                                 qStart,stride,dilation,adj,
                                 only_full, reflect_bounds)
        return grad_patches,None,None,None,None,None,None,None,None

class iFold(th.nn.Module):
    # [patches -> video] @ nlInds [with k == 1]

    def __init__(self,vid_shape,coords,stride=1,dilation=1,adj=0,
                 only_full=False,reflect_bounds=True,device="cuda"):
        super(iFold, self).__init__()
        self.vshape = vid_shape
        self.vid_shape = vid_shape
        self.vid = self.allocate_vid(vid_shape,device)
        self.stride = stride
        self.dilation = dilation
        self.coords = coords
        self.adj = adj
        self.only_full = only_full
        self.reflect_bounds = reflect_bounds
        if self.coords is None:
            b,t,c,h,w = vid_shape
            self.coords = [0,0,h,w]

    def allocate_vid(self,vid_shape,device):
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        return vid

    def forward(self, patches, qStart):
        ps = patches.shape[-1]
        bpatches,qStart = patches,qStart
        vid = self.allocate_vid(self.vid_shape,patches.device)
        vid = ifold.apply(bpatches, vid, self.coords, qStart,
                          self.stride,self.dilation,self.adj,
                          self.only_full,self.reflect_bounds)
        self.vid = self.vid + vid
        return self.vid

