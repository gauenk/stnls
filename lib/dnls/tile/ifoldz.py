"""

Fold"z" but with an inset rectangle of any size

"""


# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


def allocate_patches(nq,k,ps,pt,c,device):
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class ifoldz(th.autograd.Function):
    """
    [patches -> video] @ nlInds

    nlInds.shape = [NumQueries,K,3]
    patches.shape = [NumQueries,K,pt,c,ps,ps]
    """

    @staticmethod
    def forward(ctx, patches, vid, zvid, coords, qStart, stride, dilation, adj,
                only_full,use_reflect):
        top,left,btm,right = coords
        patches = patches.contiguous()
        print(vid.shape,zvid.shape,patches.shape)
        dnls_cuda.ifoldz_forward(vid, zvid, patches, top, left, btm, right,
                                 qStart, stride, dilation, adj, only_full, use_reflect)
        ctx.coords = coords
        ctx.qStart = qStart
        ctx.stride = stride
        ctx.qNum = patches.shape[0]
        ctx.dilation = dilation
        ctx.adj = adj
        ctx.only_full = only_full
        ctx.use_reflect = use_reflect
        ctx.pt = patches.shape[2]
        ctx.ps = patches.shape[5]
        return vid,zvid

    @staticmethod
    def backward(ctx, grad_vid, grad_zvid_is_none):

        # -- unpack --
        grad_vid = grad_vid.contiguous()
        ps,pt = ctx.ps,ctx.pt
        dilation = ctx.dilation
        qStart = ctx.qStart
        stride = ctx.stride
        qNum = ctx.qNum
        adj = ctx.adj
        only_full = ctx.only_full
        use_reflect = ctx.use_reflect
        top,left,btm,right = ctx.coords

        # -- alloc --
        t,c,h,w  = grad_vid.shape
        npix = t*h*w
        colors = grad_vid.shape[1]
        device = grad_vid.device
        grad_patches = allocate_patches(qNum,1,ps,pt,colors,device)

        # -- backward --
        dnls_cuda.ifold_backward(grad_vid,grad_patches,
                                 top, left, btm, right,
                                 qStart,stride,dilation,adj,
                                 only_full, use_reflect)
        return grad_patches,None,None,None,None,None,None,None,None,None

class iFoldz(th.nn.Module):
    # [patches -> video] @ nlInds [with k == 1]

    def __init__(self,vid_shape,coords,stride=1,dilation=1,adj=0,
                 only_full=False,use_reflect=True,device="cuda"):
        super(iFoldz, self).__init__()
        self.vshape = vid_shape
        self.vid_shape = vid_shape
        self.vid = self.allocate_vid(vid_shape,device)
        self.zvid = self.allocate_vid(vid_shape,device)
        self.stride = stride
        self.dilation = dilation
        self.coords = coords
        self.adj = adj
        self.only_full = only_full
        self.use_reflect = use_reflect
        if self.coords is None:
            t,c,h,w = vid_shape
            self.coords = [0,0,h,w]

    def allocate_vid(self,vid_shape,device):
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        return vid

    def forward(self, patches, qStart):
        ps = patches.shape[-1]
        bpatches,qStart = patches,qStart
        vid = self.allocate_vid(self.vid_shape,patches.device)
        zvid = self.allocate_vid(self.vid_shape,patches.device)
        vid,zvid = ifoldz.apply(bpatches, vid, zvid, self.coords, qStart,
                           self.stride,self.dilation,self.adj,
                           self.only_full,self.use_reflect)
        self.vid = self.vid + vid
        self.zvid = self.zvid + zvid
        return self.vid,self.zvid

