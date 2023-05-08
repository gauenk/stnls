"""

Fold"z" but with an inset rectangle of any size

"""


# -- python --
import torch as th

# -- cpp cuda kernel --
import stnls_cuda


def allocate_patches(b,nq,k,ps,pt,c,device):
    patches = th.zeros((b,nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class ifoldz(th.autograd.Function):
    """
    [patches -> video]

    patches.shape = [B,NumQueries,1,pt,c,ps,ps]
    vid.shape = [B,T,C,H,W]


    """

    @staticmethod
    def forward(ctx, patches, vid, zvid, coords, qstart, stride,
                dilation, use_adj, only_full,reflect_bounds):
        top,left,btm,right = coords
        patches = patches.contiguous()
        assert patches.shape[2] == 1,"Must be k == 1"
        # print("only_full: ",only_full)
        # print("vid.shape: ",vid.shape)
        # print("patches.shape:" ,patches.shape,vid.shape)
        ps = patches.shape[-1]
        adj = ps//2 if use_adj else 0
        stnls_cuda.ifoldz_forward(vid, zvid, patches,
                                  top, left, btm, right,
                                  qstart, stride, dilation,
                                  adj, only_full, reflect_bounds)
        ctx.coords = coords
        ctx.qstart = qstart
        ctx.stride = stride
        ctx.qNum = patches.shape[1]
        ctx.dilation = dilation
        ctx.use_adj = use_adj
        ctx.only_full = only_full
        ctx.reflect_bounds = reflect_bounds
        ctx.pt = patches.shape[3]
        ctx.ps = patches.shape[6]
        return vid,zvid

    @staticmethod
    def backward(ctx, grad_vid, grad_zvid_is_none):

        # -- unpack --
        grad_vid = grad_vid.contiguous()
        ps,pt = ctx.ps,ctx.pt
        dilation = ctx.dilation
        qstart = ctx.qstart
        stride = ctx.stride
        qNum = ctx.qNum
        use_adj = ctx.use_adj
        only_full = ctx.only_full
        reflect_bounds = ctx.reflect_bounds
        top,left,btm,right = ctx.coords

        # -- alloc --
        b,t,c,h,w  = grad_vid.shape
        npix = t*h*w
        colors = grad_vid.shape[2]
        device = grad_vid.device
        grad_patches = allocate_patches(b,qNum,1,ps,pt,colors,device)

        # -- backward --
        adj = ps//2 if use_adj else 0
        stnls_cuda.ifold_backward(grad_vid,grad_patches,
                                 top, left, btm, right,
                                 qstart,stride,dilation,adj,
                                 only_full, reflect_bounds)

        return grad_patches,None,None,None,None,None,None,None,None,None

class iFoldz(th.nn.Module):
    # [patches -> video] @ nlInds [with k == 1]

    def __init__(self,vid_shape,coords=None,stride=1,dilation=1,use_adj=False,
                 only_full=False,reflect_bounds=True,device="cuda"):
        super(iFoldz, self).__init__()
        self.vshape = vid_shape
        self.vid_shape = vid_shape
        self.vid = self.allocate_vid(vid_shape,device)
        self.zvid = self.allocate_vid(vid_shape,device)
        self.stride = stride
        self.dilation = dilation
        self.coords = coords
        self.use_adj = use_adj
        self.only_full = only_full
        self.reflect_bounds = reflect_bounds
        if self.coords is None:
            b,t,c,h,w = vid_shape
            self.coords = [0,0,h,w]

    def allocate_vid(self,vid_shape,device):
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        return vid

    def forward(self, patches, qstart=0):
        # -- set device --
        self.vid = self.vid.to(patches.device)
        self.zvid = self.zvid.to(patches.device)

        # -- forward --
        ps = patches.shape[-1]
        bpatches,qstart = patches,qstart
        vid = self.allocate_vid(self.vid_shape,patches.device)
        zvid = self.allocate_vid(self.vid_shape,patches.device)
        vid,zvid = ifoldz.apply(bpatches, vid, zvid, self.coords, qstart,
                           self.stride,self.dilation,self.use_adj,
                           self.only_full,self.reflect_bounds)
        self.vid = self.vid + vid
        self.zvid = self.zvid + zvid
        return self.vid,self.zvid

