"""

Fold"z" but with an inset rectangle of any size

"""


# -- python --
import torch as th

# -- cpp cuda kernel --
import stnls_cuda


def allocate_patches(b,nq,ps,pt,c,device):
    patches = th.zeros((b,nq,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class nlfold(th.autograd.Function):
    """
    [patches -> video]

    patches.shape = [BatchSize,NumQueries,Patch Size Time, Num Ftrs, PatchSize, PatchSize]
     = (B,Q,pt,F,ps,ps)
    vid.shape = [B,T,C,H,W]


    """

    @staticmethod
    def forward(ctx, patches, vid_shape, stride,
                dilation, use_adj, reflect_bounds):

        # -- inputs --
        qstart = 0
        patches = patches.contiguous()
        # assert patches.shape[2] == 1,"Must be k == 1"
        ps = patches.shape[-1]

        # -- allocate --
        device = patches.device
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        zvid = th.zeros(vid_shape,device=device,dtype=th.float32)
        # print("patches.shape: ",patches.shape)


        # -- exec --
        stnls_cuda.nlfold_forward(vid, zvid, patches,
                                  stride, dilation,
                                  use_adj,  reflect_bounds)

        # -- backward --
        ctx.save_for_backward(zvid)
        ctx.qstart = qstart
        ctx.stride = stride
        ctx.qNum = patches.shape[1]
        ctx.dilation = dilation
        ctx.use_adj = use_adj
        ctx.reflect_bounds = reflect_bounds
        ctx.pt = patches.shape[2]
        ctx.ps = patches.shape[5]

        return vid/zvid

    @staticmethod
    def backward(ctx, grad_vid):

        # -- unpack --
        grad_vid = grad_vid.contiguous()
        zvid, = ctx.saved_tensors
        ps,pt = ctx.ps,ctx.pt
        dilation = ctx.dilation
        qstart = ctx.qstart
        stride = ctx.stride
        qNum = ctx.qNum
        use_adj = ctx.use_adj
        reflect_bounds = ctx.reflect_bounds

        # -- alloc --
        b,t,c,h,w  = grad_vid.shape
        npix = t*h*w
        colors = grad_vid.shape[2]
        device = grad_vid.device
        grad_patches = allocate_patches(b,qNum,ps,pt,colors,device)

        # -- backward --
        grad_vid /= zvid
        stnls_cuda.nlfold_backward(grad_patches,grad_vid,
                                   stride,dilation,
                                   use_adj,reflect_bounds)

        return grad_patches,None,None,None,None,None,None,None

class NlFold(th.nn.Module):

    def __init__(self,vid_shape,stride=1,dilation=1,
                 use_adj=False,reflect_bounds=True):
        super().__init__()
        self.vid_shape = vid_shape
        self.stride = stride
        self.dilation = dilation
        self.use_adj = use_adj
        self.reflect_bounds = reflect_bounds

    def forward(self, patches):
        vid = nlfold.apply(patches, self.vid_shape,
                           self.stride,self.dilation,
                           self.use_adj,self.reflect_bounds)
        return vid

