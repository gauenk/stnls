"""

A gather _without_ a race condition with k == 1

"""


# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(nq,k,ps,pt,c,device):
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class iUnfoldFunction(th.autograd.Function):
    """
    [patches -> video] @ nlInds

    nlInds.shape = [NumQueries,K,3]
    patches.shape = [NumQueries,K,pt,c,ps,ps]
    """

    @staticmethod
    def forward(ctx, patches, vid, start, coords, stride, dilation,
                adj, only_full, use_reflect):

        # -- unpack --
        top,left = coords[:2]
        btm,right = coords[2:]

        # -- allocate --
        colors = vid.shape[1]
        device = vid.device

        # -- forward --
        dnls_cuda.iunfold_forward(vid, patches,
                                  top,left,btm,right,
                                  start, stride, dilation,
                                  adj, only_full, use_reflect)

        # -- store --
        ctx.start = start
        ctx.coords = coords
        ctx.dilation = dilation
        ctx.vid_shape = vid.shape
        ctx.stride = stride
        ctx.adj = adj
        ctx.only_full = only_full
        ctx.use_reflect = use_reflect

        return patches

    @staticmethod
    def backward(ctx, grad_patches):

        # -- fmt --
        grad_patches = grad_patches.contiguous()
        start = ctx.start
        stride = ctx.stride
        dilation = ctx.dilation
        vid_shape = ctx.vid_shape
        coords = ctx.coords
        adj = ctx.adj
        only_full = ctx.only_full
        use_reflect = ctx.use_reflect

        # -- unpack --
        top,left = coords[:2]
        btm,right = coords[2:]

        # -- allocate --
        grad_vid = allocate_vid(vid_shape,grad_patches.device)

        # -- forward --
        dnls_cuda.iunfold_backward(grad_vid,grad_patches,
                                   top,left,btm,right,
                                   start,stride,dilation,
                                   adj,only_full,use_reflect)

        return None,grad_vid,None,None,None,None,None,None,None,None

class iUnfold(th.nn.Module):
    # [patches -> video] @ nlInds [with k == 1]

    def __init__(self, ps, coords, pt=1, stride=1, dilation=1,
                 adj=0,only_full=False,match_nn=False,
                 border="reflect"):
        super(iUnfold, self).__init__()
        self.ps = ps
        self.pt = pt
        self.stride = stride
        self.dilation = dilation
        self.patches = th.empty(0)
        self.coords = coords

        # -- handle borders --
        self.use_reflect = border == "reflect"
        assert border in ["reflect","zero"]

        # -- mutually exclusive (match_nn, (adj,only_full) --
        self.match_nn = match_nn
        if match_nn:
            self.adj = ps//2
            self.only_full = True
        else:
            self.adj = adj
            self.only_full = only_full

    def _get_coords(self,vshape):
        # top,left,btm,right
        if self.coords is None:
            t,c,h,w = vshape
            return [0,0,h,w]
        else:
            return self.coords

    def _get_start_num(self,start,num,coords,vshape):
        t = vshape[0]
        ps,dil,stride = self.ps,self.dilation,self.stride
        if start == -1: start = 0
        if num == -1:
            top,left,btm,right = coords
            h,w = btm - top,right - left
            if self.only_full:
                n_h = (h - (ps-1)*dil - 1)//stride + 1
                n_w = (w - (ps-1)*dil - 1)//stride + 1
            else:
                n_h = (h - 1)//stride + 1
                n_w = (w - 1)//stride + 1
            num = t * n_h * n_w
        return start,num

    def forward(self, vid, start=-1, num=-1):
        coords = self._get_coords(vid.shape)
        start,num = self._get_start_num(start,num,coords,vid.shape)
        colors = vid.shape[1]
        patches = allocate_patches(num,1,self.ps,self.pt,colors,vid.device)
        patches = iUnfoldFunction.apply(patches,vid,start,coords,self.stride,
                                        self.dilation,self.adj,self.only_full,
                                        self.use_reflect)
        return patches


