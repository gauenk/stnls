
# -- python --
import torch as th

# -- cpp cuda kernel --
import stnls_cuda



def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    wvid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid,wvid

def allocate_patches(inds,ps,pt,c):
    device = inds.device
    nq,k = inds.shape[:2]
    patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches

class fold_k(th.autograd.Function):
    """
    [patches -> video] @ inds

    inds.shape = [NumQueries,K,3]
    patches.shape = [NumQueries,K,pt,c,ps,ps]
    """

    @staticmethod
    def forward(ctx, patches, dists, inds, vid, wvid,
                dilation, rand, exact, nreps):
        assert nreps >= 0
        if nreps == 1:
            stnls_cuda.foldk_forward_race(vid, wvid, patches, dists, inds,
                                          dilation, rand, exact)
        elif nreps > 1:
            for i in range(nreps):
                _vid,_wvid = allocate_vid(vid.shape,vid.device)
                stnls_cuda.foldk_forward_race(_vid, _wvid, patches, dists, inds,
                                              dilation, rand, exact)
                vid,wvid = vid+_vid,wvid+_wvid
            vid,wvid = vid/nreps,wvid/nreps
        ctx.save_for_backward(dists,inds)
        ctx.dilation = dilation
        ctx.pt = patches.shape[2]
        ctx.ps = patches.shape[5]
        return vid,wvid

    @staticmethod
    def backward(ctx, grad_vid, grad_wvid):
        dists,inds = ctx.saved_tensors
        grad_vid = grad_vid.contiguous()
        dilation = ctx.dilation
        ps,pt = ctx.ps,ctx.pt
        patches = allocate_patches(inds,ps,pt,grad_vid.shape[1])
        stnls_cuda.foldk_backward(grad_vid,patches,dists,inds,dilation)
        return patches,None,None,None,None,None,None,None,None,None,None

class FoldK(th.nn.Module):
    # [patches -> video] @ inds

    def __init__(self, vid_shape, dilation=1, rand=True,
                 exact=False, nreps=1, device="cuda:0"):
        super(FoldK, self).__init__()
        self.vid_shape = vid_shape
        self.vid,self.wvid = allocate_vid(vid_shape,device)
        self.dilation = dilation
        self.rand = rand
        self.exact = exact
        self.nreps = nreps
        self.device = device

    def forward(self, patches, dists, inds):
        vid,wvid = allocate_vid(self.vid_shape,self.device)
        vid,wvid = fold_k.apply(patches,dists,inds,vid,wvid,
                                self.dilation,self.rand,self.exact,
                                self.nreps)
        self.vid += vid
        self.wvid += wvid
        return self.vid,self.wvid


def _apply(vshape,patches,dists,inds,dilation=1,rand=True,exact=False,nreps=1):
    vid,wvid = allocate_vid(vshape,patches.device)
    vid,wvid = fold_k.apply(patches,dists,inds,vid,wvid,
                            dilation,rand,exact,nreps)
    return vid,wvid

