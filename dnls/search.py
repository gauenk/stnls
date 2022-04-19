
# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


class SearchNlFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid, queryInds, k, ps, pt, ws, wt):
        """
        vid = [T,C,H,W]
        patches = [NumQueries,K,pt,C,ps,ps]
        queryInds = [NumQueries,K,3]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """
        patches = allocate_patches(vid,queryInds,k,ps,pt)
        nlInds,nlDists = allocate_bufs(patches)
        dnls_cuda.search_forward(vid, patches, queryInds, nlInds, nlDists, ws, wt)
        ctx.save_for_backward([nlInds,nlDists,vid.shape])
        return patches

    @staticmethod
    def backward(ctx, grad_patches):
        nlInds,nlDists,vid_shape = ctx.saved_tensors
        vid = allocate_vid(vid_shape,grad_patches.device)
        dnls_cuda.gather_backward(grad_patches.contiguous(),
                                  vid,nlInds,nlDists)
        return vid

    @staticmethod
    def allocate_vid(vid_shape,device):
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        return vid

    @staticmethod
    def allocate_patches(vid,queryInds,k,ps,pt):
        device = vid.device
        nq,c = queryInds.shape[0],vid.shape[1]
        assert c in [1,3],"Must be the color channel."
        patches = th.zeros((nq,k,pt,c,ps,ps),device=device,dtype=th.float32)
        return patches

    @staticmethod
    def allocate_bufs(patches):
        device = patches.device
        nq,k = patches.shape[:2]
        nlInds = th.zeros((nq,k,3),device=device,dtype=th.int32)
        nlDists = th.zeros((nq,k),device=device,dtype=th.float32)
        return nlInds,nlDists

class SearchNl(th.nn.Module):
    # [patches -> video] @ queryInds

    def __init__(self, k, ps, pt, ws, wt):
        super(SearchNl, self).__init__()
        self.k = k
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.wt = wt

    def forward(self, vid, patches, queryInds):
        return SearchNlFunction.apply(vid,patches,queryInds,self.ws,self.wt)

