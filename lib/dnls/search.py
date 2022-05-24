
# -- python --
import torch as th

# -- cpp cuda kernel --
import dnls_cuda


class SearchNlFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid, queryInds, fflow, bflow, k, ps, pt, ws, wt, chnls):
        """
        vid = [T,C,H,W]
        patches = [NumQueries,K,pt,C,ps,ps]
        queryInds = [NumQueries,K,3]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)
        """
        # patches = allocate_patches(vid,queryInds,k,ps,pt)
        nlDists,nlInds = allocate_bufs(queryInds,k)
        dnls_cuda.search_forward(vid, queryInds, fflow, bflow,
                                 nlDists, nlInds, ps, pt, ws, wt, chnls)
        ctx.save_for_backward([nlInds,nlDists,vid.shape])
        return nlDists,nlInds

    @staticmethod
    def backward(ctx, grad_nlDists, grad_nlInds):
        nlInds,nlDists,vid_shape = ctx.saved_tensors
        # vid = allocate_vid(vid_shape,grad_patches.device)
        dnls_cuda.gather_backward(grad_patches.contiguous(),
                                  vid,nlInds,nlDists)
        return vid

    @staticmethod
    def allocate_vid(vid_shape,device):
        vid = th.zeros(vid_shape,device=device,dtype=th.float32)
        return vid

    @staticmethod
    def allocate_bufs(queryInds,k):
        device = queryInds.device
        nq = queryInds.shape[0]
        nlInds = th.zeros((nq,k,3),device=device,dtype=th.int32)
        nlDists = th.zeros((nq,k),device=device,dtype=th.float32)
        return nlInds,nlDists

class SearchNl(th.nn.Module):
    # [patches -> video] @ queryInds

    def __init__(self, fflow, bflow, k, ps, pt, ws, wt):
        super(SearchNl, self).__init__()
        self.k = k
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.wt = wt
        self.fflow = fflow
        self.bflow = bflow

    def forward(self, vid, patches, queryInds):
        return SearchNlFunction.apply(vid,queryInds,
                                      self.fflow,self.bflow,
                                      self.k,self.ps,self.pt,self.ws,self.wt)

