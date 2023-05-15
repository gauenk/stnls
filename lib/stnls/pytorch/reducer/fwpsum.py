"""

FoldedWeightedPatchSum

input: video
output: video

"""

# -- python --
import torch as th
from einops import rearrange

# -- fold --
# from stnls.pytorch.tile import iFoldz
import stnls
from .wpsum import WeightedPatchSum

# -- cpp cuda kernel --
import stnls_cuda

# -- misc --
from ...utils.timer import ExpTimer

def allocate_vid(vid_shape,device):
    vid = th.zeros(vid_shape,device=device,dtype=th.float32)
    return vid

def allocate_patches(b,nq,nhead,pt,c,ps,device):
    patches = th.zeros((b,nq,nhead,pt,c,ps,ps),device=device,dtype=th.float32)
    return patches


class FoldedWeightedPatchSum(th.nn.Module):
    # [video -> patches] @ inds

    def __init__(self, ps, stride0, batchsize=-1, pt=1, dilation=1,
                 reflect_bounds=True, use_adj=True, off_H=0, off_W=0,
                 rbwd=False, nbwd=1, exact=False, use_atomic=True):
        super().__init__()

        self.ps = ps
        self.pt = pt
        self.stride0 = stride0
        self.batchsize = batchsize
        self.dilation = int(dilation)
        self.use_adj = use_adj
        self.reflect_bounds = reflect_bounds
        self.off_H = off_H
        self.off_W = off_W
        self.rbwd = rbwd
        self.nbwd = nbwd
        self.exact = exact
        self.use_atomic = use_atomic

        # -- init --
        self.wpsum = WeightedPatchSum(self.ps,self.pt,self.dilation,
                                      self.reflect_bounds,self.use_adj,
                                      self.off_H,self.off_W,self.use_atomic)

    def forward(self, vid, dists, inds):

        # -- init --
        wpsum = self.wpsum
        fold = stnls.iFoldz(vid.shape,stride=self.stride0,dilation=self.dilation,
                            use_adj=self.use_adj,reflect_bounds=self.reflect_bounds,
                            device=vid.device)

        # -- batching info --
        nbatch = self.batchsize
        nqueries = dists.shape[2]
        if nbatch < 0: nbatch = nqueries
        nbatches = (nqueries-1)//nbatch+1

        # -- run batches --
        for batch in range(nbatches):

            # -- get batch --
            qstart = nbatch*batch
            qend = min(qstart+nbatch,nqueries)
            # print(qstart,qend)
            if not(qend - qstart > 0): break
            dists_b = dists[:,:,qstart:qend].contiguous()
            inds_b = inds[:,:,qstart:qend].contiguous()

            # -- exec --
            patches = wpsum(vid,dists_b,inds_b)

            # -- fold --
            patches = rearrange(patches,'b H q pt c h w -> b q 1 pt (H c) h w')
            fold(patches,qstart)

        # -- normalize --
        vid_agg,vidz = fold.vid,fold.zvid
        vid_agg = vid_agg / vidz
        if th.any(th.isnan(vid_agg)):
            # print_nan_info(vid,vid_agg,vidz,dists,inds,state,self.search_cfg)
            print("[stnls/.../fwpsum.py] Nan found.")
            exit(0)

        return vid_agg

    def flops(self, nrefs, chnls_per_head, nheads, k):

        # -- init --
        flops = 0

        # -- unpack --
        chnls = chnls_per_head
        ps,pt = self.ps,self.pt

        # -- compute weighted patch sum --
        flops_per_patch = 2 * (chnls * ps * ps * pt) # multi weight & add to accumulate
        flops_per_ref = flops_per_patch * k # accumulate over "k" patches
        flops = flops_per_ref * nrefs * nheads# do for each reference

        return flops


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#   [Direct API]  stnls.reducer.fwpsum(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid, dists, inds, ps, batchsize=-1,
           pt=1, dilation=1,reflect_bounds=True,
           use_adj=True, off_H0=0, off_W0=0, off_H1=0, off_W1=0,
           rbwd=True, nbwd=1, exact=False, use_atomic=True):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = FoldedWeightedPatchSumFunction.apply
    return fxn(vid,dists,inds,ps,batchsize,
               pt,dilation,reflect_bounds,use_adj,
               off_H0,off_W0,off_H1,off_W1,
               rbwd,nbwd,exact,use_atomic)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#   [Python Dict API] stnls.reducer.fwpsum(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg):
    pairs = {"ps":7,"batchsize":-1,"pt":1,"dilation":1,
             "reflect_bounds":True, "use_adj":True,
             "off_H0":0,"off_W0":0,"rbwd":True, "nbwd":1,
             "exact":False, "use_atomic": True}
    return extract_pairs(pairs,cfg)

def init(cfg):
    cfg = extract_config(cfg)
    reducer = FoldedWeightedPatchSum(
        cfg.ps, batchsize=cfg.batchsize,
        pt=cfg.pt, dilation=cfg.dilation,
        reflect_bounds=cfg.reflect_bounds,
        adj=cfg.use_adj,off_H=cfg.off_H0,off_W=cfg.off_W0,
        rbwd=cfg.rbwd, nbwd=cfg.nbwd,
        exact=cfg.exact, use_atomic=cfg.use_atomic)
    return reducer


