"""

Patch Database Sum (pdb sum) Aggregation

Code from N3Net

"""

import torch as th
import torch.nn as nn
from einops import rearrange
import stnls

# -- local --
import .utils import indexed_matmul_2_efficient,vid_to_raster_inds

def init(cfg):
    return PdbAgg(cfg.k_a,cfg.ps,cfg.pt,cfg.stride0,cfg.pdbagg_chunk_size)

class PdbAgg(nn.Module):

    def __init__(self,k,ps,pt,stride0,chunk_size):
        super().__init__()
        self.k = k
        self.ps = ps
        self.pt = pt
        self.stride0 = stride0
        self.unfold = stnls.iUnfold(ps,stride=stride0)
        self.chunk_size = chunk_size

    def __call__(self,vid,dists,inds):

        # -- get indices --
        dev = vid.device
        b,t,c,iH,iW = vid.shape
        stride0 = self.stride0
        ps,pt = self.ps,self.pt
        nheads = dists.shape[1]
        Q = dists.shape[2]

        # -- limiting --
        if self.k > 0:
            dists = dists[...,:self.k].contiguous()
            inds = inds[...,:self.k,:].contiguous()

        # -- shape for index_matmul... --
        dists = rearrange(dists,"B HD Q K -> (B HD) Q K 1")
        inds  = rearrange(inds,"B HD Q K tr -> (B HD) Q K 1 tr")

        # -- convert inds --
        inds = vid_to_raster_inds(inds,iH,iW,stride0,dev)[0] # from inds[None,:]

        # -- get patches --
        patches = self.unfold(vid)
        shape_str = 'B Q 1 1 (HD C) ph pw -> (B HD) Q (C ph pw)'
        patches = rearrange(patches,shape_str,HD=nheads)

        # -- accumulate! --
        patches = indexed_matmul_2_efficient(patches, dists, inds,
                                             chunk_size=self.chunk_size)
        # assert patches.shape[-1] == 1


        # -- reshape --
        # shape_str = 'b h q c ph pw -> (b q ph pw) (h c)'
        # shape_str = 'b h (Q n) c ph pw -> (b o ph pw) n (h c)'
        shape_str = '(B HD) Q (C ph pw) 1 -> (B Q ph pw) (HD C)'
        patches = rearrange(patches,shape_str,HD=nheads,ph=ps,pw=ps)

        return patches

