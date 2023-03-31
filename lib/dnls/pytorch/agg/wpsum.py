import torch as th
import torch.nn as nn
from einops import rearrange
import dnls

def init(cfg):

    # -- unpack params --
    ps      = cfg.ps
    pt      = cfg.pt
    dil     = cfg.dilation
    exact = cfg.exact
    reflect_bounds = cfg.reflect_bounds
    adj = ps//2

    # -- init --
    wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=0, w_off=0,
                                                dilation=dil,
                                                reflect_bounds=reflect_bounds,
                                                adj=adj, exact=exact)

    return WpSumAgg(cfg.k_a,wpsum)

class WpSumAgg(nn.Module):

    def __init__(self,k,wpsum):
        super().__init__()
        self.k = k
        self.wpsum = wpsum

    def __call__(self,vid,dists,inds):

        # -- limiting --
        if self.k > 0:
            dists = dists[...,:self.k]
            inds = inds[...,:self.k,:]

        # -- contiguous --
        dists = dists.contiguous()
        inds = inds.contiguous()

        # -- aggregate --
        patches = self.wpsum(vid,dists,inds)

        # -- reshape --
        ps = patches.shape[-1]
        ntotal = dists.shape[-2]
        shape_str = 'b h q c ph pw -> (b q ph pw) (h c)'
        patches = rearrange(patches,shape_str)


        return patches
