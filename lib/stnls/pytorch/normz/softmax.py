# -- torch nn --
import torch as th
import torch.nn as nn
from einops import rearrange


def init(cfg):
    return SoftmaxNormalize(cfg.normz_scale,cfg.normz_drop_rate,cfg.dist_type,cfg.k_agg)

class SoftmaxNormalize(nn.Module):

    def __init__(self,scale,drop_rate=0.,dist_type="l2",k_agg=-1):
        super().__init__()
        self.scale = scale
        self.drop_rate = drop_rate
        self.norm = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(drop_rate)
        self.dist_type = dist_type
        self.k_agg = k_agg

    def __call__(self,dists,inds):

        # -- restrict --
        if self.k_agg > 0:
            dists = dists[...,:self.k_agg].contiguous()
            inds = inds[...,:self.k_agg,:].contiguous()
        else:
            dists = dists.contiguous()
            inds = inds.contiguous()

        # -- handle dist type --
        if self.dist_type == "l2":
            dists = -dists

        # -- scale --
        dists = self.scale * dists

        # -- normalize --
        dists = self.norm(dists)

        # -- drop-rate --
        dists = self.drop(dists)

        # # -- contiguous --
        # dists = dists.contiguous()
        # inds = inds.contiguous()

        return dists,inds
