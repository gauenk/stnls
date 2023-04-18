import torch as th
import torch.nn as nn
from einops import rearrange


def init(cfg):
    return SoftmaxNormalize(cfg.k_n,cfg.normz_scale,cfg.normz_drop_rate,cfg.dist_type)

class SoftmaxNormalize(nn.Module):

    def __init__(self,k,scale,drop_rate=0.,dist_type="l2"):
        super().__init__()
        self.k = k
        self.scale = scale
        self.drop_rate = drop_rate
        self.norm = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(drop_rate)
        self.dist_type = dist_type

    def __call__(self,dists):

        # -- handle dist type --
        if self.dist_type == "l2":
            dists = -dists

        # -- limiting --
        if self.k > 0:
            dists = dists[...,:self.k].contiguous()

        # -- scale --
        dists = self.scale * dists

        # -- normalize --
        dists = self.norm(dists)

        # -- drop-rate --
        dists = self.drop(dists)

        # -- contiguous --
        dists = dists.contiguous()

        return dists
