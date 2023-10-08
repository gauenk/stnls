import torch as th
import torch.nn as nn
from einops import rearrange
import stnls

# def init(cfg):

#     # -- unpack params --
#     ps      = cfg.ps
#     pt      = cfg.pt
#     dil     = cfg.dilation
#     reflect_bounds = cfg.reflect_bounds
#     use_adj = False

#     # -- init --
#     wpsum = stnls.reducer.WeightedPatchSum(ps, pt,
#                                            dilation=dil,
#                                            reflect_bounds=reflect_bounds,
#                                            use_adj=use_adj,use_atomic=True)

#     return WpSumAgg(wpsum)

class WeightedSum(nn.Module):

    def __init__(self,ps,pt=1,dilation=1,reflect_bounds=True,use_adj=True):
        super().__init__()
        self.wpsum = stnls.reducer.WeightedSum(ps, pt,dilation=dilation,
                                               reflect_bounds=reflect_bounds,
                                               use_adj=use_adj)


    def __call__(self,vid_in,dists,inds):

        # -- contiguous --
        # print("dists.shape: ",dists.shape)
        B,HD,T,nH,nW,K = dists.shape
        # dists = dists.reshape(B,HD,T*nH*nW,K)
        # inds = inds.reshape(B,HD,T*nH*nW,K,3)
        dists = dists.contiguous()
        inds = inds.contiguous()


        # -- aggregate --
        vid_out = self.wpsum(vid_in,dists,inds)

        return vid_out
