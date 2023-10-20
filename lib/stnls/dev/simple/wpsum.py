
import stnls
from einops import rearrange
import torch as th

def run_patches(vid,dists,inds,ps,stride0,use_adj=True,
                pt=1,dilation=1,reflect_bounds=True):

    # -- init --
    unfoldk = stnls.UnfoldK(ps,pt=pt,dilation=dilation,
                            use_adj=use_adj,reflect_bounds=reflect_bounds,
                            use_atomic=True)

    # -- forward --
    nheads = inds.shape[1]
    vid = rearrange(vid,'b t (HD c) h w -> HD b t c h w',HD=nheads)
    wpatches = []
    dists = dists[...,None,None,None,None]
    for h in range(nheads):
        patches = unfoldk(vid[h],inds[:,h])
        wpatches_h = th.sum(dists[:,h] * patches,2)
        wpatches.append(wpatches_h)
    wpatches = th.stack(wpatches,1) # b q HD k pt c ph pw (k == 1)
    return wpatches

def run(vid,dists,inds,ps,stride0,use_adj=True,pt=1,dilation=1,reflect_bounds=True):

    # -- init --
    unfoldk = stnls.UnfoldK(ps,pt=pt,dilation=dilation,
                            use_adj=use_adj,reflect_bounds=reflect_bounds,
                            use_atomic=True)
    fold = stnls.iFoldz(vid.shape,stride=stride0,
                        dilation=dilation,use_adj=use_adj,
                        reflect_bounds=reflect_bounds)

    # -- forward --
    patches = unfoldk(vid,inds)
    wpatches = th.sum(dists * patches,1)
    fold(wpatches)
    vid = fold.vid / fold.zvid
    return vid
