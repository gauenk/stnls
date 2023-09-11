# -- basic --
import numpy as np
import torch as th
from torchvision.utils import save_image
from easydict import EasyDict as edict
from einops import rearrange

# -- data --
import data_hub


def get_video():
    cfg = edict()
    cfg.device = "cuda:0"
    cfg.dset = "te"
    cfg.dname = "set8"
    cfg.nframes = 6
    cfg.frame_start = 0
    cfg.frame_end = 5
    cfg.isize = None
    cfg.vid_name = "sunflower"
    cfg.sigma = 0.1
    cfg.read_flows = True
    cfg.sigma = 15

    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                     cfg.frame_start,cfg.frame_end)

    index = indices[0]

    sample = data[cfg.dset][index]
    noisy,clean = sample['noisy']/255.,sample['clean']/255.
    fflow,bflow = sample['fflow'],sample['bflow']
    # noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)

    return noisy,clean,fflow,bflow

def get_search_grid(ws):
    device = "cpu"
    ws_min = -(ws-1)//2
    ws_max = ws_min + ws
    grid_y, grid_x = th.meshgrid(th.arange(ws_min, ws_max, dtype=th.long, device=device),
                                 th.arange(ws_min, ws_max, dtype=th.long, device=device))
    grid = th.stack((grid_y, grid_x), 2)  # W(x), H(y), 2
    grid = rearrange(grid,'H W two -> two H W')
    grid = th.cat([th.zeros_like(grid[[0]]),grid],0)
    return grid

def bound(loc,lim):
    if loc < 0:
        return -loc
    elif loc > (lim-1):
        return 2*(lim-1) - loc
    else:
        return loc

def get_pix(vid,loc):
    T,C,H,W = vid.shape
    pix = th.zeros(C)
    for ix in range(2):
        for jx in range(2):
            i = int(loc[1]+ix)
            wi = max(0.,1-abs(1.*i - loc[1]))
            j = int(loc[2]+jx)
            wj = max(0.,1-abs(1.*j - loc[2]))
            w = wi * wj
            i = bound(i,H)
            j = bound(j,W)
            # print(w)
            pix += w*vid[int(loc[0]),:,i,j]
    return pix

def delta_patch(vid0,vid1,loc0,loc1,ps):
    delta = 0
    poff = -ps//2
    for pi in range(ps):
        for pj in range(ps):
            loc0_ij = [loc0[0],loc0[1]+pi+poff,loc0[2]+pj+poff]
            loc1_ij = [loc1[0],loc1[1]+pi+poff,loc1[2]+pj+poff]
            pix0 = get_pix(vid0,loc0_ij)
            pix1 = get_pix(vid1,loc1_ij)
            delta += th.sum((pix0-pix1)**2)
    return delta


def search_deltas(vid0,vid1,fflow,bflow,loc0,grid,stride1,ws,ps,K=9):
    dmap = th.zeros_like(grid[0])*1.
    flow = get_pix(fflow,loc0).flip(0)
    for wi in range(ws):
        for wj in range(ws):
            # -- get search location --
            off_i = grid[:,wi,wj]
            loc1 = [1,]+[loc0[i] + flow[(i-1)] + stride1*off_i[i] for i in range(1,3)]
            # print([wi,wj],off_i)
            # loc1 = [0,]+[loc0[i] + stride1*off_i[i] for i in range(1,3)]
            # print(wi,wj,[stride1*off_i[i] for i in range(1,3)])

            # -- compute delta ---
            dmap[wi,wj] = delta_patch(vid0,vid1,loc0,loc1,ps)
            # print(loc1,dmap[wi,wj])
            # if off_i[1] == 0 and off_i[2] == 0:
            #     dmap[wi,wj] = 10000.
    # dmap -= dmap.min()
    eps = 1e-10
    # print(dmap)
    # print(dmap)
    # dmap = th.log(dmap + eps)
    dmap -= dmap.min()
    dmap /= dmap.max()
    dmap = th.exp(-10*dmap)
    # # print(dmap)
    dmap -= dmap.min()
    dmap /= dmap.max()
    # print(dmap)

    # -- viz topk --
    H,W = dmap.shape
    dmap0 = dmap.clone().view(-1)
    topk = th.topk(dmap0,K,largest=True)
    # print("vals.")
    # print(topk.values)
    # inds = topk.indices
    krank = 1#th.exp(-(1./2)*th.arange(K)/K)
    # print(krank)
    # print("search_deltas: ")
    # print(th.stack([topk.indices//ws,topk.indices%ws],1))
    # print(th.stack([th.floor_divide(topk.indices,ws)-(ws//2),
    #                 topk.indices%ws-(ws//2)],1)*stride1)

    # dmap0[topk.indices] = dmap.max()+1e-5
    dmap = dmap.reshape(-1)
    dmap[topk.indices] = 0
    # dmap /= dmap.max()
    # dmap /= dmap.max()
    # dmap0[topk.indices] = dmap.max()+1e-5
    dmap0[topk.indices] = 1.

    dmap0 = dmap0.reshape(1,H,W)
    dmap = dmap.reshape(1,H,W).repeat(2,1,1)
    dmap = th.cat([dmap0,dmap],0)
    # print(dmap.max())
    # dmap /= dmap.max()
    # print(dmap.shape)

    return dmap

def main():

    # -- config --
    # ws = 51
    ws = 41
    ps = 7
    stride1 = 1.25

    # -- read data --
    noisy,clean,fflow,bflow = get_video()
    # print(noisy.shape,fflow.shape)

    # -- compute map --
    grid = get_search_grid(ws)
    loc0 = [0,300,300]
    dmap = search_deltas(clean,clean,fflow,bflow,loc0,grid,stride1,ws,ps)

    # -- save --
    save_image(dmap,"dmap.png")

if __name__ == "__main__":
    main()
