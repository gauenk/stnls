"""

 Slic is easy with our packages

"""

# -- basic --
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from dev_basics.utils import vid_io

# -- exps --
from dev_basics.utils.misc import set_seed

# -- optical flow --
from dev_basics import flow

# -- data --
import data_hub

# -- non-local opts --
import stnls

# -- benchmarking --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt

# -- view segmentation --
from torchvision.utils import draw_segmentation_masks
from skimage.segmentation import mark_boundaries


"""

    SLIC


"""

def run_cas(vid,ws,wt,ps,stride0,full_ws,M=0.5,
            softmax_weight=10.,niters=-1,use_rand=False):

    # -- config --
    use_flow = False
    agg_ps = ps

    # -- init video --
    vid = append_grid(vid,M,stride0)
    B,T,F,H,W = vid.shape

    # -- init flows --
    flows = flow.orun(vid,use_flow,ftype="cv2")
    flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,stride0)
    flows = flows[:,None].round().int()

    # -- init slic state --
    device = vid.device
    HD = 1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    dists_k = th.ones((B,HD,T,nH,nW,1),device=device,dtype=th.float32)
    flows_k = th.zeros((B,HD,T,nH,nW,1,3),device=device,dtype=th.int)

    # -- init centroids --

    weights = th.softmax(-softmax_weight*dists_k,-1)
    agg = stnls.agg.PooledPatchSum(agg_ps,stride0,itype="int")
    pooled = rearrange(agg(vid,weights,flows_k),'b hd t c h w -> b t (hd c) h w')
    # print("Delta: ",th.mean((pooled-pooled0)**2).item())

    # -- compute pairwise searches --

    search = stnls.search.NonLocalSearch(ws,wt,ps,-1,nheads=1,dist_type="l2",
                                         stride0=stride0,strideQ=agg_ps,
                                         self_action="anchor_self",
                                         full_ws=full_ws,itype="int")
    dists_k,flows_k = search(pooled,vid,flows)
    print(flows_k[0,0,0,:2,:2])

    # -- pool all neighbors --

    dists_k = th.ones_like(dists_k)
    weights = th.softmax(-softmax_weight*dists_k,-1)
    agg = stnls.agg.PooledPatchSum(agg_ps,stride0,itype="int")
    pooled = rearrange(agg(vid,weights,flows_k),'b hd t c h w -> b t (hd c) h w')
    # print("Delta: ",th.mean((pooled-pooled0)**2).item())

    # -- find most similar to average --
    search = stnls.search.NonLocalSearch(ws,wt,ps,1,nheads=1,dist_type="l2",
                                         stride0=stride0,strideQ=agg_ps,
                                         # self_action="anchor_self",
                                         full_ws=full_ws,itype="int")
    dists_k,flows_k = search(pooled,vid,flows)
    print(dists_k[0,0,0,:2,:2])
    print(flows_k[0,0,0,:2,:2])

    # -- compute mask for output --
    B,T,F,H,W = vid.shape
    mask = get_masks(flows_k,stride0,B,T,H,W)
    return mask

def slic_v2_run(vid,ws,wt,ps,stride0,full_ws,M=0.5,
                softmax_weight=10.,niters=-1,use_rand=False):

    # -- config --
    use_flow = False
    agg_ps = ps

    # -- init video --
    vid = append_grid(vid,M,stride0)
    B,T,F,H,W = vid.shape

    # -- init flows --
    flows = flow.orun(vid,use_flow,ftype="cv2")
    flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,stride0)
    flows = flows[:,None].round().int()

    # -- init slic state --
    device = vid.device
    HD = 1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    dists_k = th.ones((B,HD,T,nH,nW,1),device=device,dtype=th.float32)
    flows_k = th.zeros((B,HD,T,nH,nW,1,3),device=device,dtype=th.int)

    # -- iterations --
    for i in range(niters):


        #
        # -- update centroids --
        #

        # -- pooling (u_c) --
        weights = th.softmax(-softmax_weight*dists_k,-1)
        agg = stnls.agg.PooledPatchSum(agg_ps,stride0,itype="int")
        pooled = rearrange(agg(vid,weights,flows_k),'b hd t c h w -> b t (hd c) h w')
        # print("Delta: ",th.mean((pooled-pooled0)**2).item())

        # -- compute pairwise searches --
        search = stnls.search.NonLocalSearch(ws,wt,ps,-1,nheads=1,dist_type="l2",
                                             stride0=stride0,strideQ=agg_ps,
                                             self_action="anchor_self",
                                             full_ws=full_ws,itype="int")
        dists_k,flows_k = search(pooled,vid,flows)
        # r0,r1 = th.rand_like(pooled),vid
        # dists_k,flows_k = search(r0,r1,flows)

        #
        # -- get similaries of all pixels to each cluster [q_c(p)] --
        #

        # -- create the scattering labels [aka *magic*] --
        names,labels = stnls.agg.scatter_labels(flows,flows_k,ws,wt,
                                                stride0,1,H,W,full_ws)

        # # -- top-k --
        # topk,K0 = stnls.agg.scatter_topk,-1
        # s_dists,s_flows,s_labels = topk(scatter_dists,scatter_flows,
        #                                 scatter_labels,K0,descending=False)

        # # -- [scattering] keys to queries --
        # dists_k,flows_k = run_scatter_k2q(s_dists,s_flows,s_labels,stride0,T,H,W)

        # -- [scattering] queries to keys --
        outs = run_scatter_q2k(dists_k,flows_k,flows,labels,ws,wt,stride0,H,W,full_ws)
        scatter_dists,scatter_flows,scatter_labels = outs

        # -- pooling (u_c) --
        weights = th.softmax(-softmax_weight*dists_k,-1)
        agg = stnls.agg.PooledPatchSum(agg_ps,stride0,itype="int")
        pooled = rearrange(agg(pooled,weights,flows_k),'b hd t c h w -> b t (hd c) h w')
        # print("Delta: ",th.mean((pooled-pooled0)**2).item())


        # -- score --
        print("Score: ",compute_score(vid, pooled, flows_k, ws, wt, ps, stride0))

    return pooled,dists_k,flows_k,s_dists,s_flows

def slic_run(vid,ws,wt,ps,stride0,full_ws,M=0.5,softmax_weight=10.,niters=1):

    # -- config --
    use_flow = False
    agg_ps = ps

    # -- init video --
    vid = append_grid(vid,M,stride0)
    B,T,F,H,W = vid.shape

    # -- init flows --
    flows = flow.orun(vid,use_flow,ftype="cv2")
    flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,stride0)
    flows = flows[:,None].round().int()

    # -- init slic state --
    device = vid.device
    HD = 1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    dists_k = th.ones((B,HD,T,nH,nW,1),device=device,dtype=th.float32)
    flows_k = th.zeros((B,HD,T,nH,nW,1,3),device=device,dtype=th.int)

    # -- iterations --
    for i in range(niters):


        #
        # -- update centroids --
        #

        weights = th.softmax(-softmax_weight*dists_k,-1)
        agg = stnls.agg.PooledPatchSum(agg_ps,stride0,itype="int")
        pooled = rearrange(agg(vid,weights,flows_k),'b hd t c h w -> b t (hd c) h w')
        # print("Delta: ",th.mean((pooled-pooled0)**2).item())

        #
        # -- compute pairwise searches --
        #

        search = stnls.search.NonLocalSearch(ws,wt,ps,-1,nheads=1,dist_type="l2",
                                             stride0=stride0,strideQ=agg_ps,
                                             self_action="anchor_self",
                                             full_ws=full_ws,itype="int")
        dists_k,flows_k = search(pooled,vid,flows)
        # r0,r1 = th.rand_like(pooled),vid
        # dists_k,flows_k = search(r0,r1,flows)

        #
        # -- sparsify (dists_k,flows_k) for each module within cell --
        #

        # -- create the scattering labels [aka *magic*] --
        names,labels = stnls.agg.scatter_labels(flows,flows_k,ws,wt,
                                                stride0,1,H,W,full_ws)

        # -- [scattering] queries to keys --
        outs = run_scatter_q2k(dists_k,flows_k,flows,labels,ws,wt,stride0,H,W,full_ws)
        scatter_dists,scatter_flows,scatter_labels = outs

        # -- top-k --
        topk,K0 = stnls.agg.scatter_topk,1
        s_dists,s_flows,s_labels = topk(scatter_dists,scatter_flows,
                                        scatter_labels,K0,descending=False)

        # -- [scattering] keys to queries --
        dists_k,flows_k = run_scatter_k2q(s_dists,s_flows,s_labels,stride0,T,H,W)

        # -- score --
        print("Score: ",compute_score(vid, pooled, flows_k, ws, wt, ps, stride0))

    return pooled,dists_k,flows_k,s_dists,s_flows

def get_slic_sampling_mask(vid,ws,wt,ps,stride0,full_ws,M=0.5,use_rand=False):
    pooled,dists_k,flows_k,_,_ = slic_run(vid,ws,wt,ps,stride0,full_ws,M)
    vid = append_grid(vid,M,stride0)
    mask = get_sampling_mask(vid, pooled, flows_k, ws, wt, ps, stride0, use_rand)
    return mask

def run_scatter_k2q(s_dists,s_flows,s_labels,stride0,T,H,W):

    # -- prepare weights and flows --
    B,HD,_,K0 = s_dists.shape
    s_dists = s_dists.reshape(B,HD,T,H,W,K0)
    s_flows = s_flows.reshape(B,HD,T,H,W,K0,3)
    s_labels = s_labels.reshape(B,HD,T*H*W,-1)

    # -- scatter --
    stride1 = 1
    dists = stnls.agg.scatter_tensor(s_dists,s_flows,s_labels,
                                     stride1,stride0,H,W)
    flows = stnls.agg.scatter_tensor(s_flows,s_flows,s_labels,
                                     stride1,stride0,H,W)
    flows = -flows

    # -- reshape --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    dists = dists.reshape(B,HD,T,nH,nW,-1)
    flows = flows.reshape(B,HD,T,nH,nW,-1,3)

    return dists,flows

def run_scatter_q2k(dists_k,flows_k,flows,labels,ws,wt,stride0,H,W,full_ws):

    # -- create unique scatter indices [the *magic*] --
    stride1 = 1
    gather_labels = labels.reshape_as(dists_k)

    # -- scattering top-K=1 --
    scatter_weights = stnls.agg.scatter_tensor(dists_k,flows_k,labels,
                                               stride0,stride1,H,W)
    scatter_flows_k = stnls.agg.scatter_tensor(flows_k,flows_k,labels,
                                               stride0,stride1,H,W)
    scatter_labels = stnls.agg.scatter_tensor(gather_labels,flows_k,labels,
                                              stride0,stride1,H,W,invalid=-th.inf)
    scatter_flows_k = -scatter_flows_k

    return scatter_weights,scatter_flows_k,scatter_labels


def compute_score(vid, pooled, flows_k, ws, wt, ps, stride0, use_rand=False):
    # -- re-compute distances --
    wr,kr = 1,1.
    k = flows_k.shape[-2]
    refine = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads=1,
                                       stride0=stride0,strideQ=ps,
                                       off_Hq=ps//2,off_Wq=ps//2,
                                       dist_type="l2", itype="int")
    if use_rand: pooled,vid = th.rand_like(pooled),th.rand_like(vid)
    dists_k,flows_k = refine(pooled,vid,flows_k)
    # print("dists_k.shape: ",dists_k.shape)
    # print(dists_k[0,0,0,:2,:2])
    valid = th.logical_and(dists_k>=0,th.isfinite(dists_k))
    return dists_k[th.where(valid)].mean().item()

def get_sampling_mask(vid, pooled, flows_k, ws, wt, ps, stride0, use_rand=False):

    # -- re-compute distances --
    wr,k,kr = 1,1,1.
    refine = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads=1,
                                       stride0=stride0,strideQ=ps,
                                       off_Hq=ps//2,off_Wq=ps//2,
                                       dist_type="l2", itype="int")
    if use_rand: pooled,vid = th.rand_like(pooled),th.rand_like(vid)
    # print(pooled,vid,flows_k)
    dists_k,flows_k = refine(pooled,vid,flows_k)

    # -- compute mask for output --
    B,T,F,H,W = vid.shape
    mask = get_masks(flows_k,stride0,B,T,H,W)
    return mask

def get_masks(flows_k,stride0,B,T,H,W):

    # -- create inds --
    inds = stnls.utils.misc.flow2inds(flows_k,stride0).long()
    inds = rearrange(inds,'b hd t h w 1 tr -> (b hd) (t h w) tr')
    select = inds[:,:,0]*H*W + inds[:,:,1]*W + inds[:,:,2]

    # -- create mask --
    mask = th.zeros(B,T*H*W).to(select.device)
    for bi in range(B):
        mask[bi,select[bi]] = 1
    mask = mask.reshape(B,1,H,W)
    return mask

def append_grid(vid,M,S):
    B,T,F,H,W = vid.shape
    dtype,device = vid.dtype,vid.device
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                 th.arange(0, W, dtype=dtype, device=device))
    grid = th.stack((grid_x, grid_y), -1).float()  # 2, W(x), H(y)
    grid = repeat(grid,'h w two -> b t two h w',b=B,t=T)
    vid = th.cat([vid,M/S*grid],2)
    return vid


"""


    Misc


"""

def load_video(cfg):
    device = "cuda:0"
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,0,cfg.nframes)
    vid = data[cfg.dset][indices[0]]['clean'][None,:].to(device)/255.
    # F = 32
    # B,T,_,H,W = vid.shape
    # vid = th.randn((B,T,F,H,W),device=device,dtype=vid.dtype)
    return vid

def slic_select(vid,ws):

    print("vid.shape: ",vid.shape)
    # -- config --
    ps = 1
    # ws = 3
    wt = 0
    stride0 = 7
    ws = 5
    # stride0,ws = 3,5
    stride1 = 1
    K0 = 1
    softmax_weight = 10.
    k = -1
    full_ws = True
    use_flow = True
    M = 0.5
    use_rand = False
    agg_ps = ps

    # -- compute search window --
    B,T,F,H,W = vid.shape
    search = stnls.search.NonLocalSearch(ws,wt,ps,k,
                                         nheads=1,dist_type="l2",
                                         stride0=stride0,
                                         self_action="anchor_self",
                                         full_ws=full_ws,itype="int")

    flows = flow.orun(vid,use_flow,ftype="cv2")
    flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,stride0)
    flows = flows[:,None].round().int()

    vid = append_grid(vid,M,stride0)
    dists,flows_k = search(vid,vid,flows)
    # inds = stnls.utils.misc.flow2inds(flows_k,stride0)

    # -- scattering top-K=1 --
    K0 = 1
    gather_weights = dists
    names,labels = stnls.agg.scatter_labels(flows,flows_k,ws,wt,
                                            stride0,stride1,H,W,full_ws)
    gather_labels = labels.reshape_as(gather_weights)
    scatter_weights = stnls.agg.scatter_tensor(gather_weights,flows_k,labels,
                                               stride0,stride1,H,W)
    scatter_flows_k = stnls.agg.scatter_tensor(flows_k,flows_k,labels,
                                               stride0,stride1,H,W)
    scatter_labels = stnls.agg.scatter_tensor(gather_labels,flows_k,labels,
                                              stride0,stride1,H,W,invalid=-th.inf)
    scatter_flows_k = -scatter_flows_k
    s_weight,s_flows_k,s_labels = stnls.agg.scatter_topk(scatter_weights,scatter_flows_k,
                                                         scatter_labels,K0,
                                                         descending=False)

    # -- prepare weights and flows --
    pool_type = "pool"
    # pool_type = "wpsum"
    pooled,weights,flows_k = slic_clusters(vid,s_weight,s_flows_k,s_labels,
                                          agg_ps,stride0,stride1,K0,
                                          softmax_weight,pool_type)
    # # print(th.cat([weights[...,None],flows_k],-1))
    # print(pooled.shape,vid.shape)
    # # pooled = pooled[...,psHalf:,psHalf:]
    # print(pooled.shape,vid.shape)

    # -- refine --
    # assert pooled.shape[-2:] == vid.shape[-2:],"Same Spatial Dim [H x W]"
    wr,k,kr = 1,1,1.
    if pool_type == "pool":
        ps = 1
        refine = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads=1,
                                           strideQ=ps,stride0=stride0,
                                           off_Hq=ps//2,off_Wq=ps//2,
                                           dist_type="l2", itype="int")
    elif pool_type == "wpsum":
        refine = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads=1,
                                           stride0=stride0, dist_type="l2", itype="int")
    else:
        raise ValueError(f"Uknown pool type [{pool_type}]")
    if use_rand:
        pooled = th.rand_like(pooled)
        vid = th.rand_like(vid)
    dists,flows_k = refine(pooled,vid,flows_k)
    # weights = th.softmax(-softmax_weight*dists,-1)
    # print(vid.shape,dists.shape,flows_k.shape)

    # -- flows to mask --
    # inds = inds2labels(s_flows_k,cfg,H,W)
    # print(th.cat([dists[...,None],flows_k],-1))
    print(flows_k.shape,
          flows_k[...,0].min(),flows_k[...,0].max(),
          flows_k[...,1].min(),flows_k[...,1].max(),
          flows_k[...,2].min(),flows_k[...,2].max())

    inds = stnls.utils.misc.flow2inds(flows_k,stride0).long()
    inds = rearrange(inds,'b hd t h w 1 tr -> (b hd) (t h w) tr')
    select = inds[:,:,0]*H*W + inds[:,:,1]*W + inds[:,:,2]
    print("select.shape: ",select.shape)
    print(inds.shape,
          inds[...,0].min(),inds[...,0].max(),
          inds[...,1].min(),inds[...,1].max(),
          inds[...,2].min(),inds[...,2].max())
    # # print(inds)
    # # # print(inds.shape)

    mask = th.zeros(B,T*H*W).to(select.device)
    for bi in range(B):
        mask[bi,select[bi]] = 1
    # print(mask.shape,select.shape)
    # mask = mask.scatter_(1,select,1)
    mask = mask.reshape(B,1,H,W)
    # mask = th.zeros((B,H*W))
    # mask[inds] = 1
    # print(H*W/mask.sum())

    # exit()

    return mask

def run_slic(vid,flows,cfg):

    # -- compute search window --
    B,T,F,H,W = vid.shape
    search = stnls.search.NonLocalSearch(cfg.ws,cfg.wt,cfg.ps,cfg.k,
                                         nheads=1,dist_type="l2",
                                         stride0=cfg.stride0,
                                         self_action="anchor_self",
                                         full_ws=cfg.full_ws,itype="int")
    vid = append_grid(vid,cfg.M,cfg.stride0)
    dists,flows_k = search(vid,vid,flows)
    # inds = stnls.utils.misc.flow2inds(flows_k,cfg.stride0)

    # -- scattering top-K=1 --
    K0 = 1
    gather_weights = dists
    names,labels = stnls.agg.scatter_labels(flows,flows_k,cfg.ws,cfg.wt,
                                            cfg.stride0,cfg.stride1,H,W,cfg.full_ws)
    gather_labels = labels.reshape_as(gather_weights)
    scatter_weights = stnls.agg.scatter_tensor(gather_weights,flows_k,labels,
                                               cfg.stride0,cfg.stride1,H,W)
    scatter_flows_k = stnls.agg.scatter_tensor(flows_k,flows_k,labels,
                                               cfg.stride0,cfg.stride1,H,W)
    scatter_labels = stnls.agg.scatter_tensor(gather_labels,flows_k,labels,
                                              cfg.stride0,cfg.stride1,H,W)

    # -- topk --
    scatter_flows_k = -scatter_flows_k
    s_weight,s_flows_k,s_labels = stnls.agg.scatter_topk(scatter_weights,scatter_flows_k,
                                                         scatter_labels,K0,
                                                         descending=False)

    # -- pooling --
    pooled,_,_ = slic_clusters(vid,s_weight,s_flows_k,s_labels,
                          cfg.ps,cfg.stride0,cfg.stride1,K0,cfg.softmax_weight)
    return pooled[:,:,:3],s_flows_k


def slic_clusters(vid,s_weights,s_flows_k,s_labels,ps,stride0,stride1,K0,
                 softmax_weight,pool_method="pool"):

    # -- prepare weights and flows --
    B,T,F,H,W = vid.shape
    HD = s_weights.shape[1]
    s_weights = s_weights.reshape(B,HD,T,H,W,K0)
    s_flows_k = s_flows_k.reshape(B,HD,T,H,W,K0,3)
    s_labels = s_labels.reshape(B,HD,T*H*W,-1)

    # -- run scatters --
    weights = stnls.agg.scatter_tensor(s_weights,s_flows_k,s_labels,
                                       stride1,stride0,H,W)
    flows_k = stnls.agg.scatter_tensor(s_flows_k,s_flows_k,s_labels,
                                       stride1,stride0,H,W)
    flows_k = -flows_k

    # -- reshape --
    K = weights.shape[-1]
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1
    weights = weights.reshape(B,HD,T,nH,nW,K)
    flows_k = flows_k.reshape(B,HD,T,nH,nW,K,3)

    # -- renormalize weights --
    weights = th.softmax(-softmax_weight*weights,-1)
    # print(th.where(weights.sum(-1)>1.))
    # print(th.where(th.abs(weights.sum(-1)-1.)>1e-3))
    # print(weights.sum(-1).round(decimals=2).unique())

    # -- aggregate --
    # print(vid.shape,weights.shape,flows_k.shape,ps,stride0)
    if pool_method == "pool":
        # ps = stride0//2+1
        # ps = ps + (1 - ps % 2) # ensure odd
        # ps = stride0
        # ps = ps + (1 - ps % 2) # ensure odd
        agg = stnls.agg.PooledPatchSum(ps,stride0,itype="int")
    elif pool_method == "wpsum":
        # ps = stride0*2
        ps = ps + (1 - ps % 2) # ensure odd
        agg = stnls.agg.WeightedPatchSum(ps,stride0,itype="int")
    else:
        raise ValueError(f"Uknown pool method [{pool_method}]")
    # print(weights[0,0,0].sum(-1))
    # wsum = weights[0,0,0].sum(-1)
    # print(th.where(th.abs(wsum-1.)>1e-3))
    # print(flows_k[0,0,0])
    vout = agg(vid,weights,flows_k)
    vout = rearrange(vout,'b hd t c h w -> b t (hd c) h w')

    return vout,weights,flows_k

def inds2labels(s_flows_k,cfg,H,W):

    # -- get segmentation labels --
    nH0,nW0 = (H-1)//cfg.stride0+1,(W-1)//cfg.stride0+1
    nH,nW = (H-1)//cfg.stride1+1,(W-1)//cfg.stride1+1
    shape_str = 'b hd (t nh nw) k tr -> b hd t nh nw k tr'
    s_flows_k = rearrange(s_flows_k,shape_str,nh=nH,nw=nW)
    s_inds = stnls.utils.misc.flow2inds(s_flows_k,cfg.stride1)
    nH0,nW0 = H//cfg.stride0,W//cfg.stride0
    s_inds = s_inds[:,0,...,0,:].contiguous() # 1 head, 1 k
    stnls.utils.misc.reflect_inds(s_inds,H,W)

    # -- labels --
    seg_labels = s_inds[...,0]*nH0*nW0
    seg_labels += th.div(s_inds[...,1],cfg.stride0,rounding_mode="floor")*nW0
    seg_labels += th.div(s_inds[...,2],cfg.stride0,rounding_mode="floor")

    # -- fill invalid --
    valid = th.logical_and(seg_labels<100000,seg_labels>-100000)
    S = seg_labels[th.where(valid)].max()
    seg_labels[th.where(~valid)] = S+1

    # -- view --
    # print(seg_labels.shape)
    # print(seg_labels[0,0,-5:,-5:])

    return seg_labels

def labels2masks(labels):
    S = labels.max()+1
    masks = th.zeros([S,]+list(labels.shape),dtype=th.bool).to(labels.device)
    for si in range(S):
        masks[si] = labels==si
    return masks
