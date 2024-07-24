"""

 Slic is easy with our packages

"""

# -- basic --
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from dev_basics.utils import vid_io

# -- interpolation --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

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


def load_video(cfg):
    device = "cuda:0"
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,0,cfg.nframes)
    vid = data[cfg.dset][indices[0]]['clean'][None,:].to(device)/255.
    noisy = data[cfg.dset][indices[0]]['noisy'][None,:].to(device)/255.
    seg_info = None
    if "seg" in data[cfg.dset][indices[0]]:
        seg_info = data[cfg.dset][indices[0]]['seg']
    prepare_seg(seg_info)
    print("vid.shape: ",vid.shape)

    # -- crop --
    vid = vid[...,330:400,420:490]
    noisy = noisy[...,330:400,420:490]
    # exit()
    # vid_io.save_video(vid,"output/segs/","vid")
    # exit()
    return vid,noisy,seg_info

def prepare_seg(seg_info):
    if seg_info is None: return None
    segs = []
    for i in range(seg_info.shape[1]):
        assert len(seg_info[0,i]) == 1
        assert len(seg_info[0,i][0]) == 1
        # print(len(seg_info[0,i][0][0]))
        # print(seg_info[0,i][0][0][0])
        # print(seg_info[0,i][0][0][1])
        for j in range(len(seg_info[0,i][0][0])):
            print(seg_info[0,i][0][0][j])
            print("seg_info[0,i][0][j].shape: ",seg_info[0,i][0][0][j].shape)
            seg_ij = th.from_numpy(seg_info[0,i][0][0][j]*1.)
            segs.append(seg_ij)
    segs = th.stack(segs)[:,None,None]
    segs /= segs.max()
    print(segs.shape)
    vid_io.save_video(segs,"output/segs/","ex")
    return segs

def append_grid(vid,M,S):
    B,T,F,H,W = vid.shape
    dtype,device = vid.dtype,vid.device
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                 th.arange(0, W, dtype=dtype, device=device))
    grid = th.stack((grid_x, grid_y), -1).float()  # 2, W(x), H(y)
    grid = repeat(grid,'h w two -> b t two h w',b=B,t=T)
    vid = th.cat([vid,M/S*grid],2)
    return vid

def run_exp(cfg):

    # -- set seed --
    set_seed(cfg.seed)

    # -- read video --
    vid,noisy,seg_gt = load_video(cfg)
    noisy = append_grid(noisy,cfg.M,cfg.stride0)
    vid = append_grid(vid,cfg.M,cfg.stride0)
    B,T,F,H,W = vid.shape
    # print("vid.shape: ",vid.shape)

    # -- compute flows --
    flows = flow.orun(vid,cfg.flow,ftype="cv2")
    flows = stnls.nn.search_flow(flows.fflow,flows.bflow,cfg.wt,cfg.stride0)
    flows = flows[:,None].round().int()
    # print(flows.shape)

    # -- benchmark --
    timer,memer = ExpTimer(),GpuMemer()
    # with TimeIt(timer,"slic"):
    #     with MemIt(memer,"slic"):
    #         pooled,slic_flows = run_slic(vid,flows,cfg)

    # -- benchmark pooling --
    pooled,seg = {},{}
    pooling_grid = ["ave","max","slic","nls"]
    for pooling_type in pooling_grid:
        with TimeIt(timer,pooling_type):
            with MemIt(memer,pooling_type):
                pooled_p,seg_p = run_pooling(cfg,noisy,flows,pooling_type,
                                             cfg.pooling_ksize,cfg.stride0)
                pooled[pooling_type] = pooled_p
                seg[pooling_type] = seg_p

    # -- view info --
    print(timer)
    print(memer)

    # -- slic_flow to labels for plotting --
    # seg_labels = inds2labels(slic_flows,cfg,H,W)
    seg["slic"] = inds2labels(seg["slic"],cfg,H,W)

    return vid,pooled,seg,seg_gt

def run_pooling(cfg,vid,flows,pooling_type,ksize,stride):
    ws = cfg.ws
    # ksize = ws
    # ksize = stride0
    # stride = 1#stride0//2
    # stride = stride0
    B = vid.shape[0]

    def run_standard(pool_fxn,vid,ksize,stride):
        vid = rearrange(vid,'b t c h w -> (b t) c h w')
        pooled = pool_fxn(vid, ksize, stride=stride)
        pooled = rearrange(pooled,'(b t) c h w -> b t c h w',b=B)
        return pooled

    if pooling_type == "ave":
        pool_fxn = th.nn.functional.avg_pool2d
        pooled,seg = run_standard(pool_fxn,vid,ksize,stride),None
    elif pooling_type == "max":
        pool_fxn = th.nn.functional.max_pool2d
        pooled,seg = run_standard(pool_fxn,vid,ksize,stride),None
    elif pooling_type == "slic":
        pool_fxn = th.nn.functional.avg_pool2d
        pooled,seg = run_slic(vid,flows,cfg)
        # pooled = run_standard(pool_fxn,pooled,ksize,stride)
    elif pooling_type == "nls":
        pool_fxn = th.nn.functional.avg_pool2d
        pooled,seg = run_nls(vid,flows,cfg)
        # pooled = run_standard(pool_fxn,pooled,ksize,stride)
    else:
        raise ValueError("Uknown pooling type.")
    return pooled,seg

def run_nls(vid,flows,cfg):
    # -- compute search window --
    full_ws = True
    B,T,F,H,W = vid.shape
    search = stnls.search.NonLocalSearch(cfg.ws,cfg.wt,cfg.ps,cfg.nls_k,
                                         nheads=1,dist_type="l2",
                                         stride0=cfg.stride0,
                                         self_action="anchor_self",
                                         full_ws=full_ws,itype="int")
    dists,flows_k = search(vid,vid,flows)
    weights = th.softmax(-cfg.softmax_weight*dists,-1)
    # print(weights)

    # -- aggregate --
    ps = cfg.stride0
    # ps = int(cfg.stride0*1.75)
    # agg = stnls.agg.WeightedPatchSum(ps,cfg.stride0,itype="int")
    ps = ps + (1 - ps % 2) # ensure odd
    agg = stnls.agg.PooledPatchSum(ps,cfg.stride0,itype="int")
    vout = agg(vid,weights,flows_k)
    vout = rearrange(vout,'b hd t c h w -> b t (hd c) h w')

    return vout[:,:,:3],None

def run_slic_dev(vid,flows,cfg):
    from stnls.dev.slic import run_slic
    outs = run_slic(vid,cfg.ws,cfg.wt,cfg.ps,cfg.stride0,cfg.full_ws,
                    cfg.M,cfg.softmax_weight,cfg.niters)
    pooled,dists_k,flows_k,s_dists,s_flows = outs
    return pooled[:,:,:3],s_flows

def run_slic(vid,flows,cfg):
    return run_slic_dev(vid,flows,cfg)

    # # # -- compute search window --
    # # B,T,F,H,W = vid.shape
    # # search = stnls.search.NonLocalSearch(cfg.ws,cfg.wt,cfg.ps,cfg.k,
    # #                                      nheads=1,dist_type="l2",
    # #                                      stride0=cfg.stride0,
    # #                                      self_action="anchor_self",
    # #                                      full_ws=cfg.full_ws,itype="int")
    # # dists,flows_k = search(vid,vid,flows)
    # # # print(dists.shape,flows_k.shape)
    # # inds = stnls.utils.misc.flow2inds(flows_k,cfg.stride0)
    # # # print(inds.shape)
    # # # print(inds[0,0,0,:4,:4])

    # # # print(inds[0,0,0,:2,:2,0])
    # # # print(inds[0,0,0,-4:,-4:,0])

    # # # -- scattering top-K=1 --
    # # K0 = 1
    # # # gather_weights = th.softmax(-dists,-1)
    # # gather_weights = dists
    # # # timer,memer = ExpTimer(),GpuMemer()
    # # # with TimeIt(timer,"labels"):
    # # #     with MemIt(memer,"labels"):
    # # print(cfg.full_ws)
    # # names,labels = stnls.agg.scatter_labels(flows,flows_k,cfg.ws,cfg.wt,
    # #                                         cfg.stride0,cfg.stride1,H,W,cfg.full_ws)
    # # # print(timer,memer)
    # # # print(labels.min().item(),labels.max().item())
    # # # print("[scattering]: ",gather_weights.shape,flows_k.shape,labels.shape)
    # # # print(gather_weights[0,0,0,0,0])
    # # # print(labels[0,0,0])
    # # gather_labels = labels.reshape_as(gather_weights)
    # # scatter_weights = stnls.agg.scatter_tensor(gather_weights,flows_k,labels,
    # #                                            cfg.stride0,cfg.stride1,H,W)
    # # scatter_flows_k = stnls.agg.scatter_tensor(flows_k,flows_k,labels,
    # #                                            cfg.stride0,cfg.stride1,H,W)
    # # scatter_labels = stnls.agg.scatter_tensor(gather_labels,flows_k,labels,
    # #                                           cfg.stride0,cfg.stride1,H,W,
    # #                                           invalid=-th.inf)
    # # print("[a]: ",scatter_flows_k.shape,flows_k.shape,scatter_labels.shape)


    # # # -- checking in --
    # # # nH,nW = H//cfg.stride1,W//cfg.stride1
    # # # shape_str = 'b hd (t nh nw) k tr -> b hd t nh nw k tr'
    # # # scatter_flows_k = rearrange(scatter_flows_k,shape_str,nh=nH,nw=nW)
    # # # shape_str = 'b hd (t nh nw) k -> b hd t nh nw k'
    # # # scatter_weights = rearrange(scatter_weights,shape_str,nh=nH,nw=nW)
    # # # print(scatter_weights.shape,scatter_flows_k.shape)
    # # # print(scatter_weights[0,0,0,-3:,-3:])
    # # # print(scatter_flows_k[0,0,0,-3:,-3:])
    # # # exit()

    # # both = th.cat([scatter_weights[...,None],scatter_flows_k],-1)
    # # # print(both.shape)
    # # # print(both[0,0,0])
    # # # exit()

    # # # -- topk --
    # # scatter_flows_k = -scatter_flows_k
    # # s_weight,s_flows_k,s_labels = stnls.agg.scatter_topk(scatter_weights,scatter_flows_k,
    # #                                                      scatter_labels,K0,
    # #                                                      descending=False)
    # # # print(s_flows_k.shape,s_labels.shape)
    # # # s_flows_k = s_flows_k.int()
    # # # print(th.any(s_weight<-1000).item())
    # # # print(th.any(s_flows_k<-1000).item(),th.any(s_flows_k>1000).item())
    # # # print(th.where(s_flows_k[...,0]<-1000))
    # # # print(s_weight[th.where(s_flows_k[...,0]<-1000)])
    # # # print(th.where(s_flows_k<-1000))
    # # # print(s_weight.shape)
    # # # print(s_flows_k.shape)
    # # # print(s_weight[0,0,:3])
    # # # print(s_weight[0,0,100:103])
    # # # print(s_weight[0,0,-3:])
    # # # print(s_flows_k[0,0,:3])
    # # # print(s_flows_k[0,0,100:103])
    # # # print(s_flows_k[0,0,-3:])
    # # both = th.cat([s_weight[...,None],s_flows_k],-1)
    # # # print(both.shape)
    # # # print(both[0,0,:,:])
    # # # exit()

    # # pooled = slic_pooling(vid,s_weight,s_flows_k,s_labels,
    # #                       cfg.ps,cfg.stride0,cfg.stride1,K0,cfg.softmax_weight)
    # # # pooled = None

    # # # print(pooled.shape)

    # return pooled[:,:,:3],s_flows_k


def slic_pooling(vid,s_weights,s_flows_k,s_labels,ps,stride0,stride1,K0,softmax_weight):

    # -- prepare weights and flows --
    B,T,F,H,W = vid.shape
    HD = s_weights.shape[1]
    s_weights = s_weights.reshape(B,HD,T,H,W,K0)
    s_flows_k = s_flows_k.reshape(B,HD,T,H,W,K0,3)
    s_labels = s_labels.reshape(B,HD,T*H*W,-1)

    # -- run scatters --
    # print("pooling: ",s_weights.shape,s_flows_k.shape,s_labels.shape)
    weights = stnls.agg.scatter_tensor(s_weights,s_flows_k,s_labels,
                                       stride1,stride0,H,W)
    flows_k = stnls.agg.scatter_tensor(s_flows_k,s_flows_k,s_labels,
                                       stride1,stride0,H,W)
    # print(weights.shape,flows_k.shape)

    # -- reshape --
    K = weights.shape[-1]
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1
    weights = weights.reshape(B,HD,T,nH,nW,K)
    flows_k = flows_k.reshape(B,HD,T,nH,nW,K,3)

    # -- renormalize weights --
    # print(weights)
    weights = th.softmax(-softmax_weight*weights,-1)
    # print(weights)
    # weights = weights / th.sum(weights,-1,keepdim=True)
    # print(th.sum(weights,-1))
    # print(weights[0,0,:2,:2])
    # print(weights[0,0,-2:,-2:])

    # -- aggregate --
    ps = stride0
    ps = ps + (1 - ps % 2) # ensure odd
    # ps = stride0
    # agg = stnls.agg.WeightedPatchSum(ps,stride0,itype="int")
    # print(th.sum(weights,-1))
    # print(ps,stride0)
    agg = stnls.agg.PooledPatchSum(ps,stride0,itype="int")
    # vid = th.ones_like(vid)
    # print("weights [min,max]: ",weights.min().item(),weights.max().item())
    vout = agg(vid,weights,flows_k)
    vout = rearrange(vout,'b hd t c h w -> b t (hd c) h w')
    # print("vin [min,max]: ",vid[...,:3,:,:].min().item(),vid[...,:3,:,:].max().item())
    # print("vout [min,max]: ",vout[...,:3,:,:].min().item(),vout[...,:3,:,:].max().item())
    # # vout = None
    # print("vout.shape,vid.shape: ",vout.shape,vid.shape)

    return vout

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


    # # -- info --
    # print(seg_labels.shape)
    # print(s_inds.shape)
    # inds = s_inds
    # # exit()
    # print("-"*20)
    # print("-"*20)
    # print("-"*20)
    # print(inds[0,0,H//2+1,29])
    # print("-"*20)
    # print(inds[0,0,H//2,29])
    # print("-"*20)
    # print(inds[0,0,H//2-1,29])
    # print("-"*20)
    # print(inds[0,0,H//2+2,29])


    # print("-"*20)
    # print("-"*20)
    # print("-"*20)
    # print(inds[0,0,H//2+6,29])
    # print("-"*20)
    # print(inds[0,0,H//2+7,29])
    # print("-"*20)
    # print(inds[0,0,H//2+5,29])
    # print("-"*20)
    # print(inds[0,0,H//2+8,29])
    # print("-="*20)


    # -- fill invalid --
    valid = th.logical_and(seg_labels<10000,seg_labels>-10000)
    S = seg_labels[th.where(valid)].max()
    seg_labels[th.where(~valid)] = S+1

    # -- view --
    print(seg_labels.shape)
    print(seg_labels[0,0,-5:,-5:])

    return seg_labels

def labels2masks(labels):
    S = labels.max()+1
    masks = th.zeros([S,]+list(labels.shape),dtype=th.bool).to(labels.device)
    for si in range(S):
        masks[si] = labels==si
    return masks

def ensure_size(img,tH,tW):
    B = img.shape[0]
    img = rearrange(img,'b t c h w -> (b t) c h w')
    img = TF.resize(img,(tH,tW),InterpolationMode.NEAREST)
    img = rearrange(img,'(b t) c h w -> b t c h w',b=B)
    return img

def main():

    # -- config --
    cfg = edict()
    cfg.seed = 123
    # cfg.dset = "tr"
    # cfg.dname = "bsd500"
    cfg.dset = "tr"
    cfg.dname = "iphone_sum2023"
    # cfg.dset = "val"
    # cfg.dname = "set8"
    # cfg.isize = "540_540"
    # cfg.isize = "260_260"
    # cfg.isize = "256_256"
    # cfg.isize = "128_128"
    # cfg.isize = "816_1216"
    # cfg.isize = "640_480"
    # cfg.isize = None
    # cfg.isize = "400_400"
    # cfg.isize = "300_300"
    # cfg.vid_name = "sunflower"
    # cfg.vid_name = "hypersmooth"
    cfg.vid_name = "quilt"
    # cfg.vid_name = "12074"
    cfg.ntype = "g"
    cfg.sigma = .001
    cfg.nframes = 2
    cfg.flow = False
    cfg.full_ws = False
    cfg.wt = 0
    # cfg.stride0 = 8
    # cfg.ws = 15
    cfg.stride0 = 10
    cfg.ws = cfg.stride0*2-1
    # cfg.ws = 11

    # cfg.ws = 21
    # cfg.stride0 = 3
    # cfg.ws = 3
    # if cfg.ws == 1: cfg.ws += 1
    cfg.stride1 = 1
    cfg.k = -1#cfg.ws*cfg.ws
    cfg.nls_k = 8
    cfg.ps = 1
    cfg.M = 0.3
    cfg.pooling_ksize = 1
    cfg.softmax_weight = 10.
    cfg.niters = 5

    # -- run slic --
    vid,pooled,segs,seg_gt = run_exp(cfg)
    vid = vid[:,:,:3]
    # pooled = pooled[:,:,:3]
    labels = segs['slic']
    # print(vid.shape,pooled.shape)

    # -- save output --
    vid = (255*vid).type(th.uint8)
    B,T,F,H,W = vid.shape
    seg = []
    for bi in range(B):
        for ti in range(T):
            # mask = labels2masks(labels[bi,ti]).to(vid.device)
            # print(vid[bi,ti].shape,mask.shape)
            vid_bt = rearrange(vid[bi,ti].cpu().numpy(),'tr h w -> h w tr')
            labels_bt = labels[bi,ti].cpu().numpy()
            seg_bt = mark_boundaries(vid_bt,labels_bt)
            seg_bt = rearrange(seg_bt,'h w tr -> tr h w')
            # seg_bt = draw_segmentation_masks(vid[bi,ti].cpu(),mask.cpu())
            seg.append(th.tensor(seg_bt))
    seg = th.stack(seg).view(B,T,F,H,W)
    tH,tW = 128,128
    seg = ensure_size(seg,tH,tW)
    vid = ensure_size(vid,tH,tW)
    print("[slic] seg.shape: ",seg.shape)
    vid_io.save_video(vid,"output/slic","clean")
    vid_io.save_video(seg,"output/slic","ex_n%d"%cfg.niters)

    # -- mark grid --
    vid[...,0,::cfg.stride0,::cfg.stride0] = 0
    vid[...,1,::cfg.stride0,::cfg.stride0] = 0
    vid[...,2,::cfg.stride0,::cfg.stride0] = 255.
    tH,tW = 252,252
    vid = vid[...,:2*cfg.stride0+1,:2*cfg.stride0+1]
    # vid[...,-1,-1] = 1.
    print(cfg.stride0)
    print("[a] vid.shape: ",vid.shape)
    vid = ensure_size(vid,tH,tW)
    print("[b] vid.shape: ",vid.shape)
    vid_io.save_video(vid,"output/slic","clean_marked")

    # # -- save --
    # H = vid.shape[-2]
    # # print(seg.shape)
    # vid_io.save_video(seg[:,[0],:,H//2+1:H//2+7,28:32],
    #                   "output/slic","ex_n%d"%cfg.niters)

    # -- pooled --
    tH,tW = 128,128
    for ptype in pooled:
        print(pooled[ptype].type,pooled[ptype].shape,pooled[ptype][:,:,:3].max())
        vid = pooled[ptype][:,:,:3]
        vid = ensure_size(vid,tH,tW)
        vid_io.save_video(vid,"output/slic_pooled/",ptype)


if __name__ == "__main__":
    main()
