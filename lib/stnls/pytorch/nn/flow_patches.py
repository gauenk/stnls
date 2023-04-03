
"""

vid[:,0]
[vid[:,1],...,vid[:,T]]

vid[:,1]
[vid[:,0],...,vid[:,T]]

...

vid[:,T]
[vid[:,0],...,vid[:,T-1]]


"""


"""

Get/Compare the patches from optical flow to assess their quality for nls

"""

import torch as th
import dnls
from einops import rearrange
from easydict import EasyDict as edict

# -- api --
def get_patches(vid,flows,ps):

    patches = edict()
    UnfoldK = dnls.UnfoldK(ps)
    for f in ["fflow","bflow"]:

        # -- non-local indices --
        inds = flow2inds(flows[f],f)

        # -- patches --
        patches[f] = UnfoldK(vid,inds)

    return patches


# -- api --
def get_mse(vid,flows,ps):
    patches = get_patches(vid,flows,ps)
    mse = edict()
    for k in patches:
        mse[k] = 0
        mse[k] += th.mean((patches[k][:,:,0] - patches[k][:,:,1])**2).item()
        # mse[k] += th.mean((patches[k][:,:,0] - patches[k][:,:,2])**2)
        # mse[k] /= 2.
    return mse

# -- api --
# def get_warp_2f(vid,fflow,bflow,ws=3,ps=7,stride0=1):
def tmp():

    # -- compute metrics --
    flows = edict({"fflow":fflow,"bflow":bflow})
    UnfoldK = dnls.UnfoldK(1)
    for f in ["fflow","bflow"]:

        # -- non-local indices --
        inds = flow2inds(flows[f],f)
        print(inds.shape)

        # -- patches --
        patches = UnfoldK(vid,inds[None,:])

        # -- fold --
        ps2 = patches.shape[-1]//2
        pix = patches[...,ps2,ps2]
        warp = rearrange(pix,'')

        # # -- compute patch deltas --
        # mse = th.mean((patches[:,:,0] - patches[:,:,1])**2)
        # print(mse)
        # records["%s_mse" % f] = mse.cpu().numpy().item()
        # records["%s_psnr" % f] =  -10 * th.log10(mse).cpu().numpy().item()

        # # -- save example --
        # root = "output/warps/flow_%s/%s" % (f,cfg.vid_name)
        # warp_image(patches,clean,root,cfg)
    return warp

def get_warp_2f(vid,fflow,bflow,ws=3,ps=5,stride0=1,warp_ps=4,k=4):
    sim_fwd_bwd = dnls.warp.SimFwdBwd(warp_ps,ws,ps,k,stride0)
    fwd,bwd = sim_fwd_bwd(vid,fflow,bflow)
    return fwd,bwd

def tmp(vid,fflow,bflow,ws=2,ps=7,stride0=2):

    # -- unpack --
    vid = vid.contiguous()
    zflow = th.zeros_like(fflow[:,[0]])
    fflow = th.cat([fflow,zflow],1)
    bflow = th.cat([zflow,bflow],1)
    T = vid.shape[1]
    wt = 1
    warp_ps = 4

    # -- search --
    dists,inds = dnls.search.nls(vid,vid,fflow,bflow,
                                 ws,wt,ps,-1,dist_type="l2",
                                 stride0=stride0,
                                 anchor_self=False,use_adj=True)

    # -- top-K across time --
    # dists,inds = dnls.nn.topk(dists,inds,8,dim=3,anchor=False)
    dists,inds = dnls.nn.topk_time(dists,inds,4,ws,dim=3,anchor=False)
    # print("inds.shape: ",inds.shape)
    # inds_ref = inds[:,:,:,0]
    # inds0 = inds[:,:,:,1::2]
    # inds1 = inds[:,:,:,2::2]
    # # print(inds_ref.shape)
    # # print(inds0.shape)
    # # print(inds1.shape)
    # # print(inds.shape)
    # inds = inds[:,:,:,:8]
    # print(inds[0,0,0])

    # -- pick across time --
    inds_ref = inds[:,:,:,[0]]
    T = vid.shape[1]
    B,HD,Q,_,_ = inds.shape
    K = 4
    Qt = Q//T
    inds_fwd = th.zeros((T,B,1,Qt,K,3),device=inds.device,dtype=inds.dtype)
    inds_bwd = th.zeros((T,B,1,Qt,K,3),device=inds.device,dtype=inds.dtype)
    for t in range(T):

        # -- fwd (t to t+1) --
        if t < (T-1):

            # -- get inds @ t+1 --
            args_t = th.where(inds_ref[...,0] == t+1)
            inds_t = inds[:,:,args_t[2]]

            # -- best match in t --
            args_k = th.where(inds_t[...,0] == t) # should be one each
            for i in range(3):
                inds_fwd[t][...,i] = inds_t[...,i][args_k].reshape(1,1,Qt,K)

        # -- bwd (t-1 to t) --
        if t > 0:

            # -- get inds @ t+1 --
            args_t = th.where(inds_ref[...,0] == t)
            inds_t = inds[:,:,args_t[2]]

            # -- best match in t --
            args_k = th.where(inds_t[...,0] == t-1) # should be one each
            for i in range(3):
                inds_bwd[t][...,i] = inds_t[...,i][args_k].reshape(1,1,Qt,K)

    # -- reshape --
    # print("inds_fwd.shape: ",inds_fwd.shape)
    inds_fwd = rearrange(inds_fwd,'t b hd q k tr -> b hd (t q) k tr')
    inds_bwd = rearrange(inds_bwd,'t b hd q k tr -> b hd (t q) k tr')
    fwd = warp_frame(vid,inds_fwd,warp_ps,stride0)
    bwd = warp_frame(vid,inds_bwd,warp_ps,stride0)
    # fwd[:,:] = 0
    # fwd[:,:] = 0
    fwd[:,-1] = 0
    bwd[:,0] = 0
    # print(fwd.shape,bwd.shape)

    # -- format --
    # fwd = rearrange(fwd[:,:-1],'b t c h w -> b')
    # bwd = rearrange(bwd[:,:-1],'b t c h w -> b')

    return bwd,fwd

def warp_frame(vid,inds,ps,stride0):

    # -- patches --
    print(inds[0,0,:3,:])
    print(inds[0,0,-3:,:])
    UnfoldK = dnls.UnfoldK(ps,adj=0)
    patches = UnfoldK(vid,inds[:,0]) # nheads == 1
    print(patches.shape)
    K = patches.shape[2]
    # assert K == 1
    patches = rearrange(patches,'b q k pt c ph pw -> (b k) q 1 pt c ph pw')
    print(patches.shape)

    # -- fold --
    B = vid.shape[0]
    vshape = [B*K,] + list(vid.shape[1:])
    fold = dnls.iFoldz(vshape,None,stride=stride0)
    fold(patches)
    warp = fold.vid / fold.zvid
    print(warp.shape)

    # -- stack channels --
    warp = rearrange(warp,'(b k) t c h w -> b t (k c) h w',k=K)

    return warp

def flow2inds(flow,fdir):

    # -- init --
    B,T,_,H,W = flow.shape
    raster = get_raster_inds(flow)
    inds = th.zeros((B,T-1,2,3,H,W),device="cuda:0")
    # print(flow[0,:,:3,:3])
    flow = th.flip(flow,(2,))
    # print(flow[0,:,:3,:3])

    # -- frame 0 is identity --
    for t in range(T-1):

        if fdir == "fflow":
            t_curr = t
            t_next = t+1
        else:
            t_curr = t+1
            t_next = t

        inds[:,t,0,0] = t_curr
        inds[:,t,0,1:] = raster

        inds[:,t,1,0,:,:] = t_next
        inds[:,t,1,1:,:,:] = flow[:,t_curr]+raster+0.5

    inds = rearrange(inds,'b s k tr h w -> b (s h w) k tr').contiguous()
    inds = inds.type(th.int32)

    # -- clip to legal region --
    # print("p: ",inds.max(),inds.min())
    # print(inds[...,0].min(),inds[...,0].max())
    inds[...,1] = th.clamp(inds[...,1],min=0,max=H-1)
    inds[...,2] = th.clamp(inds[...,2],min=0,max=W-1)
    # print("post: ",inds.max(),inds.min())

    return inds

def get_raster_inds(vid):
    B,T,_,H,W = vid.shape
    inds = th.zeros((B,2,H,W),device="cuda:0")
    inds[:,0,:,:] = th.arange(H)[:,None]
    inds[:,1,:,:] = th.arange(W)[None,:]
    # print(inds[:,:3,:3])
    # print(inds[:,-3:,-3:])
    return inds

