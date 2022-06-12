"""

An example script for a non-local means.

Still in progress... scatter/gather coming soon.

"""

# -- imports --
import tqdm
import dnls
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- load video --
sigma = 30.
device = "cuda:1"
th.cuda.set_device(device)
vid = dnls.testing.data.load_burst("./data","davis_baseball_64x64",ext="jpg")
vid = th.from_numpy(vid).to(device)
noisy = vid + sigma * th.randn_like(vid)

# -- params --
vshape = vid.shape
t,c,h,w = vid.shape
ps = 5 # patch size
pt = 1 # patch size across time
stride = 1 # spacing between patch centers
dilation = 1 # spacing between kernels
batch_size = 1024 # num of patches per batch
nkeep = -1
coords = [0,0,h,w] # full image
# coords = [4,8,60,50] # interior rectangle to processes (top,left,btm,right)

# -- helper --
def yuv2rgb(burst):
    # -- weights --
    t,c,h,w = burst.shape
    w = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)/np.sqrt(3)]
    # -- copy channels --
    y,u,v = burst[:,0].clone(),burst[:,1].clone(),burst[:,2].clone()
    # -- yuv -> rgb --
    burst[:,0,...] = w[0] * y + w[1] * u + w[2] * 0.5 * v
    burst[:,1,...] = w[0] * y - w[2] * v
    burst[:,2,...] = w[0] * y - w[1] * u + w[2] * 0.5 * v
def rgb2yuv(burst):
    # -- weights --
    t,c,h,w = burst.shape
    # -- copy channels --
    r,g,b = burst[:,0].clone(),burst[:,1].clone(),burst[:,2].clone()
    # -- yuv -> rgb --
    weights = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)*2./np.sqrt(3)]
    # -- rgb -> yuv --
    burst[:,0,...] = weights[0] * (r + g + b)
    burst[:,1,...] = weights[1] * (r - b)
    burst[:,2,...] = weights[2] * (.25 * r - 0.5 * g + .25 * b)

# -- convert --
rgb2yuv(noisy)

# -- search params --
flow = None # no flow
k = 100 # number of neighbors
ws = 10 # spatial-search space in each 2-dim direction
wt = 5 # time-search space across in each fwd-bwd direction
chnls = 1 # number of channels to use for search
verbose = True

# -- compute interior square size --
sq_h = coords[2] - coords[0]
sq_w = coords[3] - coords[1]
sq_hw = sq_h * sq_w

# -- init iunfold and ifold --
scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dilation,device=device)
gather_nl = dnls.gather.GatherNl(vid.shape,device=device)

# -- compute number of batches --
n_h = (sq_h-1)//stride+1
n_w = (sq_w-1)//stride+1
n_total = t * n_h * n_w
nbatches = (n_total-1) // batch_size + 1

# -- example function --
def viz_cov(patches_nl_i,dists):
    p = rearrange(patches_nl_i[0],'k pt c ph pw -> c k (pt ph pw)')
    p /= 255.
    mu = p.mean(0,True)
    p_zc = p - mu
    # print(p_zc.transpose(2,1).shape)
    C = th.matmul(p_zc.transpose(2,1), p_zc)
    C = th.abs(C)
    for ci in range(3):
        Ci = repeat(C[ci],'h w -> c h w',c=3)
        Ci /= Ci.max()
        Ci = Ci * 255.
        dnls.testing.data.save_image(Ci,"patch_cov_%d_%d.png" % (k,ci))
    exit(0)

def apply_fxn(npatches,bpatches,dists,step):
    """
    n = noisy, b = basic
    npatches.shape = (batch_size,k,pt,c,ps,ps)
    with k == 1 and pt == 1
    k = number of neighbors
    pt = temporal patch size
    """

    # -- params --
    rank = 39
    thresh = 2.7 if step == 1 else 0.7
    sigma2 = (sigma/255.)**2
    sigmab2 = sigma2 if step == 1 else (sigma/(255.*10))**2

    # -- reshape --
    b,k,pt,c,ph,pw = npatches.shape
    shape_str = 'b k pt c ph pw -> b c k (pt ph pw)'
    npatches = rearrange(npatches,shape_str)
    bpatches = rearrange(bpatches,shape_str)

    # -- normalize --
    bpatches /= 255.
    npatches /= 255.
    b_centers = bpatches.mean(dim=2,keepdim=True)
    centers = npatches.mean(dim=2,keepdim=True)
    c_bpatches = bpatches - b_centers
    c_npatches = npatches - centers

    # -- group batches --
    shape_str = 'b c k p -> (b c) k p'
    c_bpatches = rearrange(c_bpatches,shape_str)
    c_npatches = rearrange(c_npatches,shape_str)
    centers = rearrange(centers,shape_str)
    bsize,num,pdim = c_npatches.shape

    # -- flat batch & color --
    C = th.matmul(c_bpatches.transpose(2,1),c_bpatches)/num
    eigVals,eigVecs = th.linalg.eigh(C)
    eigVals = th.flip(eigVals,dims=(1,))[...,:rank]
    eigVecs = th.flip(eigVecs,dims=(2,))[...,:rank]

    # -- denoise eigen values --
    eigVals = rearrange(eigVals,'(b c) r -> b c r',b=b)
    th_sigmab2 = th.FloatTensor([sigmab2]).reshape(1,1,1)
    th_sigmab2 = th_sigmab2.to(eigVals.device)
    emin = th.min(eigVals,th_sigmab2)
    eigVals -= emin
    eigVals = rearrange(eigVals,'b c r -> (b c) r')

    # -- filter coeffs --
    geq = th.where(eigVals > (thresh*sigma2))
    leq = th.where(eigVals <= (thresh*sigma2))
    eigVals[geq] = 1. / (1. + sigma2 / eigVals[geq])
    eigVals[leq] = 0.

    # -- denoise patches --
    bsize = c_npatches.shape[0]
    Z = th.matmul(c_npatches,eigVecs)
    R = eigVecs * eigVals[:,None]
    tmp = th.matmul(Z,R.transpose(2,1))
    c_npatches[...] = tmp

    # -- add patches --
    c_npatches[...] += centers

    # -- reshape --
    shape_str = '(b c) k (pt ph pw) -> b k pt c ph pw'
    patches = rearrange(c_npatches,shape_str,b=b,c=c,ph=ph,pw=pw)
    patches *= 255.
    patches = patches.contiguous()
    return patches

#
# -- Step 1 --
#

for batch in tqdm.tqdm(range(nbatches)):

    # -- batch info --
    index = min(batch_size * batch,n_total)
    batch_size_i = min(batch_size,n_total-index)

    # -- get patches --
    queries = dnls.utils.inds.get_iquery_batch(index,batch_size_i,
                                               stride,coords,t,h,w,device)
    dists,inds = dnls.simple.search.run(noisy,queries,flow,k,
                                        ps,pt,ws,wt,chnls,
                                        stride=stride,dilation=dilation)
    # -- get patches --
    noisy_patches_i = scatter_nl(noisy,inds)
    basic_patches_i = scatter_nl(noisy,inds)

    # -- process --
    patches_mod_i = apply_fxn(noisy_patches_i,basic_patches_i,dists,1)

    # -- regroup --
    patches_2_agg = patches_mod_i[:,:nkeep].contiguous()
    zeros_2_agg = th.zeros_like(dists)[:,:nkeep].contiguous()
    inds_2_agg = inds[:,:nkeep].contiguous()
    gather_nl(patches_2_agg,zeros_2_agg,inds_2_agg)


# -- post processing --
basic,weights = gather_nl.vid,gather_nl.wvid
basic /= weights
zargs = th.where(weights == 0)
basic[zargs] = noisy[zargs]

# -- setup for 2nd step --
k = 60
nkeep = 60
gather_nl = dnls.gather.GatherNl(vid.shape,device=device)

#
# -- Step 2 --
#

for batch in tqdm.tqdm(range(nbatches)):

    # -- batch info --
    index = min(batch_size * batch,n_total)
    batch_size_i = min(batch_size,n_total-index)

    # -- get patches --
    queries = dnls.utils.inds.get_iquery_batch(index,batch_size_i,
                                               stride,coords,t,h,w,device)
    dists,inds = dnls.simple.search.run(basic,queries,flow,k,
                                        ps,pt,ws,wt,chnls,
                                        stride=stride,dilation=dilation)
    # -- get patches --
    noisy_patches_i = scatter_nl(noisy,inds)
    basic_patches_i = scatter_nl(basic,inds)

    # -- process --
    patches_mod_i = apply_fxn(noisy_patches_i,basic_patches_i,dists,2)

    # -- regroup --
    patches_2_agg = patches_mod_i[:,:nkeep].contiguous()
    zeros_2_agg = th.zeros_like(dists)[:,:nkeep].contiguous()
    inds_2_agg = inds[:,:nkeep].contiguous()
    gather_nl(patches_2_agg,zeros_2_agg,inds_2_agg)

# -- format final image --
deno,weights = gather_nl.vid,gather_nl.wvid
deno /= weights
zargs = th.where(weights == 0)
deno[zargs] = basic[zargs]
yuv2rgb(deno)

#
# -- Save Results --
#

# -- compute psnr and save --
psnr = -10 * th.log10(th.mean((deno/255. - vid/255.)**2)).item()
print("PSNR: %2.3f" % psnr)
dnls.testing.data.save_burst(noisy/255.,"./output/","noisy")
dnls.testing.data.save_burst(deno/255.,"./output/","deno")
