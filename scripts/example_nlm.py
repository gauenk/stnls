"""

An example script for a non-local means.

Still in progress... scatter/gather coming soon.

"""

# -- imports --
import torch as th
import dnls
from einops import rearrange

# -- load video --
sigma = 30.
device = "cuda:0"
vid = dnls.testing.data.load_burst("./data","davis_baseball_64x64",ext="jpg")
vid = th.from_numpy(vid).to(device)
noisy = vid + sigma * th.randn_like(vid)

# -- params --
ps = 7 # patch size
pt = 1 # patch size across time
stride = 1 # spacing between patch centers
dilation = 1 # spacing between kernels
batch_size = 128 # num of patches per batch
coords = [4,8,60,50] # interior rectangle to processes (top,left,btm,right)
vshape = vid.shape
t,c,h,w = vid.shape

# -- search params --
flow = None # no flow
k = 25 # number of neighbors
ws = 10 # spatial-search space in each 2-dim direction
wt = 3 # time-search space across in each fwd-bwd direction
chnls = 3 # number of channels to use for search

# -- compute interior square size --
sq_h = coords[2] - coords[0]
sq_w = coords[3] - coords[1]
sq_hw = sq_h * sq_w

# -- init iunfold and ifold --
scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dilation)
gather_nl = dnls.gather.GatherNl(vid.shape)

# -- compute number of batches --
n_h = (sq_h-1)//stride+1
n_w = (sq_w-1)//stride+1
n_total = t * n_h * n_w
nbatches = (n_total-1) // batch_size + 1

# -- example function --
def apply_fxn(patches_nl_i,dists):
    """
    patches_i.shape = (batch_size,k,pt,c,ps,ps)
    with k == 1 and pt == 1
    k = number of neighbors
    pt = temporal patch size
    """
    lam = .5
    weights = th.exp(-lam * (dists/(255.**2)))
    weights = rearrange(weights,'b k -> b k 1 1 1 1')
    weights /= th.sum(weights,1,True)
    wpatches = patches_nl_i * weights
    wpatches = th.sum(wpatches,1,True)
    patches_nl_i[:,:] = wpatches[:,:]
    return patches_nl_i

# -- iterate over batches --
for batch in range(nbatches):

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
    patches_nl_i = scatter_nl(noisy,inds)

    # -- process --
    patches_mod_i = apply_fxn(patches_nl_i,dists)

    # -- regroup --
    gather_nl(patches_mod_i,dists,inds)

# -- save modded video --
deno,weights = gather_nl.vid,gather_nl.wvid
deno /= weights
dnls.testing.data.save_burst(noisy,"./output/","noisy")
dnls.testing.data.save_burst(deno,"./output/","deno")
