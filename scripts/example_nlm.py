"""

An example script for a non-local means.

"""

# -- imports --
import torch as th
import dnls
from einops import rearrange,repeat

# -- params --
ps = 7 # patch size
pt = 1 # patch size across time
stride0 = 1 # spacing between patch centers to search
stride1 = 1 # spacing between patch centers when searching
dilation = 1 # spacing between kernels
batch_size = 512 # num of patches per batch
sigma = 30.
device = "cuda:0"
seed = 123

# -- search params --
flow = None # no flow
k = 7 # number of neighbors
ws = 15 # spatial-search space in each 2-dim direction
wt = 3 # time-search space across in each fwd-bwd direction
chnls = 3 # number of channels to use for search
use_rand,nreps = False,1
# use_rand,nreps = True,10

# -- set seed --
th.manual_seed(seed)

# -- load video --
vid = dnls.testing.data.load_burst("./data","davis_baseball_64x64",ext="jpg")
vid = th.from_numpy(vid).to(device)
vid = th.cat([vid,vid],-1)
vid = th.cat([vid,vid],-2)
clean = vid
noisy = vid + sigma * th.randn_like(vid)
vshape = vid.shape
t,c,h,w = vid.shape

# -- init search, unfold and fold --
search = dnls.search.init("l2_with_index",None,None,k,ps,pt,ws,wt,chnls=chnls,
                          stride0=stride0,stride1=stride1,dilation=dilation)
unfoldk = dnls.UnfoldK(ps,pt,dilation=dilation,device=device)
foldk = dnls.FoldK(vid.shape,use_rand=use_rand,nreps=nreps,device=device)

# -- batching info --
nh,nw = dnls.utils.get_nums_hw(vid.shape,stride0,ps,dilation)
ntotal = t * nh * nw
nbatches = (ntotal-1) // batch_size + 1

# -- xfor img --
noisy /= 255.
clean /= 255.

def apply_nlm(patches_nl_i,dists):
    """
    patches_i.shape = (batch_size,k,pt,c,ps,ps)
    with k == 1 and pt == 1
    k = number of neighbors
    pt = temporal patch size
    """
    lam = 25.
    weights = th.exp(-lam * (dists))
    weights = rearrange(weights,'b k -> b k 1 1 1 1')
    weights /= th.sum(weights,1,True)
    wpatches = patches_nl_i * weights
    wpatches = th.sum(wpatches,1,True)
    patches_nl_i[:,:] = wpatches[:,:]
    return patches_nl_i,weights

# -- iterate over batches --
for batch in range(nbatches):

    # -- get batch --
    index = min(batch_size * batch,ntotal)
    nbatch_i = min(batch_size,ntotal-index)

    # -- search patches --
    dists,inds = search(noisy,index,nbatch_i)
    dists /= (c*ps*ps)

    # -- get patches --
    patches_nl_i = unfoldk(noisy,inds)

    # -- process --
    patches_mod_i,weights = apply_nlm(patches_nl_i,dists)

    # -- regroup --
    foldk(patches_mod_i,weights.squeeze(),inds)


# -- unpack aggregate --
deno,weights = foldk.vid,foldk.wvid
deno /= weights
zargs = th.where(weights == 0)
deno[zargs] = noisy[zargs]
# print(th.where(th.isnan(deno)))

# -- print psnr --
psnr = -10 * th.log10(th.mean((deno - vid)**2)).item()
print("PSNR: %2.3f" % psnr)
for i in range(3):
    psnr = -10 * th.log10(th.mean((deno[:,i] - vid[:,i])**2)).item()
    print("[%d] PSNR: %2.3f" % (i,psnr))

# -- save denoised --
dnls.testing.data.save_burst(noisy,"./output/","noisy")
dnls.testing.data.save_burst(deno,"./output/","deno")
