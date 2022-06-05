
# -- imports --
import torch as th
import dnls
from einops import rearrange

# -- load video --
vid = dnls.testing.data.load_burst("./data","davis_baseball_64x64",ext="jpg")
vid = th.from_numpy(vid).to("cuda:0")

# -- params --
ps = 5
stride = 2
dilation = 3
batch_size = 128
vshape = vid.shape
t,c,h,w = vid.shape
coords = [4,8,60,50] # any interior rectangle (top,left,btm,right)

# -- compute interior square size --
sq_h = coords[2] - coords[0]
sq_w = coords[3] - coords[1]
sq_hw = sq_h * sq_w

# -- init iunfold and ifold --
fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dilation)
unfold_nl = dnls.iunfold.iUnfold(ps,coords,stride=stride,dilation=dilation)

# -- compute number of batches --
n_h = (sq_h-1)//stride+1
n_w = (sq_w-1)//stride+1
n_total = t * n_h * n_w
nbatches = (n_total-1) // batch_size + 1

# -- example function --
def apply_fxn(patches_i):
    """
    patches_i.shape = (batch_size,k,pt,c,ps,ps)
    with k == 1 and pt == 1
    k = number of neighbors
    pt = temporal patch size
    """
    patches_i[:,:,:,0,:,:] = 0.
    return patches_i

# -- iterate over batches --
for batch in range(nbatches):
    index = min(batch_size * batch,n_total)
    batch_size_i = min(batch_size,n_total-index)
    patches_i = unfold_nl(vid,index,batch_size_i)
    patches_mod_i = apply_fxn(patches_i)
    vid_nl = fold_nl(patches_mod_i,index)

# -- save modded video --
vid = fold_nl.vid
vid /= vid.max()
vid = dnls.testing.data.save_burst(vid,"./output/","example")


