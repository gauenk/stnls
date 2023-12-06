"""

An efficient implementation of Video Non-Local Bayes

"""

from stnls.utils import extract_pairs

def extract_config(cfg,restrict=True):
    pairs = {"ws":-1,"wt":-1,"ps":3,"k":10,
             "dist_type":"l2", "stride0":1, "stride1":1,
             "dilation":1, "pt":1,"itype":"float","topk_mode":"all",}
    return extract_pairs(cfg,pairs,restrict=restrict)

def run_vnlb(cfg,vid):

    # -- extract options --
    cfg = extract_config(cfg)

    # -- convert --
    rgb2yuv(noisy)

    # -- init iunfold and ifold --
    search = stnls.search.NonLocalSearch(ws,wt,ps_s,K,pt=pt,chnls=chnls,
                                         stride0=stride0,stride1=stride1,
                                         dilation=dilation)
    unfold = stnls.UnfoldK(ps_d,pt,dilation=dilation,device=device)
    fold = stnls.FoldK(clean.shape,use_rand=use_rand,nreps=nreps_1,device=device)

    # -- batching info --
    nh,nw = stnls.utils.get_nums_hw(clean.shape,stride0,ps_s,dilation)
    ntotal = t * nh * nw
    nbatches = (ntotal-1) // batch_size + 1

    #
    # -- Step 1 --
    #

    for batch in tqdm.tqdm(range(nbatches)):

        # -- get batch --
        index = min(batch_size * batch,ntotal)
        nbatch_i = min(batch_size,ntotal-index)

        # -- search patches --
        dists,inds = search(noisy,index,nbatch_i)

        # -- get patches --
        noisy_patches_i = unfold(noisy,inds)
        basic_patches_i = unfold(noisy,inds)

        # -- process --
        patches_mod_i = run_denoiser(noisy_patches_i,basic_patches_i,dists,1)

        # -- regroup --
        ones = th.ones_like(dists)
        fold(patches_mod_i,ones,inds)

    # -- post processing --
    basic,weights = fold.vid,fold.wvid
    basic /= weights
    zargs = th.where(weights == 0)
    basic[zargs] = noisy[zargs]

    # -- setup for 2nd step --
    k = 60
    fold = stnls.FoldK(clean.shape,use_rand=use_rand,nreps=nreps_2,device=device)
    search.k = k

    #
    # -- Step 2 --
    #

    for batch in tqdm.tqdm(range(nbatches)):

        # -- batch info --
        index = min(batch_size * batch,ntotal)
        nbatch_i = min(batch_size,ntotal-index)

        # -- search patches --
        dists,inds = search(basic,index,nbatch_i)

        # -- get patches --
        noisy_patches_i = unfold(noisy,inds)
        basic_patches_i = unfold(basic,inds)

        # -- process --
        patches_mod_i = run_denoiser(noisy_patches_i,basic_patches_i,dists,2)

        # -- convert colors to show race condition in rgb --
        yuv2rgb_patches(patches_mod_i)

        # -- regroup --
        ones = th.ones_like(dists)
        fold(patches_mod_i,ones,inds)

    # -- format final image --
    deno,weights = fold.vid,fold.wvid
    deno /= weights
    zargs = th.where(weights == 0)
    deno[zargs] = basic[zargs]
    # yuv2rgb(deno)

    return deno



# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Helper Functions
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def yuv2rgb_patches(patches):
    patches_rs = rearrange(patches,'b k pt c ph pw -> (b k pt) c ph pw')
    yuv2rgb(patches_rs)

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

# -- vnlb denoiser --
def run_denoiser(npatches,bpatches,dists,step):
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
    sigmab2 = sigma2 if step == 1 else sigma2/100.

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


