
# -- python-only kernel --
from numba import cuda,jit
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- padding --
from dnls.utils.pads import same_padding,comp_pads

# -- fold/unfold
from torch.nn.functional import fold,unfold,pad

def np_matmul(mat_a,mat_b):
    mat_a = mat_a.detach().cpu().numpy()
    mat_b = mat_b.detach().cpu().numpy()
    mat = np.matmul(mat_a,mat_b)
    return mat

def run_nn(vid,ps,stride=4,dilation=1,mode="reflect",vid1=None):
    if vid1 is None: vid1 = vid
    dil = dilation
    vid_pad_s,_ = same_padding(vid,ps,stride,dil,mode)
    patches_s = unfold(vid_pad_s,ps,stride=stride,dilation=dil) # t (c h w) n
    vid_pad_1,_ = same_padding(vid1,ps,1,dil,mode)
    patches_1 = unfold(vid_pad_1,ps,stride=1,dilation=dil) # t (c h w) n
    patches_s = patches_s.permute(0, 2, 1)
    # patches_s = th.ones_like(patches_s)
    # patches_1 = th.ones_like(patches_1)
    # print("patches_s.shape: ",patches_s.shape)
    # print("patches_1.shape: ",patches_1.shape)
    # th.set_float32_matmul_precision(th.float32)
    # th.set_float32_matmul_precision("medium")
    # th.set_float32_matmul_precision("medium")
    # print(th.get_float32_matmul_precision())
    score = th.matmul(patches_s,patches_1)
    t,c,h,w = vid.shape

    # -- float-point error is okay --
    # score_np = np_matmul(patches_s,patches_1)
    # diff = np.abs(score.detach().cpu().numpy() - score_np).max()
    # print(diff)

    # tmp = patches_s[0,:,:] @ patches_1[0,:,:]
    # tmp = rearrange(tmp,'n (h w) -> n h w',h=64)
    # print("tmp.shape: ",tmp.shape)
    # print(tmp[:2,:4,:4])

    t,c,hp,wp = vid_pad_s.shape
    n_h = (hp - (ps-1)*dil - 1)//stride + 1
    score = rearrange(score,'1 (nh nw) n -> n nh nw',nh=n_h)
    score = rearrange(score,'(h w) nh nw -> h w nh nw',h=h)
    return score,None

def run(vid,ps,stride=4,dilation=1,start_index=0,nqueries=-1,k=-1,mode="reflect"):

    # -- unpack --
    t,c,h,w = vid.shape
    device = vid.device
    dil = dilation

    # -- pad image --
    oh1,ow1,hp,wp = comp_pads(vid.shape, ps, stride, dil)
    oh0,ow0,_,_ = comp_pads(vid.shape, ps, 1, dil)

    # -- get shapes --
    n_h = (hp - (ps-1)*dil - 1)//stride + 1
    n_w = (wp - (ps-1)*dil - 1)//stride + 1
    k,nqueries = _get_args(t,h,w,n_h,n_w,k,nqueries)

    # -- allocate --
    nlDists_exh,nlInds_exh = allocate_exh(nqueries,n_h,n_w,device)

    # -- exec --
    numba_search_launcher(vid,start_index,nqueries,nlDists_exh,
                          k,ps,stride,dilation,oh0,ow0,oh1,ow1,mode)

    # -- patches of topk --
    nlDists = nlDists_exh
    # nlDists,nlInds = allocate_k(nqueries,k,device)
    # get_topk(nlDists_exh,nlInds_exh,nlDists,nlInds)
    # nlDists[:,0] = 0. # fix the "-100" hack to 0.
    nlDists = rearrange(nlDists,'(h w) nh nw -> h w nh nw',h=h)

    return nlDists,None#,nlInds

def _get_args(t,h,w,n_h,n_w,k,nqueries):
    if k == -1: k = n_h*n_w
    if nqueries == -1:
        nqueries = t * h * w
    return k,nqueries

def allocate_k(nq,k,device):
    dists = th.zeros((nq,k),device=device,dtype=th.float32)
    dists[...] = float("inf")
    # inds = th.zeros((nq,k,3),device=device,dtype=th.int32)
    # inds[...] = -1
    return dists,None#inds

def allocate_exh(nq,n_h,n_w,device):
    dists = th.zeros((nq,n_h,n_w),device=device,dtype=th.float32)
    dists[...] = float("inf")
    # inds = th.zeros((nq,n_h,n_w,3),device=device,dtype=th.int32)
    # inds[...] = -1
    return dists,None#inds

def get_topk(l2_vals,l2_inds,vals,inds):

    # -- reshape exh --
    nq,st,ws,ws = l2_vals.shape
    l2_vals = l2_vals.view(nq,-1)
    l2_inds = l2_inds.view(nq,-1,3)

    # -- shape info --
    b,_ = l2_vals.shape
    _,k = vals.shape

    # -- take mins --
    order = th.argsort(l2_vals,dim=1,descending=False)
    vals[:b,:] = th.gather(l2_vals,1,order[:,:k])
    for i in range(inds.shape[-1]):
        inds[:b,:,i] = th.gather(l2_inds[:,:,i],1,order[:,:k])

#
# -- Numba --
#

def numba_search_launcher(vid,start_index,nqueries,nlDists,
                          k,ps,stride,dilation,oh0,ow0,oh1,ow1,mode):

    # -- unpack --
    dil = dilation
    t,c,h,w = vid.shape
    device = vid.device

    # -- numbify all params --
    vid_nba = cuda.as_cuda_array(vid)
    nlDists_nba = cuda.as_cuda_array(nlDists)
    # nlInds_nba = cuda.as_cuda_array(nlInds)

    # -- launch params --
    batches_per_block = 10
    bpb = batches_per_block

    # -- launch params --
    h_threads = min(h,32)
    w_threads = min(w,32)
    nthreads = (h_threads,w_threads)
    h_iters = (h-1)//h_threads + 1
    w_iters = (w-1)//w_threads + 1
    nblocks = (nqueries-1)//batches_per_block+1
    use_bound = mode == "reflect"

    # -- exec kernel --
    numba_search[nblocks,nthreads](vid_nba,start_index,nqueries,nlDists_nba,
                                   ps,stride,dilation,oh0,ow0,oh1,ow1,use_bound,h_iters,w_iters,bpb)


# @cuda.jit(debug=True,max_registers=64,opt=False)
@cuda.jit(debug=False,max_registers=64)
def numba_search(vid,start_index,nqueries,
                 dists,ps,stride,dilation,
                 oh0,ow0,oh1,ow1,use_bound,h_iters,w_iters,bpb):

    # -- reflective boundary --
    def bounds(val,lim):
        nval = val
        if val < 0: nval = -nval
        elif val >= lim: nval = 2*(lim-1)-nval
        return int(nval)

    # -- shapes --
    nframes,color,h,w = vid.shape
    height,width = h,w
    bsize,n_h,n_w = dists.shape
    Z = ps*ps*color
    psHalf = int(ps//2)

    # -- dense shape --
    nd_h = h# - (ps-1)
    nd_w = w# - (ps-1)
    nd_hw = nd_h * nd_w

    # -- cuda threads --
    cu_tidX = cuda.threadIdx.x
    cu_tidY = cuda.threadIdx.y
    blkDimX = cuda.blockDim.x
    blkDimY = cuda.blockDim.y


    # ---------------------------
    #
    #      search frames
    #
    # ---------------------------

    # -- access with blocks and threads --
    block_start = cuda.blockIdx.x*bpb

    # -- we want enough work per thread, so we process multiple per block --
    for __bidx in range(bpb):

        # ---------------------------
        #    extract anchor pixel
        # ---------------------------

        # -- relative block [of batch] to absolute block --
        rel_bidx = block_start + __bidx
        if rel_bidx >= nqueries: continue
        abs_bidx = rel_bidx + start_index

        # -- unpack pixel locs --
        ti = abs_bidx // nd_hw
        bidx_mod = abs_bidx % nd_hw
        wi = bidx_mod % nd_w
        hi = (bidx_mod // nd_w) % nd_h

        # ---------------------------
        #     valid (anchor pixel)
        # ---------------------------

        # assert (ti < nframes)
        # assert (hi < (height-(ps-1)))
        # assert (wi < (width-(ps-1)))

        # ---------------------------------------
        #     searching loop for (ti,top,left)
        # ---------------------------------------

        # -- we loop over search space if needed --
        for _hj in range(h_iters):
            n_hj = cu_tidX + blkDimX*_hj
            hj = n_hj * stride
            if n_hj >= n_h: continue

            for _wj in range(w_iters):
                n_wj = cu_tidY + blkDimY*_wj
                wj = n_wj * stride
                if n_wj >= n_w: continue

                # -- we have our pair (hi,wi) and (hj,wj) inside frame (ti) --

                # ---------------------------------
                #
                #  compute delta over patch vol.
                #
                # ---------------------------------
                dist = 0
                for pi in range(ps):
                    for pj in range(ps):

                        # -- inside entire image --
                        if use_bound:
                            hi_p = bounds((hi-oh0) + dilation*pi,height)
                            wi_p = bounds((wi-ow0) + dilation*pj,width)
                        else:
                            hi_p = (hi-oh0) + dilation*pi
                            wi_p = (wi-ow0) + dilation*pj

                        if use_bound:
                            hj_p = bounds((hj-oh1) + dilation*pi,height)
                            wj_p = bounds((wj-ow1) + dilation*pj,width)
                        else:
                            hj_p = (hj-oh1) + dilation*pi
                            wj_p = (wj-ow1) + dilation*pj

                        # -- valid checks [for testing w/ zero pads] --
                        ivalid = (hi_p < height) and (hi_p >= 0)
                        ivalid = ivalid and (wi_p < width) and (wi_p >= 0)

                        jvalid = (hj_p < height) and (hj_p >= 0)
                        jvalid = jvalid and (wj_p < width) and (wj_p >= 0)

                        # -- all channels --
                        for ci in range(color):

                            # -- get data --
                            pix_i = 0. if not(ivalid) else vid[ti][ci][hi_p][wi_p]
                            pix_j = 0. if not(jvalid) else vid[ti][ci][hj_p][wj_p]

                            # -- compute dist --
                            dist += pix_i * pix_j

                # -- dists --
                dists[rel_bidx,n_hj,n_wj] = dist

                # -- inds --
                # inds[bidx,n_hi,n_wi,0] = n_ti
                # inds[bidx,n_hi,n_wi,1] = n_hi
                # inds[bidx,n_hi,n_wi,2] = n_wi

