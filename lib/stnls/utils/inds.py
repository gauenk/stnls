
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- numba --
try:
    import numba
    from numba import njit,prange
except:
    pass

# -- local --
from .pads import comp_pads

def get_exh_inds(vid,stride=1):
    t,c,h,w = vid.shape
    qSearch = t*h*w // stride
    return get_query_batch(0,qSearch,stride,t,h,w,vid.device)

def get_query_batch(index,qSearch,stride,t,h,w,device):
    srch_inds = numba_query_launcher(index,qSearch,stride,t,h,w,device)
    return srch_inds

def get_iquery_batch(index,qSearch,stride,_coords,t,device=None,dtype=None):

    # -- add temporal if needed --
    coords = list(_coords) # copy
    if len(coords) == 4: # spatial only; add time
        coords = [0,t,] + coords

    # -- unpack --
    sq_t = coords[1] - coords[0]
    sq_h = coords[4] - coords[2]
    sq_w = coords[5] - coords[3]
    fstart,top,left = coords[0],coords[2],coords[3]

    # -- get inds --
    srch_inds = numba_query_launcher(index,qSearch,stride,sq_t,sq_h,sq_w,device,dtype)

    # -- add offsets --
    srch_inds[:,0] += fstart
    srch_inds[:,1] += top
    srch_inds[:,2] += left

    return srch_inds

def numba_query_launcher(index,qSearch,stride,t,h,w,device=None,dtype=None):
    # -- type assert --
    assert not(device is None) or not(dtype is None)

    # -- exec fill of search values --
    srch_inds = np.zeros((qSearch,3),dtype=np.int64)
    # assert (h % stride == 0) and (w % stride == 0)
    numba_query_raster(srch_inds,index,qSearch,stride,t,h,w)
    # numba_query_equal(srch_inds,index,qSearch,stride,t,h,w)
    srch_inds = th.from_numpy(srch_inds).contiguous()

    # -- handle device --
    if not(dtype is None):
        srch_inds = srch_inds.type(type(dtype))
    elif not(device is None):
        srch_inds = srch_inds.to(device)
    else:
        raise ValueError("We need dtype or device not None.")
    return srch_inds

try:
    @njit
    def numba_query_raster(srch_inds,index,qSearch,stride,t,h,w):
        # hs = (h-1) // (stride-1) + 1
        # ws = (w-1) // (stride-1) + 1
        nh = int((h-1) // stride) + 1
        nw = int((w-1) // stride) + 1
        npf = nh*nw
        hw = h*w
        stride2 = stride**2
        for raw_qi in prange(qSearch):
    
            # -- ind -> ti --
            # ind = stride2 * (qi + index)
            # ti = ((qi * strid2) // hw) % t
            qi = raw_qi + index
            ti = qi // (nh*nw)
            _qi = qi % (nh*nw)
            # ind_f = (stride2*qi) % hw
            # hi = (stride)*((stride*ind_f) // w)
            hi = ((_qi // nw) * stride) % h
            wi = ((_qi % nw) * stride) % w
            # wi = ((stride-1)*ind_f) % w
            # wi = (ind_f/stride) % w
            # hi = (stride)*(ind_f // (stride*w))
            # wi = (ind_f/stride) % w
    
            # -- fill --
            srch_inds[raw_qi,0] = ti
            srch_inds[raw_qi,1] = hi
            srch_inds[raw_qi,2] = wi
    
    
    @njit
    def numba_query_equal(srch_inds,index,qSearch,stride,t,h,w):
        qSearchTotal_t = h*w//stride
        stride_sr = np.sqrt(stride)
        nT = qSearchTotal_t
        wf = w*1.
        # How to evenly distribution points in a grid? I.E. how to use "stride"
        nX = np.sqrt((wf/h)*nT + (wf-h)**2/(4.*h**2)) - (wf-h)/(2.*h)
        nX = int(nX)
        start = index * qSearch
        for qi in prange(qSearch):
    
            # -- ind -> ti --
            ind = qi + start
            ti = ind // qSearchTotal_t
            ind = ind %  qSearchTotal_t
    
            # -- ind -> hi --
            hi = stride_sr*(ind // nX)
            hi = int(hi)
            if hi >= h:
                hi = h-1
    
            # -- ind -> hi --
            wi = stride_sr*(ind % nX)
            wi = int(wi)
            if wi >= w:
                wi = w-1
    
            # -- fill --
            srch_inds[qi,0] = ti
            srch_inds[qi,1] = hi
            srch_inds[qi,2] = wi
except:
    pass

def get_3d_inds(inds,stride,t,h,w):

    # -- unpack --
    hw = h*w
    bsize,num = inds.shape
    device = inds.device
    qSearchTotal_t = h*w//stride

    # -- shortcuts --
    tdiv = th.div
    tmod = th.remainder

    # -- init --
    aug_inds = th.zeros((3,bsize,num),dtype=th.int64)
    aug_inds = aug_inds.to(inds.device)

    # -- fill frame index --
    aug_inds[0,...] = tdiv(inds,qSearchTotal_t,rounding_mode='floor') # t = inds // hw

    # -- correct for stride offsets --
    delta = compute_stride_offsets(stride,t,h,w,device)
    inds_mod = tmod(inds,qSearchTotal_t)
    for ti in range(t):
        args = th.where(ti == aug_inds[0])
        # print(ti,len(args[0]))
        # print("[0]: ",inds[args][0])
        # print("[a]: ",inds_mod[args][0])
        inds_mod[args] -= delta[ti]
        # print("[b]: ",inds_mod[args])
        # print("[c]: ",inds_mod[args][0]*stride)
    exit(0)

    # -- fill spatial inds --
    aug_inds[1,...] = tdiv(inds_mod,w,rounding_mode='floor') # (inds % hw) // w
    aug_inds[2,...] = tmod(inds_mod,w)
    aug_inds = rearrange(aug_inds,'three b n -> (b n) three')

    return aug_inds

def compute_stride_offsets(stride,t,h,w,device):
    assert stride < h and stride < w
    delta = th.zeros(t,device=device,dtype=th.int32)
    hw = h*w
    qSearchTotal_t = hw//stride
    for ti in range(1,t):
        final_ind = (ti*stride*qSearchTotal_t) % hw
        delta[ti] = (hw - final_ind) % stride
    return delta

def get_nums_hw(vshape,stride,ps,dil,pad_same=True,only_full=True):

    # -- padding --
    _,_,h,w = vshape
    if pad_same:
        _,_,h,w = comp_pads(vshape, ps, stride, dil)

    # -- num each spatial direction --
    if only_full:
        n_h = (h - (ps-1)*dil - 1)//stride + 1
        n_w = (w - (ps-1)*dil - 1)//stride + 1
    else:
        n_h = (h - 1)//stride + 1
        n_w = (w - 1)//stride + 1

    return n_h,n_w

def get_batching_info(vshape,stride0,stride1,ps,dil):

    # -- padding --
    oh0,ow0,hp0,wp0 = comp_pads(vshape, ps, stride0, dil)
    oh1,ow1,hp1,wp1 = comp_pads(vshape, ps, stride1, dil)

    # -- num each spatial direction --
    n_h0 = (hp0 - (ps-1)*dil - 1)//stride0 + 1
    n_w0 = (wp0 - (ps-1)*dil - 1)//stride0 + 1
    n_h1 = (hp1 - (ps-1)*dil - 1)//stride1 + 1
    n_w1 = (wp1 - (ps-1)*dil - 1)//stride1 + 1

    # -- total --
    t = vshape[0]
    ntotal0 = t * n_h0 * n_w0
    ntotal1 = t * n_h0 * n_w0
    return ntotal0,ntotal1,(n_h0,n_w0),(n_h1,n_w1)

