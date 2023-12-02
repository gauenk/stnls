
# -- python --
import torch as th
from functools import partial
from easydict import EasyDict as edict
import torch.nn.functional as F

# -- cpp cuda kernel --
import stnls_cuda

def init():
    return run

def run(*args,**kwargs):
    if len(args) == 1:
        return run_flows(*args,**kwargs)
    elif len(args) == 2:
        return run_pair(*args,**kwargs)
    elif len(args) == 3:
        return run_pair(*args,**kwargs)

class accumulate_flow_th(th.autograd.Function):

    @staticmethod
    def forward(ctx, fflow, bflow, stride0):

        # -- get sizes --
        B,T,_,H,W = bflow.shape
        nH = (H-1)//stride0+1
        nW = (W-1)//stride0+1
        dtype = fflow.dtype
        device = fflow.device

        # -- init --
        pfflow = th.zeros((B,T,T-1,2,nH,nW),device=device,dtype=dtype)
        pbflow = th.zeros((B,T,T-1,2,nH,nW),device=device,dtype=dtype)

        # -- forward --
        stnls_cuda.accumulate_flow_forward(fflow,bflow,pfflow,pbflow,stride0)

        # -- setup ctx --
        ctx.save_for_backward(fflow,bflow,pfflow,pbflow)
        ctx_vars = {"stride0":stride0,"fshape":list(fflow.shape)}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        return pfflow,pbflow

    @staticmethod
    def backward(ctx, grad_pfflow, grad_pbflow):

        # -- init --
        grad_fflow = th.zeros(ctx.fshape,device=grad_pfflow.device)
        grad_bflow = th.zeros(ctx.fshape,device=grad_pfflow.device)

        # -- get sizes --
        stride0 = ctx.stride0
        dtype = grad_bflow.dtype
        device = grad_bflow.device
        B,T,_,H,W = grad_bflow.shape
        nH = (H-1)//stride0+1
        nW = (W-1)//stride0+1
        dev = th.zeros((B,T*nH*nW,T-1,T-1,2,2,6),device=device,dtype=dtype)

        # -- backward --
        fflow,bflow,pfflow,pbflow = ctx.saved_tensors
        bflow = bflow.flip(1)
        grad_pbflow = grad_pbflow.flip(1)
        pbflow = pbflow.flip(1)
        stnls_cuda.accumulate_flow_backward(dev,grad_fflow,grad_bflow,
                                            grad_pfflow, grad_pbflow,
                                            fflow,bflow,pfflow,pbflow,
                                            ctx.stride0)
        grad_bflow = grad_bflow.flip(1)
        # print("none check: ",grad_fflow is None,grad_bflow is None)

        return grad_fflow,grad_bflow,None


def run_flows(flows,stride0=1,dtype=None,fwd_mode="pytorch"):
    return run_pair(flows.fflow,flows.bflow,stride0=stride0,dtype=dtype,fwd_mode=fwd_mode)

def run_pair(fflow,bflow,stride0=1,dtype=None,
             interpolation_mode="bilinear",fwd_mode="pytorch"):

    # -- unpack --
    B,T,_,H,W = fflow.shape
    B,T,_,H,W = bflow.shape
    device = fflow.device
    dtype = fflow.dtype if dtype is None else dtype


    # -- run --
    if fwd_mode == "pytorch":

        # -- get size --
        nH = (H-1)//stride0+1
        nW = (W-1)//stride0+1

        # -- allocate --
        pfflow = th.zeros((B,T,T-1,2,nH,nW),device=device,dtype=dtype)
        pbflow = th.zeros((B,T,T-1,2,nH,nW),device=device,dtype=dtype)
        run_accumulate_flow(fflow,bflow,pfflow,pbflow,stride0,interpolation_mode)
    else:
        pfflow, pbflow = accumulate_flow_th.apply(fflow,bflow,stride0)
        # stnls_cuda.accumulate_flow_forward(fflow,bflow,pfflow,pbflow,stride0)

    # -- rounding to remove numerical errors with interpolation --
    # pfflow = th.round(pfflow,decimals=4)
    # pbflow = th.round(pbflow,decimals=4)

    # -- format --
    flows = edict()
    flows.fflow = pfflow
    flows.bflow = pbflow

    return flows

def extract_search_from_accumulated(fflow,bflow,wt,stride0):
    # -- setup --
    T = fflow.shape[1]
    W_t = 2*wt+1
    assert W_t <= T,"Search Window Must be at most half the number of frames."
    flows = []
    for ti in range(T):
        # -- bounds for ti --
        t_shift = min(0,ti - wt) + max(0,ti + wt - (T-1))
        t_max = min(T-1,ti + wt - t_shift)
        flows_t = []
        for si in range(1,W_t):
            # -- select adjacent frame --
            tj = ti + si
            tj = t_max - si if (tj > t_max) else tj
            dt = tj - ti
            flow_gt = fflow[:,ti,dt-1] if (ti < tj) else bflow[:,ti,-dt-1]
            flows_t.append(flow_gt[...,::stride0,::stride0])
        flows_t = th.stack(flows_t,1)
        flows.append(flows_t)
    flows = th.stack(flows,1)
    return flows

def run_accumulate_flow(fflow,bflow,pfflow,pbflow,stride0,imode):
    assert stride0 == 1
    B,T,_,H,W = fflow.shape
    for ti in range(T):
        for tj in range(T):
            if ti == tj:
                continue
            elif ti < tj:
                tn = tj-ti-1
                pfflow[:,ti,tn] = fflow[:,ti]
                for tk in range(1,tn+1):
                    pfflow[:,ti,tn] = pfflow[:,ti,tn] + \
                        flow_warp(fflow[:,ti+tk], pfflow[:,ti,tn], imode)
            else:
                tn = ti-tj-1
                pbflow[:,ti,tn] = bflow[:,ti]
                for tk in range(1,tn+1):
                    pbflow[:,ti,tn] = pbflow[:,ti,tn] + \
                        flow_warp(bflow[:,ti-tk], pbflow[:,ti,tn], imode)

def flow_warp(x, flow, interp_mode='bilinear',
              padding_mode='reflection', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """
    # -- unpack --
    H,W = x.shape[-2:]
    n, _, h, w = x.size()

    # -- create mesh grid --
    grid = index_grid(h,w,dtype=x.dtype,device=x.device)
    # grid_y, grid_x = th.meshgrid(th.arange(0, h, dtype=x.dtype, device=x.device),
    #                              th.arange(0, w, dtype=x.dtype, device=x.device))
    # grid = th.stack((grid_x, grid_y), 0).float()[None,:]  # 2, W(x), H(y)
    # grid.requires_grad = False

    vgrid = grid + flow

    # -- scale grid to [-1,1] --
    hp,wp = x.shape[-2:]
    vgrid_x = 2.0 * vgrid[:, 0, :, :] / max(wp - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1, :, :] / max(hp - 1, 1) - 1.0
    vgrid_scaled = th.stack((vgrid_x, vgrid_y), dim=-1)

    # -- resample --
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode,
                           padding_mode="reflection", align_corners=align_corners)

    return output

def index_grid(H,W,dtype=th.float,device="cuda"):
    # -- create mesh grid --
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                 th.arange(0, W, dtype=dtype, device=device))
    grid = th.stack((grid_x, grid_y), 0).float()[None,:]  # 1, 2, W(x), H(y)
    grid.requires_grad = False
    return grid

def bclip(flow,H,W):
    grid = make_grid(flow[None,:])[0]
    fgrid = grid + flow
    eps = 0#1e-4
    flow[1] = th.where(fgrid[1]>(H-1-eps),H-1-fgrid[1],fgrid[1])-grid[1]
    flow[1] = th.where(fgrid[1]<eps,-fgrid[1],fgrid[1])-grid[1]
    flow[0] = th.where(fgrid[0]>(W-1-eps),W-1-fgrid[0],fgrid[0])-grid[0]
    flow[0] = th.where(fgrid[0]<eps,-fgrid[0],fgrid[0])-grid[0]

def make_grid(x):
    b,t,_,h,w = x.shape
    grid_y, grid_x = th.meshgrid(th.arange(0, h, dtype=x.dtype, device=x.device),
                                 th.arange(0, w, dtype=x.dtype, device=x.device))
    grid = th.stack((grid_x, grid_y), 0).float()[None,:]  # 1, 2, W(x), H(y)
    grid.requires_grad = False
    return grid

def pad(vid):
    """

    This padding is why the current methods are not equal.

    """
    H,W = vid.shape[-2:]
    vid = th.cat([vid.flip(-1),vid,vid.flip(-1)],-1)
    vid = th.cat([vid.flip(-2),vid,vid.flip(-2)],-2)
    return vid

def bounds(vgrid,H,W):
    L = [W,H]
    eps = 0#1e-3
    print("vgrid: ",vgrid.shape)
    for i in range(2):
        print("pre: ",vgrid[:,i].min(),vgrid[:,i].max())
        args = th.where(vgrid[:,i] > (L[i]-1))
        vgrid[:,i][args] = 2*(L[i]-1) - vgrid[:,i][args]
        args = th.where(vgrid[:,i] < eps)
        vgrid[:,i][args] = -vgrid[:,i][args]
        print("post: ",vgrid[:,i].min(),vgrid[:,i].max())

    return vgrid
