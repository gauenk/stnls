
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

        # print(grad_pfflow[0,0,:,0])
        # print(grad_pfflow[0,:,0,0])
        # print(grad_pfflow[0,:,0,0])

        # -- backward --
        fflow,bflow,pfflow,pbflow = ctx.saved_tensors
        bflow = fflow.clone()
        # fflow[0,0,:,:4,:4] = fflow[0,1,:,:4,:4]
        # fflow[0,0,0,:4,:4] = fflow[0,1,1,:4,:4]
        # fflow[0,1,0,:4,:4] = fflow[0,1,1,:4,:4]
        # fflow[0,2,:,:4,:4] = fflow[0,1,:,:4,:4]
        # print(fflow[0,0,:,:4,:4])
        # print(fflow[0,1,:,:5,:5])
        # print(fflow[0,1,:,:4,:4])
        # print(fflow[0,2,:,:4,:4])
        # print(pfflow[0,0,:,:,:4,:4])
        stnls_cuda.accumulate_flow_backward(dev,grad_fflow,grad_bflow,
                                            grad_pfflow, grad_pbflow,
                                            fflow,bflow,pfflow,pbflow,
                                            ctx.stride0)

        print("-="*10 + " dev + " + "-="*10)
        # B, Q, patial FLOW dt, partial ACC dt, 2, 2, 4
        print(dev[0,0,0,1])
        print("-"*10)
        print(dev[0,0,1,1])
        print("-"*10)
        print(dev[0,1,1,1])
        print("-"*10)
        print(dev[0,2,1,1])
        args = th.where(th.logical_or(dev[...,-2]>0.1,dev[...,-1]>0.1))
        args_th = th.stack(args)
        print(args_th.shape)
        print(args_th)
        # exit()
        print("-="*20)
        print("-="*20)


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
                # bclip(pbflow[:,ti,tn],H,W)
                # print("[acc] fwd: ",ti,tj,tn)
                # tmp = pfflow[:,ti,tn].clone()
                # bclip(pfflow[:,ti,tn],H,W)
                for tk in range(1,tn+1):
                    # if ti == 0:
                    #     print(pfflow[0,ti,tn])
                    #     print(flow_warp(fflow[:,ti+tk], pfflow[:,ti,tn], imode)[0,:,:3,:3])
                    # pfflow[:,ti,tn] = th.round(pfflow[:,ti,tn],decimals=3)
                    pfflow[:,ti,tn] = pfflow[:,ti,tn] + flow_warp(fflow[:,ti+tk], pfflow[:,ti,tn], imode)
                    # pfflow[:,ti,tn] = th.round(pfflow[:,ti,tn],decimals=3)
                    # bclip(pfflow[:,ti,tn],H,W)
                    # tmp = pfflow[:,ti,tn].clone()
                    # bclip(pfflow[:,ti,tn],H,W)
                # if ti == 0:
                #     print(tn,pfflow[0,ti,tn,:,:2,:2])
                # # bclip(pfflow[:,ti,tn],H,W)
                # if ti == 0:
                #     print(tn,pfflow[0,ti,tn,:,:2,:2])
                    # pfflow[:,ti,tn] += flow_warp(pfflow[:,ti,tn], fflow[:,ti+tk])
            else:
                tn = ti-tj-1
                pbflow[:,ti,tn] = bflow[:,ti]
                # bclip(pbflow[:,ti,tn],H,W)
                for tk in range(1,tn+1):
                    # bclip(pbflow[:,ti,tn],H,W)
                    # pbflow[:,ti,tn] = th.round(pbflow[:,ti,tn],decimals=3)
                    pbflow[:,ti,tn] = pbflow[:,ti,tn] + flow_warp(bflow[:,ti-tk], pbflow[:,ti,tn], imode)
                    # pbflow[:,ti,tn] = th.round(pbflow[:,ti,tn],decimals=3)
                    # pbflow[:,ti,tn] += flow_warp(pbflow[:,ti,tn], bflow[:,ti-tk])

def bclip(flow,H,W):
    grid = make_grid(flow[None,:])[0]
    print(grid[0,:3,:3])
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
    # flow = flow.flip(-3)
    H,W = x.shape[-2:]
    # H,W = h,w
    # x = pad(x)
    # flow = pad(flow)
    n, _, h, w = x.size()

    # create mesh grid
    grid_y, grid_x = th.meshgrid(th.arange(0, h, dtype=x.dtype, device=x.device),
                                 th.arange(0, w, dtype=x.dtype, device=x.device))
    grid = th.stack((grid_x, grid_y), 0).float()[None,:]  # 2, W(x), H(y)
    grid.requires_grad = False

    vgrid = grid + flow
    # vgrid = vgrid.round()
    # vgrid = bounds(vgrid,h,w)
    # x = pad(x)
    # vgrid = pad(vgrid)
    # vgrid = bounds(vgrid,h,w)

    # scale grid to [-1,1]
    hp,wp = x.shape[-2:]
    vgrid_x = 2.0 * vgrid[:, 0, :, :] / max(wp - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1, :, :] / max(hp - 1, 1) - 1.0
    vgrid_scaled = th.stack((vgrid_x, vgrid_y), dim=-1)
    # print(vgrid_scaled.requires_grad)
    # print(vgrid_scaled.min(),vgrid_scaled.max())
    # print(vgrid_scaled[0,:,0,0])
    # print(vgrid_scaled[0,:,0,1])
    # vgrid_scaled = bounds(vgrid_scaled.transpose(1,-1),1,1).transpose(1,-1)


    # x = x.flip(-3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode,
                           padding_mode="reflection", align_corners=align_corners)
    # output = th.round(output,decimals=4) # allows for _int_ing of small values
    # print(output)
    # output = output.flip(-3)
    # output = output[...,H:2*H,W:2*W].contiguous()
    # output = output[...,H:H+H,W:W+W].contiguous()

    # bclip(output,H,W)

    # print(output)
    # -- clip where needed --
    # print(output.shape)
    # print(output)
    # output = output+grid
    # bclip(output,H,W)
    # output = output-grid

    return output

def pad(vid):
    """

    This padding is why the current methods are not equal.

    """

    H,W = vid.shape[-2:]
    # vid_c = th.stack([vid[...,1:H-1,1:W-1]
    # vid_c = th.stack([vid[...,1:H-1,1:W-1]
    # vid_c = th.stack([vid[...,1:H-1,1:W-1]
    vid = th.cat([vid.flip(-1),vid,vid.flip(-1)],-1)
    vid = th.cat([vid.flip(-2),vid,vid.flip(-2)],-2)
    # vid = th.cat([vid,vid,vid],-1)
    # vid = th.cat([vid,vid,vid],-2)
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
