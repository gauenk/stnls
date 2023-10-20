
import torch as th
import torch.nn.functional as tnnf
from easydict import EasyDict as edict

def rescale_flows(flows_og,H,W):

    # -- corner case --
    if flows_og is None: return None

    # -- check --
    fshape = list(flows_og.fflow.shape)
    _H,_W = fshape[-2:]
    if _H == H:
        return flows_og
    fshape[-2] = H
    fshape[-1] = W

    # -- alloc --
    fflow = flows_og.fflow.view(-1,2,_H,_W)
    bflow = flows_og.bflow.view(-1,2,_H,_W)
    shape = (H,W)

    # -- scale factor --
    scale_H =  _H/H
    scale_W =  _W/W
    scale = th.Tensor([scale_W,scale_H]).to(fflow.device)
    scale = scale.view(1,2,1,1)

    # -- create new flows --
    flows = edict()
    flows.fflow = tnnf.interpolate(fflow/scale,size=shape,
                                   mode="bilinear",align_corners=True)
    flows.bflow = tnnf.interpolate(bflow/scale,size=shape,
                                   mode="bilinear",align_corners=True)

    # -- reshape --
    flows.fflow = flows.fflow.view(*fshape)
    flows.bflow = flows.bflow.view(*fshape)

    return flows
