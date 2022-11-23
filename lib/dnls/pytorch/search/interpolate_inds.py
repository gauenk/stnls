
import torch as th
from einops import rearrange,repeat
import torch.nn.functional as nnf
import dnls_cuda

def run(inds,scale,stride,T,H,W):

    # -- use --
    nH0 = (H-1)//stride + 1
    nW0 = (W-1)//stride + 1

    # -- unpack and reshape --
    B,nheads,Q,K,_ = inds.shape
    nH1,nW1 = nH0*scale,nW0*scale
    print("[0] inds.shape: ",inds.shape)
    inds = rearrange(inds,'b H (t h w) k tr -> (b H t) h w k tr',t=T,h=nH0)
    print("[a] inds.shape: ",inds.shape)
    print("nH0: ",nH0)
    print("nH1: ",nH1)
    _B = inds.shape[0]

    # -- interpolate (K-1) neighbors --
    inds_k = inds[:,:,:,1:].contiguous()
    inds_full = th.zeros((_B,nH1,nW1,K-1,3),device=inds.device,dtype=inds.dtype)
    dnls_cuda.interpolate_inds(inds_k,inds_full,scale)

    # -- interpolate grid (K==0) --
    inds = rearrange(inds[...,0,:],'b h w tr -> b tr h w')  # B H W tr
    inds_t = inds[:,[0]].type(th.float32)
    inds_hw = inds[:,1:].type(th.float32)
    inds_t = nnf.interpolate(inds_t,size=(nH1,nW1),mode='nearest-exact').type(th.int32)
    inds_hw = nnf.interpolate(inds_hw,size=(nH1,nW1),mode='bilinear').type(th.int32)
    inds_grid = th.cat([inds_t,inds_hw],1)
    inds_grid = rearrange(inds_grid,'b tr h w -> b h w 1 tr')
    print("inds_grid.shape: ",inds_grid.shape)

    # -- append with [0,1,...,K] --
    inds_full = th.cat([inds_full,inds_grid],-2)

    # -- final reshape --
    inds_full = rearrange(inds_full,'(b H t) h w k tr -> b H (t h w) k tr',H=nheads,t=T)
    print("inds_full.shape: ",inds_full.shape)

    return inds_full
