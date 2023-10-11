"""

Shared Logical Units

"""

import stnls

def manage_self(dists,inds,anchor_self,remove_self,qshift,stride0,H,W):
    assert not(remove_self and anchor_self)
    if remove_self:
        outs = stnls.nn.remove_self(dists,inds,stride0,H,W,qhift)
        dists,inds = outs
        print("invalid.")
        exit(0)
    if anchor_self:
        B,HD,T,nH,nW,W_t,ws2 = dists.shape
        dists = dists.view(B,HD,Q,-1)
        d2or3 = inds.shape[-1]
        inds = inds.view(B,HD,Q,-1,d2or3)
        stnls.nn.anchor_self(dists,inds,stride0,H,W,qshift)
        dists=dists.reshape(B,HD,T,nH0,nW0,W_t,ws*ws)
        inds=inds.reshape(B,HD,T,nH0,nW0,W_t,ws*ws,d2or3)
    return dists,inds



def normz_bwd(ctx,grad_vid0,grad_vid1):
    # -- normz --
    # from torch.nn.functional import fold
    from torchvision.transforms.functional import center_crop
    from stnls import iFoldz

    nH1 = (H-1)//ctx.stride1+1
    nW1 = (W-1)//ctx.stride1+1

    reflect_bounds = ctx.reflect_bounds
    dilation = ctx.dil
    # print(grad_vid1.shape)
    fold = stnls.iFoldz(grad_vid1[:1,:1,:1].shape,
                        stride=ctx.stride0,dilation=dilation,
                        reflect_bounds=reflect_bounds,
                        device=vid1.device,use_adj=False)
    counts = th.ones((1,nH0*nW0,1,1,1,ctx.ps,ctx.ps),device=grad_dists.device)
    counts,_ = fold(counts)

    # counts = th.ones((1,ctx.ps*ctx.ps,nH0*nW0),device=grad_dists.device)
    # Hp = (nH0 - 1) * ctx.stride0 + ctx.ps
    # Wp = (nW0 - 1) * ctx.stride0 + ctx.ps
    # counts = fold(counts, (Hp,Wp), [ctx.ps]*2, 1, [0,0], ctx.stride0)
    # counts = center_crop(counts,(H,W))
    # sH,sW = (Hp-H+1)//2,(Wp-W+1)//2
    # counts = counts[...,sH:sH+H,sW:sW+W]

    # print("[nls] grad_vid0:")
    # print(grad_vid0[0,0,0,:3,:3])
    # print(counts[0,0,:3,:3])
    # grad_vid0 /= counts[None,]
    grad_vid0 /= counts

    # fold = stnls.iFoldz(grad_vid1[:1,:1,:1].shape,
    #                     stride=ctx.stride1,dilation=dilation,
    #                     reflect_bounds=reflect_bounds,
    #                     device=vid1.device,use_adj=False)
    # counts = th.ones((1,nH1*nW1,1,1,1,ctx.ps,ctx.ps),device=grad_dists.device)
    # counts,_ = fold(counts)

    from torch.nn.functional import fold
    counts = th.ones((1,ctx.ps*ctx.ps,nH1*nW1),device=grad_dists.device)
    Hp = (nH1 - 1) * ctx.stride1 + ctx.ps
    Wp = (nW1 - 1) * ctx.stride1 + ctx.ps
    counts = fold(counts, (Hp,Wp), [ctx.ps]*2, 1, [0,0], ctx.stride1)
    sH,sW = (Hp-H+1)//2,(Wp-W+1)//2
    counts = counts[...,sH:sH+H,sW:sW+W]

    # counts = center_crop(counts,(H,W))
    # print(counts)
    # print("counts.shape: ",counts.shape)
    # print("[nls] grad_vid1:")
    # print(grad_vid1[0,0,0,:3,:3])
    # print("[nls] counts: ")
    # print(counts[0,0,:3,:3])
    # grad_vid1 /= counts[None,]
    grad_vid1 /= counts



