
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

# -- package --
import stnls

# -- api --
from .utils import extract_pairs

# -- local --
from .utils import shape_vids,allocate_pair,dist_type_select,allocate_vid
from .shared import manage_self
from .nls_bwd_impl import nls_backward
from .batching_utils import batching_info

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Pytorch Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def nls_fwd(vid0, vid1, fflow, bflow,
            ws, wt, ps, k, nheads=1, batchsize=-1,
            dist_type="prod", stride0=4, stride1=1,
            dilation=1, pt=1, reflect_bounds=True, full_ws=False,
            anchor_self=False, remove_self=False,
            use_adj=True, off_H0=0, off_W0=0, off_H1=0, off_W1=0):

        """

        Run the non-local search

        vid0 = [B,T,C,H,W] or [B,HD,T,C,H,W]
        ws = search Window Spatial (ws)
        wt = search Window Time (wt)

        """

        # -- reshape with heads --
        dtype = vid0.dtype
        device = vid0.device
        vid0,vid1 = shape_vids(nheads,[vid0,vid1])
        B,HD,T,F,H,W = vid0.shape

        # -- create patches --
        pat0 = vid2patches(vid0)
        pat1 = vid2patches(vid1)

        b = y.shape[0]
        m = y.shape[1]
        n = x.shape[1]
        o = I.shape[2]
        e = x.shape[2]
        out = torch.tensor(np.zeros(b*m*o), dtype=torch.float).reshape(b,m,o).cuda()
        # matmul_cuda.matmul1(
        stnls_cuda.n3net_mat_mult1(pat0,pat1,inds,
                                   out,n,m,e,o,b)

        # -- allocate --
        dists,inds = [],[]

        # -- compute batching info --
        ntotal,nbatches = batching_info(vid0,stride0,batchsize)
        for nbatch in range(nbatches):

            # -- extract batch --
            qshift = nbatch*batchsize
            nqueries = min(ntotal-qshift,batchsize)
            dists_b,inds_b = patch_search(pat0,pat1,fflow,bflow,ws,wt,ps,pk,
                                          dist_type,stride0,stride1,dilation,pt,
                                          anchor_self,remove_self,reflect_bounds,
                                          full_ws, use_adj,
                                          off_H0, off_W0, off_H1, off_W1)
            dists.append(dists_b)
            inds.append(inds_b)

        # -- cat --
        dists = th.cat(dists,1)
        inds = th.cat(inds,1)

        # -- return --
        return dists,inds

def patch_search(pat0,pat1,fflow,bflow,ws,wt,ps,pk,
                 dist_type,stride0,stride1,dilation,pt,
                 anchor_self,remove_self,reflect_bounds,
                 full_ws, use_adj,
                 off_H0, off_W0, off_H1, off_W1):

    # -- unpack --
    device = vid0.device
    B,HD,T,C,H,W = vid0.shape

    # -- derived shapes --
    nH0 = (H-1)//stride0+1
    nW0 = (W-1)//stride0+1
    Q = T*nH0*nW0 if Q <= 0 else Q

    # -- search space --
    ws_h,ws_w = ws,ws
    search_abs = ws == -1
    if search_abs:
        ws_h,ws_w = nH0,nW0

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    st = min(2*wt+1,T)
    base_shape = (B,HD,Q,st,ws_h,ws_w)
    dists,inds = allocate_pair(base_shape,device,vid0.dtype,idist_val)

    # -- forward --
    patch_search_impl(vid0, vid1, fflow, bflow,
                      dists, inds, wt, ps, k, dist_type_i,
                      stride0, stride1, dilation, pt, qshift,
                      reflect_bounds, full_ws, search_abs,
                      use_adj, off_H0, off_W0, off_H1, off_W1)

    # -- compress search region --
    dists=dists.view(B,HD,Q,-1)
    inds=inds.view(B,HD,Q,-1,3)

    # -- manage self dists --
    dists,inds = manage_self(dists,inds,anchor_self,
                             remove_self,qshift,stride0,H,W)

    # -- topk --
    dists,inds = stnls.nn.topk(dists,inds,k,dim=3,anchor=anchor_self,
                              descending=descending,unique=False)

    return dists,inds


def patch_search_impl(vid0, vid1, fflow, bflow,
                      dists, inds, wt, ps, k, dist_type_i,
                      stride0, stride1, dilation, pt, qshift,
                      reflect_bounds, full_ws, search_abs,
                      use_adj, off_H0, off_W0, off_H1, off_W1):
    pass

class NonLocalSearchPdbFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, vid0, vid1, fflow, bflow,
                ws, wt, ps, k, nheads=1,
                dist_type="prod", stride0=4, stride1=1,
                dilation=1, pt=1, reflect_bounds=True,
                anchor_self=False, remove_self=False, save_inds=False):
                # use_adj=True, off_H0=0, off_W0=0, off_H1=0, off_W1=0,
                # rbwd=True, nbwd=1, exact=False, queries_per_thread=4,
                # neigh_per_thread=4, channel_groups=-1):

        # -- create inds --

        # -- setup ctx --
        saves = [vid0,vid1,]
        if save_inds: saves += [fflow,bflow,]
        else: saves += [inds,]
        ctx.save_for_backward(*saves)
        ctx_vars = {"stride0":stride0,"ps":ps,"pt":pt}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        # -- create search indices --
        B,T,C,H,W = vid0.shape
        search_inds = create_inds(fflow,bflow,ws,wt,stride0)
        # search_inds.shape = (B, nH x nW, St x Ss)
        B,Q0,S = search_inds.shape

        # -- create patch database --
        pat0 = vid2patches(vid0,stride0,ps)
        pat1 = vid2patches(vid1,stride1,ps)
        B,HD,Q0,F = pat0.shape
        B,HD,Q1,F = pat1.shape

        # -- compute dists --
        z_chunks = [] # b m f k
        for qstart in range(0,Q0,batchsize):

            # -- get batch --
            nbatch = min(batchsize, Q0-qstart)
            I_chunk = I[:,qstart:qstart+nbatch,:]
            pat0_chunk = pat0[:,:,qstart:qstart+nbatch,:]
            pat1_chunk = pat1[:,:,qstart:qstart+nbatch,:]

            # -- create empty shell of weights --
            If = I_chunk.view(B,1,nbatch,F).expand(B,HD,nbatch,F)
            pat0_full = torch.cuda.FloatTensor(B,HD,nbatch,S).fill_(0)

            # -- accumulate over "nbatch"; gaurenteed unique --
            pat0_full = pat0_full.scatter_add(3,If,pat0_chunk.permute(0,1,3,2))
            # y_full
            print("y_full.shape: ",y_full.shape)

            # -- matmult --
            z_interm = torch.cat([torch.matmul(y_full[:,i_k:i_k+1,:,:], x_interm)
                                  for i_k in range(k)], 1)
            z_chunk = z_interm.permute(0,2,3,1)
            z_chunks.append(z_chunk)
        z = torch.cat(z_chunks, 1)
        return z


        # -- viz --
        # print("x.shape: ",x.shape)
        # print("y.shape: ",y.shape,chunk_size)
        # print("I.shape: ",I.shape)

        n,e = x.shape[1:3] # b n f
        m = I.shape[1] # b m o
        x_interm = x.view(b,1,n,e).detach()
        # b = batchsize
        # m = # of patches in image
        # n = # of searched locations
        # o = # searched per location; accumulated over.
        # k = # of parallel sets of weights

        # -- viz --
        # print("x_interm.shape: ",x_interm.shape)
        # print(y[0,:3,:3,0])
        # print(y[0,:3,:3,1])
        # print("-"*20)
        # print(y[0,0,:3,0])
        # print(y[0,1,:3,0])
        # print(y[0,2,:3,0])

        z_chunks = [] # b m f k
        for m_offset in range(0,m,chunk_size):
            this_chunk_size = min(chunk_size, m-m_offset)
            I_chunk = I[:,m_offset:m_offset+this_chunk_size,:]
            y_chunk = y[:,m_offset:m_offset+this_chunk_size,:,:]
            # print("y_chunk.shape: ",y_chunk.shape)

            # -- create empty shell of weights --
            If = I_chunk.view(b,1,this_chunk_size,o).expand(b,k,this_chunk_size,o)
            y_full = torch.cuda.FloatTensor(b,k,this_chunk_size,n).fill_(0)
            # y_full =y_full.scatter_add(source=y_chunk.permute(0,3,1,2), index=If,dim=3)

            # -- accumulate over "this_chunk_size"; gaurenteed unique --
            y_full = y_full.scatter_add(3,If,y_chunk.permute(0,3,1,2))

            # if m_offset == 0:
            #     print("y_chunk.permute(...).shape: ",y_chunk.permute(0,3,1,2).shape)
            #     print("y_full.shape: ",y_full.shape)
            z_interm = torch.cat([torch.matmul(y_full[:,i_k:i_k+1,:,:], x_interm)
                                  for i_k in range(k)], 1)
            z_chunk = z_interm.permute(0,2,3,1)
            z_chunks.append(z_chunk)
        z = torch.cat(z_chunks, 1)
        return z

    @staticmethod
    def backward(ctx, grad):
        x, y, I = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        b,_,o,k = y.shape
        n,e = x.shape[1:3]
        m = I.shape[1]
        x_interm = x.view(b,1,n,e).detach()
        grad_x = torch.zeros_like(x)
        grad_y_chunks = []

        for m_offset in range(0,m,chunk_size):
            this_chunk_size = min(chunk_size, m-m_offset)
            I_chunk = I[:,m_offset:m_offset+this_chunk_size,:]
            y_chunk = y[:,m_offset:m_offset+this_chunk_size,:,:]
            grad_chunk = grad[:,m_offset:m_offset+this_chunk_size,:,:].permute(0,3,2,1)

            If = I_chunk.view(b,1,this_chunk_size,o).expand(b,k,this_chunk_size,o)
            del I_chunk
            y_full = torch.cuda.FloatTensor(b,k,this_chunk_size,n).fill_(0)
            # y_full =y_full.scatter_add(source=y_chunk.permute(0,3,1,2),index=If,dim=3)
            y_full = y_full.scatter_add(3,If,y_chunk.permute(0,3,1,2))

            del y_chunk

            for i_k in range(k):
                grad_x += torch.matmul(grad_chunk[:,i_k,:,:], y_full[:,i_k,:,:]).permute(0,2,1)

            del y_full
            grad_y_full = torch.cat([torch.matmul(x_interm, grad_chunk[:,i_k:i_k+1,:,:]) for i_k in range(k)], 1)
            del grad_chunk
            grad_y_chunk = grad_y_full.gather(2, If.permute(0,1,3,2)).permute(0,3,2,1)
            del grad_y_full
            grad_y_chunks.append(grad_y_chunk)

        grad_y = torch.cat(grad_y_chunks, 1)
        return grad_x, grad_y, None, None


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Pytorch Module
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class NonLocalSearchPdb(th.nn.Module):

    def __init__(self, ws, wt, ps, k, nheads=1,
                 dist_type="prod", stride0=4, stride1=1,
                 dilation=1, pt=1, reflect_bounds=True,
                 full_ws=True, anchor_self=False,
                 remove_self=False, use_adj=True):
        super().__init__()

        # -- core search params --
        self.ws = ws
        self.wt = wt
        self.ps = ps
        self.k = k
        self.nheads = nheads
        self.dist_type = dist_type
        self.stride0 = stride0
        self.stride1 = stride1
        self.dilation = dilation
        self.pt = pt

        # -- manage patch and search boundaries --
        self.reflect_bounds = reflect_bounds
        self.full_ws = full_ws

        # -- special mods to "self" search --
        self.anchor_self = anchor_self
        self.remove_self = remove_self

        # -- searching offsets --
        self.use_adj = use_adj
        self.off_H0 = off_H0
        self.off_W0 = off_W0
        self.off_H1 = off_H1
        self.off_W1 = off_W1

    def forward(self, vid0, vid1, fflow, bflow, batchsize=-1):
        return NonLocalSearchPdbFunction(vid0,vid1,fflow,bflow,
                                         self.ws,self.wt,self.ps,self.k,
                                         self.nheads,batchsize,
                                         self.dist_type,self.stride0,
                                         self.stride1,self.dilation,self.pt,
                                         self.reflect_bounds,self.full_ws,
                                         self.anchor_self,self.remove_self,
                                         self.use_adj,self.off_H0,self.off_W0,
                                         self.off_H1,self.off_W1)

    def flops(self,T,F,H,W):
        return 0

        # -- unpack --
        ps,pt = self.ps,self.pt

        # -- compute search --
        nrefs_hw = ((H-1)//self.stride0+1) * ((W-1)//self.stride0+1)
        nrefs = T * HD * nrefs_hw
        nsearch = ws_h * ws_w * (2*wt+1)
        flops_per_search = 2 * F * ps * ps * pt
        search_flops = nrefs * nsearch * flops_per_search
        flops = search_flops

        # -- compute top-k --
        if self.k > 0:
            sort_flops = nrefs * (nsearch * np.log(nsearch))
            flops += sort_flops

        return flops

    def radius(self,H,W):
        return self.ws

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#            [Direct API]  stnls.search.nls(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid0, vid1, fflow, bflow,
           ws, wt, ps, k, nheads=1, batchsize=-1,
           dist_type="prod", stride0=4, stride1=1,
           dilation=1, pt=1, reflect_bounds=True, full_ws=True,
           anchor_self=True, remove_self=False,
           use_adj=True, off_H0=0, off_W0=0, off_H1=0, off_W1=0):
    fxn = NonLocalSearchFunction.apply
    return fxn(vid0,vid1,fflow,bflow,ws,wt,ps,k,
               nheads,batchsize,dist_type,
               stride0,stride1,dilation,pt,reflect_bounds,
               full_ws,anchor_self,remove_self,
               use_adj,off_H0,off_W0,off_H1,off_W1,
               rbwd,nbwd,exact,
               queries_per_thread,neigh_per_thread,channel_groups)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Python Dict API] stnls.search.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg):
    pairs = {"ws":-1,"wt":-1,"ps":7,"k":10,
             "nheads":1,"dist_type":"prod",
             "stride0":4, "stride1":1, "dilation":1, "pt":1,
             "reflect_bounds":True, "full_ws":True,
             "anchor_self":True, "remove_self":False,
             "use_adj":True,"off_H0":0,"off_W0":0,"off_H1":0,"off_W1":0}
    return extract_pairs(pairs,cfg)

def init(cfg):
    cfg = extract_config(cfg)
    search = NonLocalSearchPdb(cfg.ws, cfg.wt, cfg.ps, cfg.k, nheads=cfg.nheads,
                               dist_type=cfg.dist_type, stride0=cfg.stride0,
                               stride1=cfg.stride1, dilation=cfg.dilation, pt=cfg.pt,
                               reflect_bounds=cfg.reflect_bounds, full_ws=cfg.full_ws,
                               anchor_self=cfg.anchor_self, remove_self=cfg.remove_self,
                               use_adj=cfg.use_adj,off_H0=cfg.off_H0,off_W0=cfg.off_W0,
                               off_H1=cfg.off_H1,off_W1=cfg.off_W1)
    return search

