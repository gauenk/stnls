
"""

Aggregation Code derived from N3Net

"""

def indexed_matmul_2_efficient(x,y,I, chunk_size=256):
    return IndexedMatmul2Efficient.apply(x,y,I,chunk_size)


class IndexedMatmul2Efficient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, I, chunk_size=64):
        ctx.save_for_backward(x, y, I)
        ctx.chunk_size = chunk_size
        b,_,o,k = y.shape # b m o k

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


vid_index_neighbours_cache = {}
def vid_index_neighbours(b,t,n1,n2,m1,m2,s,dev,exclude_self=True):

    # -- create vars --
    o = s**2
    if exclude_self:
        o-=1
    n = n1*n2
    m = m1*m2
    assert(m==n)

    key = "{}_{}_{}_{}_{}_{}_{}_{}".format(t,n1,n2,m1,m2,s,exclude_self,dev)
    if not key in vid_index_neighbours_cache:
        I = []
        for ti in range(t):
            It = index_neighbours(1, n1, n2, m1, m2, s, dev, exclude_self=True)
            I.append(It*n*(ti+1))
        I = th.cat(I,1)
        index_neighbours_cache[key] = I
    I = index_neighbours_cache[key]
    I = I.repeat(b,1,1)
    return Variable(I, requires_grad=False)

def vid_to_raster_inds(inds,iH,iW,stride,dev):

    # -- num search --
    nH = (iH-1)//stride+1
    nW = (iW-1)//stride+1
    nHW = nH * nW
    # print("nH,nW: ",nH,nW)

    # -- rasterized --
    tI = inds[...,0]
    hI = th.div(inds[...,1],stride,rounding_mode="floor")
    wI = th.div(inds[...,2],stride,rounding_mode="floor")
    rI = tI * nH * nW + hI * nW + wI
    # print("inds.shape: ",inds.shape)
    # print(tI.shape,stride)
    # print(tI[0,:10],inds[0,:10,0])
    # print(hI[0,:10],inds[0,:10,1])
    # print(wI[0,:10],inds[0,:10,2])
    # print(rI[0,:10])
    # exit(0)


    # -- reshape --
    rI = rI[None,:].contiguous()
    rI = rI.type(th.int64)

    return rI


