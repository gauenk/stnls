
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>

template <typename scalar_t>
__global__ void anchor_self_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> inds,
    int qstart, int stride0, int H, int W, int nHW, int nW, int q_per_thread){

  // -- starting qi for thread --
  int Q = dists.size(1);
  int K = dists.size(2);
  int bi = blockIdx.y;
  int qi_thread = q_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int self_index = 0;
  bool eq_loc;
  int loc[3];
  int i_tmp[3];
  scalar_t d_tmp;
  int qi,i_mod;

  // -- for each location --
  for (int qi_ix = 0; qi_ix < q_per_thread; qi_ix++){

    // -- current query --
    qi = qi_thread + qi_ix + qstart;
    if (qi >= Q){ continue; }

    // -- unpack pixel locs --
    // get_pixel_loc(loc,  qi, tmp,  stride0, nW, nHW, H,W);
    int tmp = qi;
    loc[0] = qi / nHW;
    tmp = (tmp - loc[0]*nHW); 
    int nH_index = tmp / nW;
    loc[1] = (nH_index*stride0) % H;
    tmp = tmp - nH_index*nW;
    loc[2] = ((tmp % nW) * stride0) % W;
    // i_mod = qi % nHW;
    // loc[0] = qi / nHW;
    // loc[1] = ((i_mod / nW) * stride0) % H;
    // loc[2] = ((i_mod % nW) * stride0) % W;

    // -- search for matching index --
    for (self_index = 0; self_index < K; self_index++){

      eq_loc = true;
      for (int ix=0; ix<3; ix++){
        eq_loc = eq_loc && (inds[bi][qi][self_index][ix] == loc[ix]);
      }
      if (eq_loc){ break; }

    }
    assert(self_index<K);

    // -- swap dists --
    d_tmp = dists[bi][qi][0];
    dists[bi][qi][0] = dists[bi][qi][self_index];
    dists[bi][qi][self_index] = d_tmp;

    // -- swap inds --
    for(int ix=0; ix<3; ix++){
      i_tmp[ix] = inds[bi][qi][0][ix];
    }
    for(int ix=0; ix<3; ix++){
      inds[bi][qi][0][ix] = loc[ix];
    }
    for(int ix=0; ix<3; ix++){
      inds[bi][qi][self_index][ix] = i_tmp[ix];
    }
    
  }
}

void anchor_self_forward_cuda(
     torch::Tensor dists,
     torch::Tensor inds,
     int qstart, int stride0, int H, int W){
  
  // -- unpack --
  int B = dists.size(0);
  int Q = dists.size(1);
  int K = dists.size(2);

  // -- derivative --
  int nH = (H-1)/stride0+1;
  int nW = (W-1)/stride0+1;
  int nHW = nH*nW;

  // -- num 2 run --
  int nRun = Q;

  // -- kernel params --
  int q_per_thread = 1;
  int _nthreads = 256;
  dim3 nthreads(_nthreads);
  int _nblocks = (nRun-1)/(_nthreads*q_per_thread)+1;
  dim3 nblocks(_nblocks,B);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
  // fprintf(stdout,"nH,nW,stride0: %d,%d,%d\n",nH,nW,stride0);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(dists.type(), "anchor_self_kernel", ([&] {
     anchor_self_kernel<scalar_t><<<nblocks, nthreads>>>(
       dists.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
       inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
       qstart, stride0, H, W, nHW, nW, q_per_thread);
      }));

}

