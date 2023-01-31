
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "shared_nn_utils.cu"
using namespace at;


/****************************

       Forward Pass

****************************/

__global__ void interpolate_inds_forward_kernel(
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds_full,
    int scale, int stride, int stride_sparse, int iH, int iW){

  // -- unpack indices --
  int B = inds_full.size(0);
  int nH = inds_full.size(1);
  int nW = inds_full.size(2);
  int K = inds_full.size(3);
  int nHW = nH*nW;
  int KnHW = K*nH*nW;
  int raster_index = threadIdx.x + blockDim.x * blockIdx.x;

  // -- image indices for video --
  // int iH = nH*stride;
  // int iW = nW*stride;
  if (raster_index >= KnHW){ return; } // don't run invalid threads.

  // -- assign inds --
  int bi = blockIdx.y;
  int ki = raster_index / nHW;
  raster_index = (raster_index - ki*nHW);
  int hi = raster_index / nW;
  raster_index = (raster_index - hi*nW);
  int wi = raster_index;
  
  // -- shifts [could be randomized; fixed across ki] --
  int shift_h = 0;//raster_index % scale;
  int shift_w = 0;//raster_index / scale;
 
  // -- sparse index inds --
  int hi_sparse = hi / scale;
  int wi_sparse = wi / scale;

  // -- selected shifted sparse --
  int hj = hi_sparse + shift_h;
  int wj = wi_sparse + shift_w;

  // -- the (h,w) shift in the full resolution space --
  int delta_h = stride*hi - stride_sparse*hj; // stride * hi - scale * stride * hi;
  int delta_w = stride*wi - stride_sparse*wj;

  // -- copy --
  inds_full[bi][hi][wi][ki][0] = inds[bi][hj][wj][ki][0];
  inds_full[bi][hi][wi][ki][1] = bounds(inds[bi][hj][wj][ki][1]+delta_h,iH);
  inds_full[bi][hi][wi][ki][2] = bounds(inds[bi][hj][wj][ki][2]+delta_w,iW);
  // inds_full[bi][hi][wi][ki][1] = inds[bi][hj][wj][ki][1]+delta_h;
  // inds_full[bi][hi][wi][ki][2] = inds[bi][hj][wj][ki][2]+delta_w;

} // fxn

void interpolate_inds_forward_cuda(
    torch::Tensor inds, torch::Tensor inds_full,
    int scale, int stride, int stride_sparse, int iH, int iW){

   // -- unpack --
   int B = inds_full.size(0);  
   int nH = inds_full.size(1);
   int nW = inds_full.size(2);
   int K = inds_full.size(3);
   int nHW = nH*nW;
   int KnHW = K*nH*nW;
   // int QHW = Q*nH*nW;

   // -- nthreads --
   int nwarps = 29;
   int warp_size = 32;
   int num_threads = warp_size*nwarps;
   num_threads = min(num_threads,KnHW);
   // dim3 nthreads(nH,nW);
   int nthreads = num_threads;

   // -- nblocks --
   // int nquery_blocks = (QHW-1)/nthreads+1;
   int KnHW_blocks = (KnHW-1) / nthreads + 1;
   // qpt = ((nqueries - 1) / nquery_blocks) + 1;
   dim3 nblocks(KnHW_blocks,B);

   // launch kernel
   interpolate_inds_forward_kernel<<<nblocks, nthreads>>>(
       inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
       inds_full.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
       scale,stride,stride_sparse,iH,iW);
}


