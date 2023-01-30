
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
using namespace at;

/****************************

       Inline Functions

****************************/

__inline__ __device__ int bounds(int val, int lim ){
  int vval = val;
  if (val < 0){
    vval = -val;
  }else if (val >= lim){
    vval = 2*(lim-1) - val;
  }
  return vval;
}

inline __host__ __device__
int unravel_index(int& ti, int& hi, int& wi, const int qindex,
                  const int h, const int w, const int hw){
  // index to pixel location
  int i_mod = qindex % hw;
  ti = qindex / hw;
  wi = (i_mod % w);
  hi = (i_mod / w) % h;
}

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
  int KnHW = K*nH*nW;
  int raster_index = blockIdx.y * blockDim.x + threadIdx.x;

  // -- image indices for video --
  // int iH = nH*stride;
  // int iW = nW*stride;

  // -- assign inds --
  int bi = blockIdx.x;
  int ki = raster_index % K;
  raster_index = raster_index / K;
  int hi = raster_index % nH;
  int wi = (raster_index/nH) % nW;
  
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
  int delta_h = stride*hi - stride_sparse*hj;
  int delta_w = stride*wi - stride_sparse*wj;

  // -- copy --
  inds_full[bi][hi][wi][ki][0] = inds[bi][hj][wj][ki][0];
  inds_full[bi][hi][wi][ki][1] = bounds(inds[bi][hj][wj][ki][1]+delta_h,iH);
  inds_full[bi][hi][wi][ki][2] = bounds(inds[bi][hj][wj][ki][2]+delta_w,iW);

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
   dim3 nblocks(B,KnHW_blocks);

   // launch kernel
   interpolate_inds_forward_kernel<<<nblocks, nthreads>>>(
       inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
       inds_full.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
       scale,stride,stride_sparse,iH,iW);
}


