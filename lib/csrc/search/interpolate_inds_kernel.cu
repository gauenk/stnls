
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
    int scale){

  // -- unpack indices --
  int B = inds_full.size(0);
  int nH = inds_full.size(1);
  int nW = inds_full.size(2);
  int K = inds_full.size(3);
  int KnHW = K*nH*nW;
  int raster_index = blockIdx.y * blockDim.x + threadIdx.x;

  // -- assign inds --
  int bi = blockIdx.x;
  int ki = raster_index % K;
  raster_index = raster_index / K;
  int hi = raster_index % nW;
  int wi = raster_index / nW;
  
  // -- full inds to smaller inds --
  int hj = hi / scale;
  int wj = wi / scale;
  bool at_grid = (hi % scale == 0) && (wi % scale == 0);
  int kj = ki;
  int shift_h = 0;
  int shift_w = 0;

  if (!copy){
    // -- (k == 0) is actually our 1st neighbor (NOT "self") --
    shift_h = hi % scale;
    shift_w = wi % scale;
  }

  // -- copy --
  inds_full[bi][hi][wi][ki][0] = inds[bi][hj][wj][ki][0];
  inds_full[bi][hi][wi][ki][1] = inds[bi][hj][wj][ki][1]-shift_h;
  inds_full[bi][hi][wi][ki][2] = inds[bi][hj][wj][ki][2]-shift_w;

} // fxn

void interpolate_inds_forward_cuda(
    torch::Tensor inds, torch::Tensor inds_full,
    int scale){

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
   int nthreads = warp_size*nwarps;
   // nthreads = min(nthreads,KnHW);
   // dim3 nthreads(nthreads);

   // -- nblocks --
   // int nquery_blocks = (QHW-1)/nthreads+1;
   int KnHW_blocks = (KnHW-1) / nthreads + 1;
   // qpt = ((nqueries - 1) / nquery_blocks) + 1;
   dim3 nblocks(B,KnHW_blocks);

   // launch kernel
   interpolate_inds_forward_kernel<<<nblocks, nthreads>>>(
       inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
       inds_full.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
       scale);
}


