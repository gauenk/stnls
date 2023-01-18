
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void optical_flow_accumulate_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> pfflow,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> pbflow,
    int stride0, int locs_per_thread){

  // -- unpack --
  int T = fflow.size(1);
  int bi = blockIdx.y;
  int raster_index = locs_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int H = fflow.size(4);
  int W = fflow.size(5);
  int nH = (H-1)/stride0+1;
  int nW = (W-1)/stride0+1;
  int nHW = nH*nW;

  // -- get location --
  for (int loc; loc < locs_per_thread; loc++){

    // -- get location --
    int qi = raster_index + loc;
    int i_mod = qi % nHW;
    int ti = qi / nHW;
    int wi = ((i_mod % nW) * stride0) % W;
    int hi = ((i_mod / nW) * stride0) % H;
    float hf = hi;
    float wf = wi;

    // -- accumulate flow counter --
    int ta = 0;

    // -- run left --
    auto flow = bflow;
    auto pflow = pbflow;
    hf,wf = hi,wi;
    for(int tj=ti-1; tj >= 0; tj--){

      // -- accumulate --
      hf += flow[bi][tj][0][hi][wi];
      wf += flow[bi][tj][1][hi][wi];

      // -- fill the pre-computed offsets --
      pflow[bi][ta][ti][0][hi][wi] = int(hf);
      pflow[bi][ta][ti][1][hi][wi] = int(wf);

      // -- incriment pre-computed frame index --
      ta++;
    }

    // -- run right --
    flow = fflow;
    pflow = pfflow;
    hf,wf = hi,wi;
    for(int tj=ti+1; tj < T; tj--){
      // -- accumulate --
      hf += flow[bi][tj][0][hi][wi];
      wf += flow[bi][tj][1][hi][wi];

      // -- fill the pre-computed offsets --
      pflow[bi][ta][ti][0][hi][wi] = int(hf);
      pflow[bi][ta][ti][1][hi][wi] = int(wf);

      // -- incriment pre-computed frame index --
      ta++;
    }

  }
    
}


void optical_flow_accumulate_cuda(
     const torch::Tensor fflow, const torch::Tensor bflow,
     torch::Tensor pfflow, torch::Tensor pbflow, int stride0){
  
  // -- unpack --
  int B = fflow.size(0);
  int T = fflow.size(1);
  int H = fflow.size(3);
  int W = fflow.size(4);

  // -- num 2 run --
  int nH = (H-1)/stride0+1;
  int nW = (W-1)/stride0+1;
  int nRun = T*nH*nW;

  // -- kernel params --
  int _nthreads = 1024;
  dim3 nthreads(_nthreads);
  int _nblocks = (nRun-1)/_nthreads+1;
  dim3 nblocks(B,_nblocks);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(fflow.type(), "optical_flow_accumulate_kernel", ([&] {
     optical_flow_accumulate_kernel<scalar_t><<<nblocks, nthreads>>>(
       fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       pfflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
       pbflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
       stride0,1);
      }));

}
