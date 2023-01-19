
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void optical_flow_accumulate_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> pfflow,
    torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> pbflow,
    int stride0, int locs_per_thread){

  // -- unpack --
  int bi = blockIdx.y;
  int raster_index = locs_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int T = fflow.size(1);
  int H = fflow.size(3);
  int W = fflow.size(4);
  int nH = (H-1)/stride0+1;
  int nW = (W-1)/stride0+1;
  int nHW = nH*nW;
  int hj = 0;
  int wj = 0;

  // -- get location --
  for (int loc = 0; loc < locs_per_thread; loc++){

    // -- get location --
    int qi = raster_index + loc;
    int qi_mod = qi % nHW;
    int ti = qi / nHW;
    int wn = qi_mod % nW;
    int hn = (qi_mod / nW) % nH;
    int wi = ((qi_mod % nW) * stride0) % W;
    int hi = ((qi_mod / nW) * stride0) % H;
    float hf = hi;
    float wf = wi;

    // -- run left --
    int ta = 0;
    auto flow = bflow;
    auto pflow = pbflow;
    hj = hi;
    wj = wi;
    for(int tj=ti; tj > 0; tj--){

      // -- accumulate --
      hf = (float)(hj) + flow[bi][tj][1][hj][wj];
      wf = (float)(wj) + flow[bi][tj][0][hj][wj];
      hj = (int)max(0,min(H-1,int(hf+0.5)));
      wj = (int)max(0,min(W-1,int(wf+0.5)));

      // -- fill the pre-computed offsets --
      pflow[bi][ta][ti][1][hn][wn] = hj;
      pflow[bi][ta][ti][0][hn][wn] = wj;

      // -- incriment pre-computed frame index --
      ta++;
    }

    // -- run right --
    ta = 0;
    flow = fflow;
    pflow = pfflow;
    hj = hi;
    wj = wi;
    for(int tj=ti; tj < (T-1); tj++){

      // -- accumulate --
      hf = (float)(hj) + flow[bi][tj][1][hj][wj];
      wf = (float)(wj) + flow[bi][tj][0][hj][wj];
      hj = (int)max(0,min(H-1,int(hf+0.5)));
      wj = (int)max(0,min(W-1,int(wf+0.5)));

      // -- fill the pre-computed offsets --
      pflow[bi][ta][ti][1][hn][wn] = hj;
      pflow[bi][ta][ti][0][hn][wn] = wj;

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
  int locs_per_thread = 1;
  int _nthreads = 256;
  dim3 nthreads(_nthreads);
  int _nblocks = (nRun-1)/(_nthreads*locs_per_thread)+1;
  dim3 nblocks(_nblocks,B);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
  // fprintf(stdout,"stride0: %d\n",stride0);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(fflow.type(), "optical_flow_accumulate_kernel", ([&] {
     optical_flow_accumulate_kernel<scalar_t><<<nblocks, nthreads>>>(
       fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       pfflow.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
       pbflow.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
       stride0,locs_per_thread);
      }));

}
