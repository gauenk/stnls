
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>

template <typename scalar_t>
__global__ void temporal_inds_kernel(
    const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds_t,
    int locs_per_thread){

  // -- unpack --
  int bi = blockIdx.y;
  int raster_index = locs_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int T = fflow.size(1);
  int H = fflow.size(3);
  int W = fflow.size(4);
  int B = inds_t.size(0);
  int Q = inds_t.size(1);
  int K = inds_t.size(2);
  int nT = inds_t.size(3);
  int nT_half = (nT-1)/2+1;
  int hj = 0;
  int wj = 0;
  float hf = 0;
  float wf = 0;

  // -- get location --
  for (int loc = 0; loc < locs_per_thread; loc++){

    // -- get location --
    int ni = raster_index + loc;
    // int ni_mod = ni % Q;
    int qi = ni / Q;
    int ki = ni % K;
    int ti = inds[bi][qi][ki][0];
    int hi = inds[bi][qi][ki][1];
    int wi = inds[bi][qi][ki][2];
    int t_shift = min(0,ti - nT_half) + max(0,ti + nT_half - T);
    int t_left = max(ti - nT_half - t_shift,0);
    int t_right = min(T-1,ti + nT_half - t_shift);

    // -- run left --
    int ta = 0;
    auto flow = bflow;
    hj = hi;
    wj = wi;
    for(int tj=ti; tj > t_left; tj--){

      // -- accumulate --
      hf = (float)(hj) + flow[bi][tj][1][hj][wj];
      wf = (float)(wj) + flow[bi][tj][0][hj][wj];
      hj = (int)max(0,min(H-1,int(hf+0.5)));
      wj = (int)max(0,min(W-1,int(wf+0.5)));

      // -- fill the pre-computed offsets --
      inds_t[bi][qi][ki][ta][0] = tj;
      inds_t[bi][qi][ki][ta][1] = hj;
      inds_t[bi][qi][ki][ta][2] = wj;

      // -- incriment pre-computed frame index --
      ta++;
    }

    // -- run right --
    ta = 0;
    flow = fflow;
    hj = hi;
    wj = wi;
    for(int tj=ti; tj < t_right; tj++){

      // -- accumulate --
      hf = (float)(hj) + flow[bi][tj][1][hj][wj];
      wf = (float)(wj) + flow[bi][tj][0][hj][wj];
      hj = (int)max(0,min(H-1,int(hf+0.5)));
      wj = (int)max(0,min(W-1,int(wf+0.5)));

      // -- fill the pre-computed offsets --
      inds_t[bi][qi][ki][ta][0] = tj;
      inds_t[bi][qi][ki][ta][1] = hj;
      inds_t[bi][qi][ki][ta][2] = wj;

      // -- incriment pre-computed frame index --
      ta++;
    }
    assert(ta == nT);//,"Must be equal."

  }
    
}


void temporal_inds_cuda(
     const torch::Tensor inds,
     const torch::Tensor fflow,
     const torch::Tensor bflow,
     torch::Tensor inds_t){
  
  // -- unpack --
  int B = inds_t.size(0);
  int Q = inds_t.size(1);
  int K = inds_t.size(2);
  int nT = inds_t.size(3);

  // -- num 2 run --
  int nRun = Q*K;

  // -- kernel params --
  int locs_per_thread = 1;
  int _nthreads = 256;
  dim3 nthreads(_nthreads);
  int _nblocks = (nRun-1)/(_nthreads*locs_per_thread)+1;
  dim3 nblocks(_nblocks,B);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
  // fprintf(stdout,"stride0: %d\n",stride0);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(fflow.type(), "temporal_inds_kernel", ([&] {
     temporal_inds_kernel<scalar_t><<<nblocks, nthreads>>>(
       inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
       fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       inds_t.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
       locs_per_thread);
      }));

}
