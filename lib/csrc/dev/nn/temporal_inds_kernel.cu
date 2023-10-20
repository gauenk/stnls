
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>

template <typename scalar_t>
__global__ void temporal_inds_kernel(
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> inds_t,
    int locs_per_thread){

  // -- unpack --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int raster_index = locs_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int T = fflow.size(1);
  int H = fflow.size(3);
  int W = fflow.size(4);
  int B = inds_t.size(0);
  int Q = inds_t.size(2);
  int K = inds_t.size(3);
  int nT = inds_t.size(4); // always even since = wt*2
  int wt = (nT-1)/2+1; // *should* always be == nT/2
  int hj = 0;
  int wj = 0;
  int QK = Q*K;
  int hj_tmp,wj_tmp;

  // -- get location --
  for (int loc = 0; loc < locs_per_thread; loc++){

    // -- get location --
    int ni = raster_index + loc;
    int qi = (ni / K);
    if (qi >= Q){continue;}
    int ki = (ni - qi*K) % K;
    int ti = inds[ibatch][ihead][qi][ki][0];
    int hi = inds[ibatch][ihead][qi][ki][1];
    int wi = inds[ibatch][ihead][qi][ki][2];
    assert((ti >= 0) && (ti < T));
    int t_shift = min(0,ti - wt) + max(0,ti + wt - (T-1));
    int t_left = max(ti - wt - t_shift,0);
    int t_right = min(T-1,ti + wt - t_shift);

    // -- skip invalid --
    if (ti < 0){ continue; }
    if (hi < 0){ continue; }
    if (wi < 0){ continue; }

    // -- run left --
    int ta = 0;
    int t_prev = ti;
    auto flow = bflow;
    hj = hi;
    wj = wi;
    for(int tj=ti-1; tj >= t_left; tj--){

      // -- accumulate --
      hj_tmp = hj;
      wj_tmp = wj;
      hj = int(1.*hj + flow[ibatch][t_prev][1][hj_tmp][wj_tmp] + 0.5);
      wj = int(1.*wj + flow[ibatch][t_prev][0][hj_tmp][wj_tmp] + 0.5);
      hj = max(0,min(H-1,hj));
      wj = max(0,min(W-1,wj));

      // -- fill the pre-computed offsets --
      inds_t[ibatch][ihead][qi][ki][ta][0] = tj;
      inds_t[ibatch][ihead][qi][ki][ta][1] = hj;
      inds_t[ibatch][ihead][qi][ki][ta][2] = wj;

      // -- update previous flow --
      t_prev = tj;

      // -- incriment pre-computed frame index --
      ta++;
    }

    // -- run right --
    flow = fflow;
    t_prev = ti;
    hj = hi;
    wj = wi;
    for(int tj=ti+1; tj <= t_right; tj++){

      // -- accumulate --
      hj_tmp = hj;
      wj_tmp = wj;
      hj = int(1.*hj + flow[ibatch][t_prev][1][hj_tmp][wj_tmp] + 0.5);
      wj = int(1.*wj + flow[ibatch][t_prev][0][hj_tmp][wj_tmp] + 0.5);
      hj = max(0,min(H-1,hj));
      wj = max(0,min(W-1,wj));

      // -- fill the pre-computed offsets --
      inds_t[ibatch][ihead][qi][ki][ta][0] = tj;
      inds_t[ibatch][ihead][qi][ki][ta][1] = hj;
      inds_t[ibatch][ihead][qi][ki][ta][2] = wj;

      // -- update previous flow --
      t_prev = tj;

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
  int HD = inds_t.size(1);
  int Q = inds_t.size(2);
  int K = inds_t.size(3);
  int nT = inds_t.size(4);

  // -- num 2 run --
  int nRun = Q*K;

  // -- kernel params --
  int locs_per_thread = 1;
  int _nthreads = 256;
  dim3 nthreads(_nthreads);
  int _nblocks = (nRun-1)/(_nthreads*locs_per_thread)+1;
  dim3 nblocks(_nblocks,B,HD);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(fflow.type(), "temporal_inds_kernel", ([&] {
     temporal_inds_kernel<scalar_t><<<nblocks, nthreads>>>(
       inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
       fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       inds_t.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
       locs_per_thread);
      }));

}
