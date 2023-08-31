
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include <cstddef>
#include <math.h>
#include <ATen/ATen.h>
#include <cuda/std/type_traits>
#include <cstdio>
#include "shared_nn_utils.cu"


using namespace at;

template<typename dtype=int>
__device__ __forceinline__ dtype bounds_clip(dtype val, int lim ){
  dtype vval = val;
  if (val < 0){
    vval = -val; // want ("-1" -> "1") _not_ ("-1" -> "0")
    vval = vval > (lim-1) ? 0 : vval;
  }else if (val > (lim-1)){
    vval = 2*(lim-1)-val; // want ("H" -> "H-2") _not_ ("H" -> "H-1")
    vval = vval < 0 ? lim-1 : vval;
  }
  return vval;
}

template<typename scalar_t, typename itype=int>
__device__ __forceinline__ 
void update_centers_flow(itype& hj_center, itype& wj_center, int H, int W,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> flow){


  // -- fixed so we can read both --
  itype hj_tmp = hj_center;
  itype wj_tmp = wj_center;

  // -- update --
  if(is_same_v<itype,int>){

    // // -- simple rounding if "int" --
    // wj_center = int(1.*wj_center + flow[0][hj_tmp][wj_tmp] + 0.5);
    // hj_center = int(1.*hj_center + flow[1][hj_tmp][wj_tmp] + 0.5);

    // // -- wrap around boarders --
    // wj_center = max(0,min(W-1,(int)wj_center));
    // hj_center = max(0,min(H-1,(int)hj_center));

  }else{


    // -- weighted average of neighbors --
    float weight = 0;
    int hj = 0, wj = 0;
    for (int i=0;i<2;i++){
      for (int j=0;j<2;j++){

        // -- compute int locaion with weight --
        hj = __float2int_rd(hj_tmp + i);
        wj = __float2int_rd(wj_tmp + j);
        weight = max(0.,1-fabs(hj-hj_tmp)) * max(0.,1-fabs(wj-wj_tmp));

        // -- ensure legal boudns --
        hj = bounds(hj,H);
        wj = bounds(wj,W);

        // -- update with shift --
        wj_center += weight*flow[0][hj][wj];
        hj_center += weight*flow[1][hj][wj];
      }
    }

    // -- wrap around boarders --
    // wj_center = max((float)0.,(float)min((float)1.*W-1,(float)wj_center));
    // hj_center = max((float)0.,(float)min((float)1.*H-1,(float)hj_center));

  }
}

template <typename scalar_t, typename itype>
__global__ void accumulate_flow_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<itype,6,torch::RestrictPtrTraits> pfflow,
    torch::PackedTensorAccessor32<itype,6,torch::RestrictPtrTraits> pbflow,
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
  int TnHW = T*nH*nW;
  int tmp;
  int ref[3];

  // -- get location --
  for (int loc = 0; loc < locs_per_thread; loc++){

    // -- get location --
    int qi = raster_index + loc;
    if (qi >= TnHW){ return; } 
    get_pixel_loc(ref,qi,tmp,stride0,nW,nHW,H,W);
    int ti = ref[0];
    int wn = ref[1];
    int hn = ref[2];

    itype hi_a,wi_a;
    if (is_same_v<itype,int>){
      hi_a = (hn * stride0) % H;
      wi_a = (wn * stride0) % W;
    }else{
      hi_a = trunc(__int2float_rn((hn * stride0) % H));
      wi_a = trunc(__int2float_rn((wn * stride0) % W));
    }

    // -- run left --
    int ta = 0;
    auto flow = bflow;
    auto pflow = pbflow;
    itype h_center = hi_a;
    itype w_center = wi_a;
    for(int tj=ti; tj > 0; tj--){

      // -- accumulate center offset  --
      update_centers_flow<scalar_t,itype>(h_center,w_center,H,W,flow[bi][tj]);

      // -- assignment  --
      pflow[bi][ti][ta][1][hn][wn] = h_center - hi_a;
      pflow[bi][ti][ta][0][hn][wn] = w_center - wi_a;

      // -- incriment pre-computed frame index --
      ta++;
    }

    // -- run right --
    ta = 0;
    flow = fflow;
    pflow = pfflow;
    h_center = hi_a;
    w_center = wi_a;
    for(int tj=ti; tj < (T-1); tj++){

      // -- accumulate center offset  --
      update_centers_flow(h_center,w_center,H,W,flow[bi][tj]);

      // -- assignment  --
      // pflow[bi][ti][ta][1][hn][wn] = h_center - hi_a;
      // pflow[bi][ti][ta][0][hn][wn] = w_center - wi_a;
      pflow[bi][ti][ta][1][hn][wn] = h_center - hi_a;
      pflow[bi][ti][ta][0][hn][wn] = w_center - wi_a;

      // -- incriment pre-computed frame index --
      ta++;

    }
  }
    
}


void accumulate_flow_forward_cuda(
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
  if(pfflow.dtype() == torch::kInt32){
    AT_DISPATCH_FLOATING_TYPES(fflow.type(), "accumulate_flow_forward_kernel", ([&] {
        accumulate_flow_forward_kernel<scalar_t,int><<<nblocks, nthreads>>>(
         fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         pfflow.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
         pbflow.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
         stride0,locs_per_thread);
        }));
  }else{
    AT_DISPATCH_FLOATING_TYPES(fflow.type(), "accumulate_flow_forward_kernel", ([&] {
        accumulate_flow_forward_kernel<scalar_t,scalar_t><<<nblocks, nthreads>>>(
         fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         pfflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
         pbflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
         stride0,locs_per_thread);
        }));
  }

}

/*******************************************


             Backward Flow


*******************************************/


// template <typename scalar_t, typename itype>
// __global__ void accumulate_flow_backward_kernel(
//     torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_fflow,
//     torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_bflow,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_inds,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
//     torch::PackedTensorAccessor32<itype,6,torch::RestrictPtrTraits> pfflow,
//     torch::PackedTensorAccessor32<itype,6,torch::RestrictPtrTraits> pbflow,
//     int stride0, int locs_per_thread){

//   // -- unpack --
//   int bi = blockIdx.y;
//   int raster_index = locs_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
//   int T = fflow.size(1);
//   int H = fflow.size(3);
//   int W = fflow.size(4);
//   int nH = (H-1)/stride0+1;
//   int nW = (W-1)/stride0+1;
//   int nHW = nH*nW;
//   int TnHW = T*nH*nW;
//   int tmp;
//   int ref[3];

//   // -- get location --
//   for (int loc = 0; loc < locs_per_thread; loc++){

//     // -- get location --
//     int qi = raster_index + loc;
//     if (qi >= TnHW){ return; } 
//     get_pixel_loc(ref,qi,tmp,stride0,nW,nHW,H,W);
//     int ti = ref[0];
//     int wn = ref[1];
//     int hn = ref[2];

//     itype hi_a,wi_a;
//     if (is_same_v<itype,int>){
//       hi_a = (hn * stride0) % H;
//       wi_a = (wn * stride0) % W;
//     }else{
//       hi_a = trunc(__int2float_rn((hn * stride0) % H));
//       wi_a = trunc(__int2float_rn((wn * stride0) % W));
//     }

//     // -- run left --
//     int ta = 0;
//     auto flow = bflow;
//     auto pflow = pbflow;
//     itype h_center = hi_a;
//     itype w_center = wi_a;
//     for(int tj=ti; tj > 0; tj--){

//       // -- accumulate center offset  --
//       update_centers_flow<scalar_t,itype>(h_center,w_center,H,W,flow[bi][tj]);

//       // -- assignment  --
//       pflow[bi][ti][ta][1][hn][wn] = h_center - hi_a;
//       pflow[bi][ti][ta][0][hn][wn] = w_center - wi_a;


//       // -- incriment pre-computed frame index --
//       ta++;
//     }

//     // -- run right --
//     ta = 0;
//     flow = fflow;
//     pflow = pfflow;
//     h_center = hi_a;
//     w_center = wi_a;
//     for(int tj=ti; tj < (T-1); tj++){

//       // -- accumulate center offset  --
//       update_centers_flow(h_center,w_center,H,W,flow[bi][tj]);

//       // -- assignment  --
//       // pflow[bi][ti][ta][1][hn][wn] = h_center - hi_a;
//       // pflow[bi][ti][ta][0][hn][wn] = w_center - wi_a;
//       pflow[bi][ti][ta][1][hn][wn] = h_center - hi_a;
//       pflow[bi][ti][ta][0][hn][wn] = w_center - wi_a;

//       // -- incriment pre-computed frame index --
//       ta++;

//     }
//   }
    
// }


// void accumulate_flow_backward_cuda(
//      const torch::Tensor fflow, const torch::Tensor bflow,
//      torch::Tensor pfflow, torch::Tensor pbflow, int stride0){
  
//   // -- unpack --
//   int B = fflow.size(0);
//   int T = fflow.size(1);
//   int H = fflow.size(3);
//   int W = fflow.size(4);

//   // -- num 2 run --
//   int nH = (H-1)/stride0+1;
//   int nW = (W-1)/stride0+1;
//   int nRun = T*nH*nW;

//   // -- kernel params --
//   int locs_per_thread = 1;
//   int _nthreads = 256;
//   dim3 nthreads(_nthreads);
//   int _nblocks = (nRun-1)/(_nthreads*locs_per_thread)+1;
//   dim3 nblocks(_nblocks,B);
//   // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
//   // fprintf(stdout,"stride0: %d\n",stride0);

//   // -- launch kernel --
//   if(pfflow.dtype() == torch::kInt32){
//     AT_DISPATCH_FLOATING_TYPES(fflow.type(), "accumulate_flow_backward_kernel", ([&] {
//         accumulate_flow_backward_kernel<scalar_t,int><<<nblocks, nthreads>>>(
//          fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//          bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//          pfflow.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
//          pbflow.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
//          stride0,locs_per_thread);
//         }));
//   }else{
//     AT_DISPATCH_FLOATING_TYPES(fflow.type(), "accumulate_flow_backward_kernel", ([&] {
//         accumulate_flow_backward_kernel<scalar_t,scalar_t><<<nblocks, nthreads>>>(
//          fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//          bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//          pfflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//          pbflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//          stride0,locs_per_thread);
//         }));
//   }

// }
