
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>


#include <cstddef>
#include <math.h>
#include <ATen/ATen.h>
#include <cuda/std/type_traits>
#include <cstdio>
#include "shared_flows.cu"

using namespace at;


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
  int ref[3];

  // -- get location --
  for (int loc = 0; loc < locs_per_thread; loc++){

    // -- get location --
    int qi = raster_index + loc;
    if (qi >= TnHW){ return; } 
    get_pixel_loc(ref,qi,stride0,nW,nHW,H,W);
    int ti = ref[0];
    int wn = ref[1];
    int hn = ref[2];


    // -- ??? -- maybe "/stride0"?
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
      update_centers_flow_acc<scalar_t,itype>(h_center,w_center,H,W,flow[bi][tj]);

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
      update_centers_flow_acc(h_center,w_center,H,W,flow[bi][tj]);

      // -- assignment  --
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

template <typename scalar_t>
__global__ void accumulate_flow_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> dev,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_fflow,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_bflow,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_pfflow,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_pbflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> pfflow,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> pbflow,
    int stride0, int nW, int nHW, int TnHW, int locs_per_thread){

  // -- unpack --
  int qi;
  int ibatch = blockIdx.y;
  int raster_index = locs_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int dir = threadIdx.y;
  int T = fflow.size(1);
  int H = fflow.size(3);
  int W = fflow.size(4);
  int ref[3];
  scalar_t refs[3];
  scalar_t prop[3];
  bool isFwd;

  // -- fwd decl --
  scalar_t gv0,gv1;
  scalar_t dAdF0[2];
  scalar_t dAdF1[2];

  // -- get directional information --
  isFwd = dir == 0;
  int t_flow = threadIdx.z;//isFwd ? threadIdx.z : threadIdx.z+1;
  auto flow = isFwd ? fflow[ibatch] : bflow[ibatch];
  auto g_flow = isFwd ? grad_fflow[ibatch] : grad_bflow[ibatch];
  auto pflow = isFwd ? pfflow[ibatch] : pbflow[ibatch];
  auto g_pflow = isFwd ? grad_pfflow[ibatch] : grad_pbflow[ibatch];

  // -- get location --
  for (int loc = 0; loc < locs_per_thread; loc++){



    // -- get reference location --
    qi = raster_index + loc;
    if (qi >= TnHW){ break; } 
    get_pixel_loc(ref,qi,stride0,nW,nHW,H,W);

    // -- init/reset --
    gv0 = 0;
    gv1 = 0;

    // -- directional indexing --
    int t_inc = 1;
    int Acc_time_start = t_flow-ref[0];
    int t_end = (T-1)-t_flow;
    if (Acc_time_start < 0) { return; }

    // -- accumulated index --
    int nh = ref[1]/stride0;
    int nw = ref[2]/stride0;

    // -- write location --
    refs[0] = __int2float_rn(ref[0]);
    if (Acc_time_start == 0){
      refs[1] = __int2float_rn(ref[1]);
      refs[2] = __int2float_rn(ref[2]);
    }else{
      refs[1] = ref[1] + pflow[ref[0]][Acc_time_start-1][1][nh][nw];
      refs[2] = ref[2] + pflow[ref[0]][Acc_time_start-1][0][nh][nw];
    }

    // -- iterate across accumulated flows --
    for(int tx=0; tx < t_end; tx++){

      // -- read gradient --
      gv0 = g_pflow[ref[0]][Acc_time_start+tx][0][nh][nw];
      gv1 = g_pflow[ref[0]][Acc_time_start+tx][1][nh][nw];

      // -- update dA[i]dF[j] as dAdFj[i] --
      if (tx==0){
        dAdF0[0] = 1;
        dAdF0[1] = 0;
        dAdF1[0] = 0;
        dAdF1[1] = 1;
      }else{

        // -- update proposed location --
        prop[1] = __int2float_rn(ref[1]) +                      \
          pflow[ref[0]][Acc_time_start+tx-1][1][nh][nw];
        prop[2] = __int2float_rn(ref[2]) +                      \
          pflow[ref[0]][Acc_time_start+tx-1][0][nh][nw];

        // -- update weights --
        update_weights(dAdF0,dAdF1,prop,H,W,tx,flow[t_flow+t_inc*tx]);
      }

      // -- assign to each of the 4 interpolated flow values --
      assign_bilin2d(dAdF0,dAdF1,gv0,gv1,refs,H,W,g_flow[t_flow]);

    }

  }
}


void accumulate_flow_backward_cuda(
     torch::Tensor dev,
     torch::Tensor grad_fflow, torch::Tensor grad_bflow,
     const torch::Tensor grad_pfflow, const torch::Tensor grad_pbflow,
     const torch::Tensor fflow, const torch::Tensor bflow,
     const torch::Tensor pfflow, const torch::Tensor pbflow, int stride0){
  
  // -- unpack --
  int B = fflow.size(0);
  int T = fflow.size(1);
  int H = fflow.size(3);
  int W = fflow.size(4);

  // -- num 2 run --
  int nH = (H-1)/stride0+1;
  int nW = (W-1)/stride0+1;
  int nHW = nH*nW;
  int nRun = T*nH*nW;

  // -- kernel params --
  int locs_per_thread = 1;
  // int _nthreads = 448/T;
  int _nthreads = 256/T;
  dim3 nthreads(_nthreads,2,T-1); // forward and backward
  int _nblocks = (nRun-1)/(_nthreads*locs_per_thread)+1;
  dim3 nblocks(_nblocks,B);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
  // fprintf(stdout,"stride0: %d\n",stride0);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(fflow.type(), "accumulate_flow_backward_kernel", ([&] {
     accumulate_flow_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
       dev.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
       grad_fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       grad_bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       grad_pfflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
       grad_pbflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
       fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       pfflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
       pbflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
       stride0,nW,nHW,nRun,locs_per_thread);
      }));
}

