
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

template <typename scalar_t>
__global__ void search_flow_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> flows,
    int wt, int stride0, int locs_per_thread){

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
  int W_t = 2*wt+1;
  int t_max;

  // -- get location --
  for (int loc = 0; loc < locs_per_thread; loc++){

    // -- get location --
    int qi = raster_index + loc;
    if (qi >= TnHW){ return; } 
    get_pixel_loc(ref,qi,stride0,nW,nHW,H,W);
    int ti = ref[0];

    // -- stridded index (stride0)  --
    int hn = ref[1]/stride0;
    int wn = ref[2]/stride0;

    // -- init flow index --
    scalar_t h_ref,w_ref;
    h_ref = __int2float_rn(ref[1]);
    w_ref = __int2float_rn(ref[2]);

    // -- temporal search bounds --
    set_time_range(t_max, ti, T, wt);

    // -- run across time --
    bool swap;
    int tj;
    scalar_t h_curr = h_ref;
    scalar_t w_curr = w_ref;
    for(int si=1; si < W_t; si++){

      // -- select time --
      tj = ti + si;
      swap = (ti + si - 1) == t_max;
      tj = (tj > t_max) ? t_max - si : tj;

      // -- reset @ swap --
      h_curr = swap ? h_ref : h_curr;
      w_curr = swap ? w_ref : w_curr;

      // -- select flow --
      auto flow = (tj > ti) ? fflow[bi][tj-1] : bflow[bi][tj+1];

      // -- accumulate center offset  --
      update_centers_flow_acc(h_curr,w_curr,H,W,flow);

      // -- assign  --
      flows[bi][ti][si-1][1][hn][wn] = h_curr - h_ref;
      flows[bi][ti][si-1][0][hn][wn] = w_curr - w_ref;

    }

  }
    
}


void search_flow_forward_cuda(
     const torch::Tensor fflow, const torch::Tensor bflow,
     torch::Tensor flows, int wt, int stride0){
  
  // -- unpack --
  int B = fflow.size(0);
  int T = fflow.size(1);
  int H = fflow.size(3);
  int W = fflow.size(4);
  int W_t = 2*wt;
  assert(W_t > 0);
  assert(W_t == flows.size(2));

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
  AT_DISPATCH_FLOATING_TYPES(fflow.type(), "search_flow_forward_kernel", ([&] {
      search_flow_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
       fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       flows.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
       wt, stride0, locs_per_thread);
      }));

}

/*******************************************


             Backward Flow


*******************************************/

template <typename scalar_t>
__global__ void search_flow_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_fflow,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_bflow,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_flows,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> flows,
    int wt, int stride0, int nW, int nHW, int TnHW, int locs_per_thread){

  // -- unpack --
  int qi;
  int ibatch = blockIdx.y;
  int raster_index = locs_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int T = fflow.size(1);
  int H = fflow.size(3);
  int W = fflow.size(4);
  int ref[3];
  scalar_t refs[3];
  int prop_i[3];
  scalar_t prop[3];
  scalar_t prop_time;
  bool isFwd;

  // -- fwd decl --
  scalar_t v0,v1,gv_W,gv_H;
  scalar_t dAdF0[2];
  scalar_t dAdF1[2];

  // -- get location --
  for (int loc = 0; loc < locs_per_thread; loc++){

    // -- get reference location --
    qi = raster_index + loc;
    if (qi >= TnHW){ break; } 
    get_pixel_loc(ref,qi,stride0,nW,nHW,H,W);

    // -- select time --
    int si = threadIdx.y+1;//isFwd ? threadIdx.z : threadIdx.z+1;
    int t_min,t_max;
    set_time_range_minmax(t_min, t_max, ref[0], T, wt);
    int tj = ref[0] + si;
    tj = (tj > t_max) ? t_max - si : tj;
    bool isFwd = tj > ref[0];
    // int t_flow = isFwd ? tj - 1 : (T-1) - tj;
    int t_flow = isFwd ? tj - 1 : tj + 1;

    // -- directional info --
    auto flow = isFwd ? fflow[ibatch] : bflow[ibatch];
    auto g_flow = isFwd ? grad_fflow[ibatch] : grad_bflow[ibatch];
    // t_flow = isFwd ? t_flow : (T-1) - t_flow;
    // auto pflow = isFwd ? pfflow[ibatch] : pbflow[ibatch];
    // auto g_pflow = isFwd ? grad_pfflow[ibatch] : grad_pbflow[ibatch];

    // -- init/reset --
    v0 = 0;
    v1 = 0;
    gv_W = 0;
    gv_H = 0;

    // -- directional indexing --
    int t_inc = isFwd ? 1 : -1;
    int t_end = isFwd ? (t_max-tj+1) : (tj-t_min+1);
    bool adjacent_frame = abs(ref[0] - tj) == 1;

    // -- floor-divide ref (accumulated index) by stride0 --
    int nh = ref[1]/stride0;
    int nw = ref[2]/stride0;

    // -- write location --
    refs[0] = __int2float_rn(ref[0]);
    if (adjacent_frame){
      refs[1] = __int2float_rn(ref[1]);
      refs[2] = __int2float_rn(ref[2]);
    }else{
      refs[1] = ref[1] + flows[ibatch][ref[0]][si-2][1][nh][nw];
      refs[2] = ref[2] + flows[ibatch][ref[0]][si-2][0][nh][nw];
    }

    // -- iterate across accumulated flows --
    for(int tx=0; tx < t_end; tx++){

      // -- read gradient --
      gv_W = grad_flows[ibatch][ref[0]][si-1+tx][0][nh][nw];
      gv_H = grad_flows[ibatch][ref[0]][si-1+tx][1][nh][nw];

      // -- update dA[i]dF[j] as dAdFj[i] --
      if (tx==0){
        dAdF0[0] = 1;
        dAdF0[1] = 0;
        dAdF1[0] = 0;
        dAdF1[1] = 1;

      }else{




        // -- update proposed location --
        prop[1] = __int2float_rn(ref[1]) + \
          flows[ibatch][ref[0]][si+tx-2][1][nh][nw];
        prop[2] = __int2float_rn(ref[2]) + \
          flows[ibatch][ref[0]][si+tx-2][0][nh][nw];

        // -- update weights --
        update_weights(dAdF0,dAdF1,prop,H,W,tx,flow[t_flow+t_inc*tx]);

      }

      // -- assign to each of the 4 interpolated flow values --
      assign_bilin2d(dAdF0,dAdF1,gv_W,gv_H,refs,H,W,g_flow[t_flow]);

    }
  }
}


void search_flow_backward_cuda(
     torch::Tensor grad_fflow, torch::Tensor grad_bflow,
     const torch::Tensor grad_flows,
     const torch::Tensor fflow, const torch::Tensor bflow,
     const torch::Tensor flows, int wt, int stride0){
  
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
  int W_t = 2*wt;

  // -- kernel params --
  int locs_per_thread = 1;
  // int _nthreads = 448/T;
  int _nthreads = 256/T;
  dim3 nthreads(_nthreads,W_t); // forward and backward
  int _nblocks = (nRun-1)/(_nthreads*locs_per_thread)+1;
  dim3 nblocks(_nblocks,B);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
  // fprintf(stdout,"stride0: %d\n",stride0);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(fflow.type(), "search_flow_backward_kernel", ([&] {
     search_flow_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
       grad_fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       grad_bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       grad_flows.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
       fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
       flows.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
       wt,stride0,nW,nHW,nRun,locs_per_thread);
      }));
}

