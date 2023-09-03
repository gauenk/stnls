
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
#include "../search/nls_bilin2d.cu"


using namespace at;

template<typename scalar_t, typename itype=int>
__device__ __forceinline__ 
void update_centers_flow_acc(itype& hj_center, itype& wj_center, int H, int W,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> flow){


  // -- fixed so we can read both --
  itype hj_tmp = hj_center;
  itype wj_tmp = wj_center;

  // -- weighted average of neighbors --
  float weight = 0;
  int hj = 0, wj = 0;
#pragma unroll
  for (int i=0;i<2;i++){
#pragma unroll
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

template <typename scalar_t>
__device__ __forceinline__ 
void assign_bilin2d(scalar_t dAdF0[2], scalar_t dAdF1[2],
                    scalar_t gv0, scalar_t gv1, scalar_t* prop, int H, int W,
     torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> g_flow){

  // -- read --
  int prop_i[2];
  scalar_t gH,gW,w;
#pragma unroll
  for (int ix=0;ix<2;ix++){
#pragma unroll
    for (int jx=0;jx<2;jx++){

      // -- interpolation weights --
      prop_i[0] = __float2int_rd(prop[1]+ix);
      gH = max(0.,1-fabs(prop_i[0]-prop[1]));
      prop_i[1] = __float2int_rd(prop[2]+jx);
      gW = max(0.,1-fabs(prop_i[1]-prop[2]));
      w = gH*gW;

      // -- bounds --
      prop_i[0] = bounds(prop_i[0],H);
      prop_i[1] = bounds(prop_i[1],W);

      // -- write --
      atomicAdd(&(g_flow[0][prop_i[0]][prop_i[1]]),w*(dAdF0[0]*gv0+ dAdF0[1]*gv1));
      atomicAdd(&(g_flow[1][prop_i[0]][prop_i[1]]),w*(dAdF1[0]*gv0+ dAdF1[1]*gv1));
    }
  }

}

template <typename scalar_t>
__device__ __forceinline__ 
void set_to_const(scalar_t w0[][2], scalar_t w1[][2], scalar_t C){
#pragma unroll
    for (int _ix = 0; _ix < 2; _ix++){
#pragma unroll
      for (int _jx = 0; _jx < 2; _jx++){
        w0[_ix][_jx] = C;
        w1[_ix][_jx] = C;
      }
    }
}

template <typename scalar_t>
__device__ __forceinline__ 
void update_weights(scalar_t dAdF0[2], scalar_t dAdF1[2],
                    scalar_t* prop, int H, int W, int tx,
    const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> flow,
    torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> dev){

  // -- read --
  scalar_t dFlow[2][2];
  int prop_i[2];
  dFlow[0][0] = 0;
  dFlow[0][1] = 0;
  dFlow[1][0] = 0;
  dFlow[1][1] = 0;

  // -- zero out any invalid --
  // bool any_zero[2];
  // any_zero[0] = false;
  // any_zero[1] = false;

  // -- check bounds --
  // scalar_t eps = 1e-7;
  // prop[1] = prop[1]-eps;
  // prop[2] = prop[2]-eps;
  int sH = check_interval(prop[1],0,H) ? 1 : -1;
  int sW = check_interval(prop[2],0,W) ? 1 : -1;

  // -- init wrap --
  prop[1] = bounds(prop[1],H);
  prop[2] = bounds(prop[2],W);

  scalar_t gH,gW,vW,vH;
#pragma unroll
  for (int ix=0;ix<2;ix++){
#pragma unroll
    for (int jx=0;jx<2;jx++){

      // -- interpolation weights --
      prop_i[0] = __float2int_rz(ix ==0 ? floorf(prop[1]) : ceilf(prop[1]));
      // prop_i[0] = __float2int_rz(ix ==0 ? floorf(prop[1]) : (floorf(prop[1])+1));
      gH = max(0.,1-fabs(prop_i[0]-prop[1]));
      // gH = (ix == 0) ? 1 - (prop[1] - prop_i[0]) : 1 - (prop_i[0] - prop[1]);
      // gH = (ix == 0) ? prop_i[0] - prop[1]: prop[1] - prop_i[0];

      prop_i[1] = __float2int_rz(jx ==0 ? floorf(prop[2]) : ceilf(prop[2]));
      // prop_i[1] = __float2int_rz(jx ==0 ? floorf(prop[2]) : (floorf(prop[2])+1));
      gW = max(0.,1-fabs(prop_i[1]-prop[2]));
      // gW = (jx == 0) ? 1 - (prop[2] - prop_i[1]) : 1 - (prop_i[1] - prop[2]);
      // gW = (jx == 0) ? prop_i[1] - prop[2] : prop[2] - prop_i[1];

      // -- compute direction --
      bool left0 = (prop_i[0]-prop[1]) < 0;
      bool right0 = (prop_i[0]-prop[1]) > 0;
      bool left1 = (prop_i[1]-prop[2]) < 0;
      bool right1 = (prop_i[1]-prop[2]) > 0;
      // bool left0 = ix == 0;
      // bool right0 = ix == 1;
      // bool left1 = jx == 0;
      // bool right1 = jx == 1;

      // -- zero out edge --
      // any_zero[0] = (not(left0) && not(right0)) or any_zero[0];
      // any_zero[1] = (not(left1) && not(right1)) or any_zero[1];

      // -- read --
      vW = flow[0][prop_i[0]][prop_i[1]];
      vH = flow[1][prop_i[0]][prop_i[1]];

      // -- update --
      dFlow[0][0] += left1 ? -gH*vW : (right1 ? gH*vW : 0); // dF[0]/dF[0]; A(0)
      dFlow[0][1] += left0 ? -gW*vW : (right0 ? gW*vW : 0); // dF[0]/dF[0]; A(1)

      dFlow[1][0] += left1 ? -gH*vH : (right1 ? gH*vH : 0); // dF[1]/dF[1]; A(0)
      dFlow[1][1] += left0 ? -gW*vH : (right0 ? gW*vH : 0); // dF[1]/dF[1]; A(1)


    }
  }
  
  // if(any_zero[1]){
  //   dFlow[0][1] = 0;
  //   dFlow[1][1] = 0;
  // }
  // if(any_zero[0]){
  //   dFlow[0][0] = 0;
  //   dFlow[1][0] = 0;
  // }

  // -- reset or accumulate --
    
  // -- assign --
  scalar_t _dAdF0[2];
  scalar_t _dAdF1[2];
  _dAdF0[0] = dAdF0[0];
  _dAdF0[1] = dAdF0[1];
  _dAdF1[0] = dAdF1[0];
  _dAdF1[1] = dAdF1[1];

  // -- update --
  dAdF0[0] += dFlow[0][0]*sW*_dAdF0[0] + dFlow[0][1]*sH*_dAdF0[1];
  dAdF0[1] += dFlow[1][0]*sW*_dAdF0[0] + dFlow[1][1]*sH*_dAdF0[1];
  dAdF1[0] += dFlow[0][0]*sW*_dAdF1[0] + dFlow[0][1]*sH*_dAdF1[1];
  dAdF1[1] += dFlow[1][0]*sW*_dAdF1[0] + dFlow[1][1]*sH*_dAdF1[1];

}



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
  int tmp;
  int ref[3];
  scalar_t refs[3];
  int prop_i[3];
  scalar_t prop[3];
  scalar_t prop_time;
  bool isFwd;

  // -- fwd decl --
  scalar_t v0,v1,gv0,gv1;
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
    get_pixel_loc(ref,qi,tmp,stride0,nW,nHW,H,W);

    // -- init/reset --
    v0 = 0;
    v1 = 0;
    gv0 = 0;
    gv1 = 0;

    // -- directional indexing --
    int t_inc = 1;
    int Acc_time_start = t_flow-ref[0];
    int t_end = (T-1)-t_flow;
    if (Acc_time_start < 0) { return; }

    // -- write location --
    refs[0] = __int2float_rn(ref[0]);
    if (Acc_time_start == 0){
      refs[1] = __int2float_rn(ref[1]);
      refs[2] = __int2float_rn(ref[2]);
    }else{
      refs[1] = ref[1] + pflow[ref[0]][Acc_time_start-1][1][ref[1]][ref[2]];
      refs[2] = ref[2] + pflow[ref[0]][Acc_time_start-1][0][ref[1]][ref[2]];
    }

    // -- iterate across accumulated flows --
    for(int tx=0; tx < t_end; tx++){

      // -- read gradient --
      gv0 = g_pflow[ref[0]][Acc_time_start+tx][0][ref[1]][ref[2]];
      gv1 = g_pflow[ref[0]][Acc_time_start+tx][1][ref[1]][ref[2]];

      // -- update dA[i]dF[j] as dAdFj[i] --
      if (tx==0){
        dAdF0[0] = 1;
        dAdF0[1] = 0;
        dAdF1[0] = 0;
        dAdF1[1] = 1;
      }else{

        // -- update proposed location --
        prop[1] = __int2float_rn(ref[1]) +                      \
          pflow[ref[0]][Acc_time_start+tx-1][1][ref[1]][ref[2]];
        prop[2] = __int2float_rn(ref[2]) +                      \
          pflow[ref[0]][Acc_time_start+tx-1][0][ref[1]][ref[2]];

        // -- update weights --
        update_weights(dAdF0,dAdF1,prop,H,W,tx,flow[t_flow+t_inc*tx],
                       dev[ibatch][qi][t_flow][tx]);
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

