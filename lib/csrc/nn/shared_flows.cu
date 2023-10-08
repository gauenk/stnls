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
    const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> flow
    // torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> dev
                    ){

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
