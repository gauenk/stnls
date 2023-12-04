
#include <cuda/std/type_traits>
#include <cstdio>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

#include <math.h>
// #include "stdio.h"
// #include "iostream"
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <vector>
// #include <chrono>
#include <ATen/ATen.h>
// template< class T, class U >
// inline constexpr bool is_same_v = cuda::std::is_same<T, U>::value;

#include "../shared_kernel.cu"
using namespace at;


/*********************************************

        Forward (Compute Distances 2d)

 *********************************************/


template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void compute_dist_2d(scalar_t& dist, //int Z,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame0,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame1,
  int* ref_patch, int* prop_patch, int* ref, int* prop,
  bool* valid_ref, bool* valid_prop,
  int ps, int dilation, bool reflect_bounds,
  int patch_offset, scalar_t invalid, int* offsets,
  int F, int qH, int qW, int kH, int kW){
                  
  scalar_t pix0,pix1,_dist;
  // Z = 0;
  for (int pi = 0; pi < ps; pi++){

    // -- ref height --
    ref[0] = ref_patch[0]+offsets[0]+dilation*(pi + patch_offset);
    ref[0] = reflect_bounds ? bounds(ref[0],qH) : ref[0];
    valid_ref[0] = check_interval(ref[0],0,qH);

    // -- proposed height --
    prop[0] = prop_patch[0]+dilation*(pi + patch_offset);
    prop[0] = reflect_bounds ? bounds(prop[0],kH) : prop[0];
    valid_prop[0] = check_interval(prop[0],0,kH);

    for (int pj = 0; pj < ps; pj++){
      
      // -- ref width --
      ref[1] = ref_patch[1]+offsets[1]+dilation*(pj + patch_offset);
      ref[1] = reflect_bounds ? bounds(ref[1],qW) : ref[1];
      valid_ref[1] = check_interval(ref[1],0,qW);

      // -- prop width --
      prop[1] = prop_patch[1]+dilation*(pj + patch_offset);
      prop[1] = reflect_bounds ? bounds(prop[1],kW) : prop[1];
      valid_prop[1] = check_interval(prop[1],0,kW);

      // -- ensure valid location --
      valid_ref[2] = true;
      valid_prop[2] = true;
      #pragma unroll
      for (int bool_idx=0; bool_idx<2; bool_idx++){
        valid_ref[2] = valid_ref[2] && valid_ref[bool_idx];
        valid_prop[2] = valid_prop[2] && valid_prop[bool_idx];
      }
      bool valid = valid_ref[2] and valid_prop[2];
      if (not valid){ continue; }
      // Z += 1;

      // -- fill each channel --
      for (int ci = 0; ci < F; ci++){

        // -- get data --
        pix0 = valid_ref[2] ? frame0[ci][ref[0]][ref[1]] : (scalar_t)0.;
        pix1 = valid_prop[2] ? frame1[ci][prop[0]][prop[1]] : (scalar_t)0.;

        // -- compute dist --
        if(DIST_TYPE == 0){ // product
          dist += pix0 * pix1;
        }else if(DIST_TYPE == 1){ // l2
          _dist = (pix0 - pix1);
          dist += _dist*_dist;
        }else{ // error
          dist = invalid;
        }

      } // features
    } // pj
  } // pi
}

/*********************************************

        Forward (Bilin2d 2d)

 *********************************************/

template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void compute_dist_bilin2d_2d(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame0,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame1,
  int* ref_patch, scalar_t* prop_patch, int* ref, scalar_t* prop, //int* prop_i,
  bool* valid_ref, bool* valid_prop, int ps, int dilation, bool reflect_bounds,
  int patch_offset, scalar_t invalid, int* offsets,
  int C, int qH, int qW, int kH, int kW){
                  
  scalar_t pix0,pix1,w;
  scalar_t interp[2];
  for (int pi = 0; pi < ps; pi++){

    // -- ref height --
    ref[0] = ref_patch[0]+offsets[0]+dilation*(pi + patch_offset);
    ref[0] = reflect_bounds ? bounds(ref[0],qH) : ref[0];
    valid_ref[0] = check_interval(ref[0],0,qH);

    // -- proposed height --
    prop[0] = prop_patch[0]+dilation*(pi + patch_offset);
    prop[0] = reflect_bounds ? bounds_clip<scalar_t>(prop[0],kH) : prop[0];
    valid_prop[0] = check_interval<scalar_t>(prop[0],0,kH);

    for (int pj = 0; pj < ps; pj++){
      
      // -- ref width --
      ref[1] = ref_patch[1]+offsets[1]+dilation*(pj + patch_offset);
      ref[1] = reflect_bounds ? bounds(ref[1],qW) : ref[1];
      valid_ref[1] = check_interval(ref[1],0,qW);

      // -- prop width --
      prop[1] = prop_patch[1]+dilation*(pj + patch_offset);
      prop[1] = reflect_bounds ? bounds_clip<scalar_t>(prop[1],kW) : prop[1];
      valid_prop[1] = check_interval<scalar_t>(prop[1],0,kW);

      // -- ensure valid location --
      valid_ref[2] = true;
      valid_prop[2] = true;
      #pragma unroll
      for (int bool_idx=0; bool_idx<2; bool_idx++){
        valid_ref[2] = valid_ref[2] && valid_ref[bool_idx];
        valid_prop[2] = valid_prop[2] && valid_prop[bool_idx];
      }

      bool valid = valid_ref[2] and valid_prop[2];
      if (not valid){ continue; }

      // -- fill each channel --
      for (int ci = 0; ci < C; ci++){

        // -- reference value --
        pix0 = valid_ref[2] ? frame0[ci][ref[0]][ref[1]] : 0;

        // -- interpolate pixel value --
        bilin2d_interpolate(pix1,prop[0],prop[1],kH,kW,frame1[ci]);

        // -- compute dist --
        if(DIST_TYPE == 0){ // product
          dist += pix0 * pix1;
        }else if(DIST_TYPE == 1){ // l2
          dist += (pix0 - pix1)*(pix0 - pix1);
        }else{ // error
          dist = invalid;
        }

      }
    }
  }
}


/*********************************************

        Backward (Compute Distances 2d)

 *********************************************/


template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void update_bwd_patch_2d(
    torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> grad_frame0,
    torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> grad_frame1,
    const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame0,
    const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame1,
    scalar_t weight, int* ref_patch, int* prop_patch,
    int ps, int dilation, bool reflect_bounds, int patch_offset,
    int ftr_start, int ftr_end,
    int* ref, int* prop, bool* valid_ref, bool* valid_prop, bool valid,
    int* offsets, int qH, int qW, int kH, int kW, scalar_t pix0, scalar_t pix1){

    for (int pi = 0; pi < ps; pi++){

      // -- ref patch --
      ref[0] = ref_patch[0]+offsets[0]+dilation*(pi + patch_offset);
      ref[0] = reflect_bounds ? bounds(ref[0],qH) : ref[0];
      valid_ref[0] = check_interval(ref[0],0,qH);

      // -- prop patch --
      prop[0] = prop_patch[0]+dilation*(pi + patch_offset);
      prop[0] = reflect_bounds ? bounds(prop[0],kH) : prop[0];
      valid_prop[0] = check_interval(prop[0],0,kH);

      for (int pj = 0; pj < ps; pj++){
        
        // -- ref patch --
        ref[1] = ref_patch[1]+offsets[1]+dilation*(pj + patch_offset);
        ref[1] = reflect_bounds ? bounds(ref[1],qW) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,qW);

        // -- prop patch --
        prop[1] = prop_patch[1]+dilation*(pj + patch_offset);
        prop[1] = reflect_bounds ? bounds(prop[1],kW) : prop[1];
        valid_prop[1] = check_interval(prop[1],0,kW);

        // -- ensure valid location --
        valid_ref[2] = true;
        valid_prop[2] = true;
        #pragma unroll
        for (int bool_idx=0; bool_idx<2; bool_idx++){
          valid_ref[2] = valid_ref[2] && valid_ref[bool_idx];
          valid_prop[2] = valid_prop[2] && valid_prop[bool_idx];
        }
        valid = valid_ref[2] && valid_prop[2];
        if (not valid) { continue; }
        
        // -- fill each channel --
        for (int iftr = ftr_start; iftr < ftr_end; iftr++){

          if (DIST_TYPE == 0){ // prod
            pix0 = weight*frame0[iftr][ref[0]][ref[1]];
            pix1 = weight*frame1[iftr][prop[0]][prop[1]];
            atomicAdd(&(grad_frame0[iftr][ref[0]][ref[1]]),pix1);
            atomicAdd(&(grad_frame1[iftr][prop[0]][prop[1]]),pix0);
          }else if(DIST_TYPE == 1){ // l2 norm
            pix0 = frame0[iftr][ref[0]][ref[1]];
            pix1 = frame1[iftr][prop[0]][prop[1]];
            scalar_t pix = 2 * weight * (pix0 - pix1);
            atomicAdd(&grad_frame0[iftr][ref[0]][ref[1]],pix);
            atomicAdd(&grad_frame1[iftr][prop[0]][prop[1]],-pix);
          }

        } // features
      } // pj
    } // pi 

}


template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void update_bwd_bilin2d_patch_2d(
    torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> grad_frame0,
    torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> grad_frame1,
    const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame0,
    const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame1,
    scalar_t* acc_dFlows, scalar_t weight, int* ref_patch, scalar_t* prop_patch,
    int ps, int dilation, bool reflect_bounds,
    int patch_offset, // int ftr_start, int ftr_end,
    // int* ref, scalar_t* prop, int* prop_i,
    bool* valid_ref, bool* valid_prop, bool valid, int* offsets,
    int qH, int qW, int kH, int kW){

    scalar_t dDists,pix0,pix1;
    int ref[2];
    int prop_i[2];
    scalar_t prop[2];
    int F = frame0.size(0);
    int signH,signW;

    for (int pi = 0; pi < ps; pi++){

      // -- ref patch --
      ref[0] = ref_patch[0]+offsets[0]+dilation*(pi + patch_offset);
      ref[0] = reflect_bounds ? bounds(ref[0],qH) : ref[0];
      valid_ref[0] = check_interval(ref[0],0,qH);

      // -- prop patch --
      prop[0] = prop_patch[0]+dilation*(pi + patch_offset);
      signH = check_interval(prop[0],0,kH) ? 1 : -1;
      prop[0] = reflect_bounds ? bounds(prop[0],kH) : prop[0];
      valid_prop[0] = check_interval(prop[0],0,kH);

      for (int pj = 0; pj < ps; pj++){
        
        // -- ref patch --
        ref[1] = ref_patch[1]+offsets[1]+dilation*(pj + patch_offset);
        ref[1] = reflect_bounds ? bounds(ref[1],qW) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,qW);

        // -- prop patch --
        prop[1] = prop_patch[1]+dilation*(pj + patch_offset);
        signW = check_interval(prop[1],0,kW) ? 1 : -1;
        prop[1] = reflect_bounds ? bounds(prop[1],kW) : prop[1];
        valid_prop[1] = check_interval(prop[1],0,kW);

        // -- ensure valid location --
        valid_ref[2] = true;
        valid_prop[2] = true;
        #pragma unroll
        for (int bool_idx=0; bool_idx<2; bool_idx++){
          valid_ref[2] = valid_ref[2] && valid_ref[bool_idx];
          valid_prop[2] = valid_prop[2] && valid_prop[bool_idx];
        }
        valid = valid_ref[2] && valid_prop[2];
        if (not valid) { continue; }
        
        // -- fill each channel --
        for (int iftr = 0; iftr < F; iftr++){
          
          // -- reference value --
          pix0 = frame0[iftr][ref[0]][ref[1]];

          // -- interpolate pixel value --
          bilin2d_interpolate(pix1, prop[0], prop[1], kH, kW, frame1[iftr]);

          // -- update grad_frame0 --
          if (DIST_TYPE == 0){ // prod
            dDists = weight * pix1;
          }else if(DIST_TYPE == 1){ // l2 norm
            dDists = weight * 2 * (pix0 - pix1);
          }
          atomicAdd(&grad_frame0[iftr][ref[0]][ref[1]],dDists);

          // -- update grad_frame1 --
          if (DIST_TYPE == 0){ // prod
            dDists = weight * pix0;
          }else if(DIST_TYPE == 1){ // l2 norm
            dDists = -dDists;
          }
          bilin2d_assign(dDists,prop[0],prop[1],kH,kW,grad_frame1[iftr]);

          // -- update accumulated dflows --
          update_dFlows(acc_dFlows,dDists,prop[0],prop[1],kH,kW,signH,signW,frame1[iftr]);

        }
      }
    }

}


// template <typename scalar_t>
// __global__ void update_bwd_flow(
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_fflow,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_bflow,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fflow,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> bflow,
//     scalar_t* iweight, scalar_t* ref,
//     int stride0, int nW, int nHW, int TnHW, int locs_per_thread){

//   // -- unpack --
//   int qi;
//   int ibatch = blockIdx.y;
//   int raster_index = locs_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
//   int dir = threadIdx.y;
//   int T = fflow.size(1);
//   int H = fflow.size(3);
//   int W = fflow.size(4);
//   int tmp;
//   // int ref[3];
//   scalar_t refs[3];
//   int prop_i[3];
//   scalar_t prop[3];
//   scalar_t prop_time;
//   bool isFwd;

//   // -- fwd decl --
//   scalar_t v0,v1,gv0,gv1;
//   scalar_t dAdF0[2];
//   scalar_t dAdF1[2];

//   // -- get directional information --
//   isFwd = dir == 0;
//   int t_flow = threadIdx.z;//isFwd ? threadIdx.z : threadIdx.z+1;
//   auto flow = isFwd ? fflow[ibatch] : bflow[ibatch];
//   auto g_flow = isFwd ? grad_fflow[ibatch] : grad_bflow[ibatch];
//   auto pflow = isFwd ? pfflow[ibatch] : pbflow[ibatch];
//   auto g_pflow = isFwd ? grad_pfflow[ibatch] : grad_pbflow[ibatch];

//   // -- get location --
//   for (int loc = 0; loc < locs_per_thread; loc++){

//     // -- get reference location --
//     qi = raster_index + loc;
//     if (qi >= TnHW){ break; } 
//     get_pixel_loc(ref,qi,tmp,stride0,nW,nHW,H,W);

//     // -- init/reset --
//     v0 = 0;
//     v1 = 0;
//     gv0 = 0;
//     gv1 = 0;

//     // -- directional indexing --
//     int t_inc = 1;
//     int Acc_time_start = t_flow-ref[0];
//     int t_end = (T-1)-t_flow;
//     if (Acc_time_start < 0) { return; }

//     // -- write location --
//     refs[0] = __int2float_rn(ref[0]);
//     if (Acc_time_start == 0){
//       refs[1] = __int2float_rn(ref[1]);
//       refs[2] = __int2float_rn(ref[2]);
//     }else{
//       refs[1] = ref[1] + pflow[ref[0]][Acc_time_start-1][1][ref[1]][ref[2]];
//       refs[2] = ref[2] + pflow[ref[0]][Acc_time_start-1][0][ref[1]][ref[2]];
//     }

//     // -- iterate across accumulated flows --
//     for(int tx=0; tx < t_end; tx++){

//       // -- read gradient --
//       gv0 = g_pflow[ref[0]][Acc_time_start+tx][0][ref[1]][ref[2]];
//       gv1 = g_pflow[ref[0]][Acc_time_start+tx][1][ref[1]][ref[2]];

//       // -- update dA[i]dF[j] as dAdFj[i] --
//       if (tx==0){
//         dAdF0[0] = 1;
//         dAdF0[1] = 0;
//         dAdF1[0] = 0;
//         dAdF1[1] = 1;
//       }else{

//         // -- update proposed location --
//         prop[1] = __int2float_rn(ref[1]) +                      \
//           pflow[ref[0]][Acc_time_start+tx-1][1][ref[1]][ref[2]];
//         prop[2] = __int2float_rn(ref[2]) +                      \
//           pflow[ref[0]][Acc_time_start+tx-1][0][ref[1]][ref[2]];

//         // -- update weights --
//         update_weights(dAdF0,dAdF1,prop,H,W,tx,flow[t_flow+t_inc*tx],
//                        dev[ibatch][qi][t_flow][tx]);
//       }

//       // -- assign to each of the 4 interpolated flow values --
//       assign_bilin2d(dAdF0,dAdF1,gv0,gv1,refs,H,W,g_flow[t_flow]);

//     }

//   }
// }

// template<typename scalar_t, int DIST_TYPE>
// __device__ __forceinline__ 
// void update_bwd_flow_accum_flows_2d(
//     torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> g_flow,
//     const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> flow,
//     scalar_t prop_time, scalar_t* iweight, int* ref,
//     int H, int W){


//     // 
//     // -- update optical flow (forward or backward) --
//     //

//     // -- pick a direction --
//     int prop_i[3];
//     int delta_t = __float2int_rd(prop_time) - ref[0];
//     int inc = delta_t > 0 ? 1 : -1;
//     // auto flow = delta_t > 0 ? fflow : bflow;
//     // auto g_flow = delta_t > 0 ? grad_fflow : grad_bflow;
//     int src_t = __float2int_rd(prop_time) - inc;
//     int delta_ta = abs(delta_t);

//     // -- init incs --
//     scalar_t w;
//     scalar_t interm[3],interm_n[3];
//     scalar_t gradW = 0;
//     scalar_t gradH = 0;
//     scalar_t gH,gW,v0,v1,gH_f,gW_f;
//     int tx;

//     // -- setup --
// #pragma unroll
//     for (int idx=0;idx<3;idx++){
//       interm[idx] = __int2float_rd(ref[idx]);
//       interm_n[idx] = interm[idx];
//     }

//     // -- compute gradient across time --
//     for (int _tx = 0; _tx < delta_ta; _tx++){

//       scalar_t gradW_t = 0;
//       scalar_t gradH_t = 0;

//       tx = inc * _tx;
//       v0,v1 = 0,0;
//       interm[0] = __float2int_rd(ref[0]) + tx;
//       prop_i[0] = __float2int_rd(interm[0]);
//       interm[1] = interm_n[1];
//       interm[2] = interm_n[2];

// #pragma unroll
//       for (int ix=0;ix<2;ix++){
// #pragma unroll
//         for (int jx=0;jx<2;jx++){
//           prop_i[1] = __float2int_rd(interm[1]+ix);
//           gH = max(0.,1-fabs(prop_i[1]-interm[1]));
//           prop_i[2] = __float2int_rd(interm[2]+jx);
//           gW = max(0.,1-fabs(prop_i[2]-interm[2]));

//           // // -- compute direction --
//           // bool left0 = (interm[1]-prop_i[1]) < 0;
//           // bool left1 = (interm[2]-prop_i[2]) < 0;
//           // bool right0 = (interm[1]-prop_i[1]) > 0;
//           // bool right1 = (interm[2]-prop_i[2]) > 0;

//           // -- ensure legal inds --
//           prop_i[1] = bounds(prop_i[1],H);
//           prop_i[2] = bounds(prop_i[2],W);

//           // -- read --
//           v0 = flow[prop_i[0]][0][prop_i[1]][prop_i[2]];
//           v1 = flow[prop_i[0]][1][prop_i[1]][prop_i[2]];

//           // -- update next location --
//           interm_n[1] += gH*gW*v0;
//           interm_n[2] += gH*gW*v1;

//           if (_tx > 0){ // first iteration (_tx=0) only updates interm_n

//             // -- update gradient --
//             // gradW_t += left0 ? gH*v0 : (right0 ? -gH*v0 : 0);
//             // gradH_t += left1 ? gW*v1 : (right1 ? -gW*v1 : 0);

//             // -- update flows --
//             // if (_tx > 1){
//             atomicAdd(&g_flow[prop_i[0]][0][prop_i[1]][prop_i[2]],gH*gW*iweight[2]);
//             atomicAdd(&g_flow[prop_i[0]][1][prop_i[1]][prop_i[2]],gH*gW*iweight[1]);
//             // }

//           }

//         }
//       }

//       // -- accumulate across time --
//       gradW += _tx > 0 ? gradW*gradW_t : 1;
//       gradH += _tx > 0 ? gradH*gradH_t : 1;
//       // gradW_t = gradW;
//       // gradH_t = gradH;

//     }

//     // -- update --
//     atomicAdd(&g_flow[ref[0]][0][ref[1]][ref[2]],gradW*iweight[2]);
//     atomicAdd(&g_flow[ref[0]][1][ref[1]][ref[2]],gradH*iweight[1]);

// }


