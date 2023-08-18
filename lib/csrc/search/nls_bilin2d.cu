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
#include "shared_kernel.cu"

using namespace at;

// #include "shared_kernel.cu"

template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void compute_dist_bilin2d(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
  int* ref_patch, scalar_t* prop_patch, int* ref, scalar_t* prop, int* prop_i,
  bool* valid_ref, bool* valid_prop,
  int ps, int pt, int dilation, bool reflect_bounds,
  int patch_offset, int* center_offsets, scalar_t invalid,
  int T, int C, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t w){
                  
  scalar_t interp[2];
  for (int pk = 0; pk < pt; pk++){

    // -- reference time --
    ref[0] = bounds(ref_patch[0] + pk,T);
    valid_ref[0] = check_interval(ref[0],0,T);

    // -- proposed time [always an "int" in value] --
    prop[0] = bounds<scalar_t>(prop_patch[0] + pk,T);
    valid_prop[0] = check_interval<scalar_t>(prop[0],0,T);
    
    for (int pi = 0; pi < ps; pi++){

      // -- ref height --
      ref[1] = (ref_patch[1]-center_offsets[0])+dilation*(pi + patch_offset);
      ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
      valid_ref[1] = check_interval(ref[1],0,H);

      // -- proposed height --
      prop[1] = (prop_patch[1]-center_offsets[1])+dilation*(pi + patch_offset);
      prop[1] = reflect_bounds ? bounds<scalar_t>(prop[1],H) : prop[1];
      valid_prop[1] = check_interval<scalar_t>(prop[1],0,H);

      for (int pj = 0; pj < ps; pj++){
        
        // -- ref width --
        ref[2] = (ref_patch[2]-center_offsets[2])+dilation*(pj + patch_offset);
        ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
        valid_ref[2] = check_interval(ref[2],0,W);

        // -- prop width --
        prop[2] = (prop_patch[2]-center_offsets[3])+dilation*(pj + patch_offset);
        prop[2] = reflect_bounds ? bounds<scalar_t>(prop[2],W) : prop[2];
        valid_prop[2] = check_interval<scalar_t>(prop[2],0,W);

        // -- ensure valid location --
        valid_ref[3] = true;
        valid_prop[3] = true;
        #pragma unroll
        for (int bool_idx=0; bool_idx<3; bool_idx++){
          valid_ref[3] = valid_ref[3] && valid_ref[bool_idx];
          valid_prop[3] = valid_prop[3] && valid_prop[bool_idx];
        }
        ref[0] = 0;
        ref[1] = 0;
        ref[2] = 0;
        valid_ref[3] = true;

        // -- set time --
        prop_i[0] = __float2int_rn(prop[0]);

        // -- fill each channel --
        for (int ci = 0; ci < C; ci++){

          // -- reference value --
          pix0 = valid_ref[3] ? vid0[ref[0]][ci][ref[1]][ref[2]] : 0;

          // -- interpolate pixel value --
          pix0 = 0;
          #pragma unroll
          for (int ix=0;ix<2;ix++){
            // #pragma unroll
            for (int jx=0;jx<2;jx++){
              prop_i[1] = __float2int_rd(prop[1]+static_cast<scalar_t>(ix));
              interp[0] = max(0.,1-fabs(static_cast<scalar_t>(prop_i[1])-prop[1]));
              prop_i[2] = __float2int_rd(prop[2]+static_cast<scalar_t>(jx));
              interp[1] = max(0.,1-fabs(static_cast<scalar_t>(prop_i[2])-prop[2]));

              // -- compute weight --
              w = interp[0] * interp[1];

              // -- ensure legal bounds --
              prop_i[1] = bounds(prop_i[1],H);
              prop_i[2] = bounds(prop_i[2],W);

              // -- update --
              pix1 += valid_prop[3] ? w*vid1[prop_i[0]][ci][prop_i[1]][prop_i[2]] : 0;
            }
          }

          // int i1 = __float2int_rn(round(prop[1]));
          // int i2 = __float2int_rn(round(prop[2]));
          // pix1 = vid1[prop_i[0]][ci][i1][i2];

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
}




template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void update_bwd_patch_bilin2d(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid0,
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid1,
    // torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_fflow,
    // torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_bflow,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
    // const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> fflow,
    // const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> bflow,
    // torch::TensorAccessor<int,3,torch::RestrictPtrTraits,int32_t> count0,
    // torch::TensorAccessor<int,3,torch::RestrictPtrTraits,int32_t> count1,
    scalar_t weight,
    int* ref_patch, scalar_t* prop_patch,
    int ps, int pt, int dilation, bool reflect_bounds,
    int* center_offsets, int patch_offset,
    int iftr, int ftr_start, int ftr_end,
    int* ref, scalar_t* prop, int* prop_i,
    bool* valid_ref, bool* valid_prop, bool valid,
    int T, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t pix, int i1){

    scalar_t interp[2];
    for (int pk = 0; pk < pt; pk++){

      // -- ref patch --

      ref[0] = bounds(ref_patch[0]+pk,T);
      valid_ref[0] = check_interval(ref[0],0,T);

      // -- prop patch --
      prop[0] = bounds(prop_patch[0]+pk,T);
      valid_prop[0] = check_interval(prop[0],0,T);

      for (int pi = 0; pi < ps; pi++){

        // -- ref patch --
        ref[1] = (ref_patch[1]-center_offsets[0])+dilation*(pi + patch_offset);
        ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,H);

        // -- prop patch --
        prop[1] = (prop_patch[1]-center_offsets[1])+dilation*(pi + patch_offset);
        prop[1] = reflect_bounds ? bounds(prop[1],H) : prop[1];
        valid_prop[1] = check_interval(prop[1],0,H);

        for (int pj = 0; pj < ps; pj++){
          
          // -- ref patch --
          ref[2] = (ref_patch[2]-center_offsets[2])+dilation*(pj + patch_offset);
          ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
          valid_ref[2] = check_interval(ref[2],0,W);

          // -- prop patch --
          prop[2] = (prop_patch[2]-center_offsets[3])+dilation*(pj + patch_offset);
          prop[2] = reflect_bounds ? bounds(prop[2],W) : prop[2];
          valid_prop[2] = check_interval(prop[2],0,W);

          // -- ensure valid location --
          valid_ref[3] = true;
          valid_prop[3] = true;
          #pragma unroll
          for (int bool_idx=0; bool_idx<3; bool_idx++){
            valid_ref[3] = valid_ref[3] && valid_ref[bool_idx];
            valid_prop[3] = valid_prop[3] && valid_prop[bool_idx];
          }
          valid = valid_ref[3] && valid_prop[3];
          if (not valid) { continue; }
          
          // -- add count --
          // if (ftr_start == 0){
          //   if ((valid_ref[3])){
          //   // if ((valid_ref[3]) && (i1==0)){
          //     atomicAdd(&(count0[ref[0]][ref[1]][ref[2]]),1);
          //   }
          //   // only add if the i0 is different from the other i0 value
          //   // ?equally? only add if i1 is same?...
          //   if ((valid_prop[3])){// && (valid_ref[3])){
          //     atomicAdd(&(count1[prop[0]][prop[1]][prop[2]]),1);
          //   }
          //   // if ((valid_prop[3]) && (i1==0)){
          //   //   atomicAdd(&(count1[prop[0]][prop[1]][prop[2]]),1);
          //   // }
          // }

          // -- set time --
          prop_i[0] = __float2int_rn(prop[0]);

          // -- fill each channel --
          for (iftr = ftr_start; iftr < ftr_end; iftr++){
            
            // -- reference value --
            pix0 = vid0[ref[0]][iftr][ref[1]][ref[2]];
  
            // -- interpolate pixel value --
            pix1 = 0;
            scalar_t w = 0;
            #pragma unroll
            for (int ix=0;ix<2;ix++){
              prop_i[1] = __float2int_rd(prop[1]+ix);
              interp[0] = max(0.,1-fabs(prop_i[1]-prop[1]));
              #pragma unroll
              for (int jx=0;jx<2;jx++){
                prop_i[2] = __float2int_rd(prop[2]+jx);
                interp[1] = max(0.,1-fabs(prop_i[2]-prop[2]));

                // -- interpolation weight --
                w = interp[0] * interp[1];

                // -- ensure legal bounds --
                prop_i[1] = bounds(prop_i[1],H);
                prop_i[2] = bounds(prop_i[2],W);

                // -- update pixel --
                pix1 += w*vid1[prop_i[0]][iftr][prop_i[1]][prop_i[2]];
              }
            }

            // -- update vid0 --
            pix = 2 * weight * (pix0 - pix1);
            if (DIST_TYPE == 0){ // prod
              atomicAdd(&(grad_vid0[ref[0]][iftr][ref[1]][ref[2]]),weight*pix1);
            }else if(DIST_TYPE == 1){ // l2 norm
              atomicAdd(&grad_vid0[ref[0]][iftr][ref[1]][ref[2]],pix);
            }

            // -- update vid1 --
            scalar_t wpix0 = weight*pix0;
            #pragma unroll
            for (int ix=0;ix<2;ix++){
              prop_i[1] = __float2int_rd(prop[1]+ix);
              interp[0] = max(0.,1-fabs(prop_i[1]-prop[1]));
              #pragma unroll
              for (int jx=0;jx<2;jx++){
                prop_i[2] = __float2int_rd(prop[2]+jx);
                interp[1] = max(0.,1-fabs(prop_i[2]-prop[2]));

                // -- interpolation weighting --
                w = interp[0] * interp[1];

                // -- ensure legal bounds --
                prop_i[1] = bounds(prop_i[1],H);
                prop_i[2] = bounds(prop_i[2],W);

                if (DIST_TYPE == 0){ // prod
                  atomicAdd(&(grad_vid1[prop_i[0]][iftr][prop_i[1]][prop_i[2]]),
                            w*wpix0);
                }else if(DIST_TYPE == 1){ // l2 norm
                  atomicAdd(&grad_vid1[prop_i[0]][iftr][prop_i[1]][prop_i[2]],-w*pix);
                }
              }
            }

          }
        }
      }
    }

}


template<typename scalar_t>
__device__ __forceinline__ 
void update_bwd_flows_bilin2d(
    // torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid0,
    // torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid1,
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_fflow,
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_bflow,
    // const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
    // const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> fflow,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> bflow,
    // torch::TensorAccessor<int,3,torch::RestrictPtrTraits,int32_t> count0,
    // torch::TensorAccessor<int,3,torch::RestrictPtrTraits,int32_t> count1,
    scalar_t* iweight, int* ref, scalar_t* prop, int* prop_i,
    int ps, int pt, int dilation, bool reflect_bounds,
    int* center_offsets, int patch_offset,
    int iftr, int ftr_start, int ftr_end,
    bool* valid_ref, bool* valid_prop, bool valid,
    int T, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t pix, int i1){


    // 
    // -- update optical flow (forward or backward) --
    //

    // -- pick a direction --
    int delta_t = __float2int_rd(prop[0]) - ref[0];
    int inc = delta_t > 0 ? 1 : -1;
    auto flow = delta_t > 0 ? fflow : bflow;
    auto g_flow = delta_t > 0 ? grad_fflow : grad_bflow;
    int src_t = __float2int_rd(prop[0]) - inc;
    int delta_ta = abs(delta_t);

    // -- init incs --
    scalar_t w;
    scalar_t interm[3],interm_n[3];
    scalar_t grad0 = 0;
    scalar_t grad1 = 0;
    scalar_t g0,g1,v0,v1,g0_f,g1_f;
    int tx;

    // -- setup --
#pragma unroll
    for (int idx=0;idx<3;idx++){
      interm[idx] = __int2float_rd(ref[idx]);
      interm_n[idx] = interm[idx];
    }

    // -- compute gradient across time --
    for (int _tx = 0; _tx < delta_ta; _tx++){

      scalar_t grad0_t = 0;
      scalar_t grad1_t = 0;

      tx = inc * _tx;
      v0,v1 = 0,0;
      interm[0] = interm[0]+inc;
      prop_i[0] = __float2int_rd(interm[0]);
      interm[1] = interm_n[1];
      interm[2] = interm_n[2];

#pragma unroll
      for (int ix=0;ix<2;ix++){
        prop_i[1] = __float2int_rd(interm[1]+ix);
        g0 = max(0.,1-fabs(prop_i[1]-interm[1]));
#pragma unroll
        for (int jx=0;jx<2;jx++){
          prop_i[2] = __float2int_rd(interm[2]+jx);
          g1 = max(0.,1-fabs(prop_i[2]-interm[2]));

          // -- compute direction --
          bool left0 = (prop_i[1]-interm[1]) < 0;
          bool left1 = (prop_i[2]-interm[2]) < 0;

          // -- ensure legal inds --
          prop_i[1] = bounds(prop_i[1],H);
          prop_i[2] = bounds(prop_i[2],H);

          // -- read --
          v0 = flow[prop_i[0]][0][prop_i[1]][prop_i[2]];
          v1 = flow[prop_i[0]][1][prop_i[1]][prop_i[2]];

          // -- update next location --
          interm_n[1] += g0*g1*v0;
          interm_n[2] += g0*g1*v1;

          // -- update gradient --
          grad0_t += left0 ? g1*v0 : -g1*v0;
          grad1_t += left1 ? g0*v1 : -g0*v1;
        }
      }

      // -- accumulate across time --
      grad0 += _tx > 1 ? grad0*grad0_t : grad0_t;
      grad1 += _tx > 1 ? grad1*grad1_t : grad1_t;
      // grad0 += 1;
      // grad1 += 1;

    }

    // -- udpate gradient --
    prop_i[0] = bounds(src_t,T);
#pragma unroll
    for (int ix=0;ix<2;ix++){
      prop_i[1] = __float2int_rd(prop[1]+ix);
      g0 = max(0.,1-fabs(prop_i[1]-prop[1]));
#pragma unroll
      for (int jx=0;jx<2;jx++){
        prop_i[2] = __float2int_rd(prop[2]+jx);
        g1 = max(0.,1-fabs(prop_i[2]-prop[2]));

        // -- finish weights --
        w = g0 * g1;

        // -- ensure legal bounds --
        prop_i[1] = bounds(prop_i[1],H);
        prop_i[2] = bounds(prop_i[2],W);

        // -- update --
        atomicAdd(&g_flow[prop_i[0]][0][prop_i[1]][prop_i[2]],\
                  w*iweight[2]*grad0);
        atomicAdd(&g_flow[prop_i[0]][1][prop_i[1]][prop_i[2]],\
                  w*iweight[1]*grad1);
      }
    }

}


