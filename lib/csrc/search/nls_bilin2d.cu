#include <cuda/std/type_traits>
#include <cstdio>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <assert.h>


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



template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void compute_dist_bilin2d(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
  int* ref_patch, scalar_t* prop_patch, int* ref, scalar_t* prop, int* prop_i,
  bool* valid_ref, bool* valid_prop,
  int ps, int pt, int dilation, bool reflect_bounds,
  int patch_offset, scalar_t invalid, int T, int C, int H, int W){
                  
  scalar_t pix0,pix1,w;
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
      ref[1] = ref_patch[1]+dilation*(pi + patch_offset);
      ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
      valid_ref[1] = check_interval(ref[1],0,H);

      // -- proposed height --
      prop[1] = prop_patch[1]+dilation*(pi + patch_offset);
      prop[1] = reflect_bounds ? bounds<scalar_t>(prop[1],H) : prop[1];
      valid_prop[1] = check_interval<scalar_t>(prop[1],0,H);

      for (int pj = 0; pj < ps; pj++){
        
        // -- ref width --
        ref[2] = ref_patch[2]+dilation*(pj + patch_offset);
        ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
        valid_ref[2] = check_interval(ref[2],0,W);

        // -- prop width --
        prop[2] = prop_patch[2]+dilation*(pj + patch_offset);
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

        // -- set time --
        prop_i[0] = __float2int_rn(prop[0]);

        // -- fill each channel --
        for (int ci = 0; ci < C; ci++){

          // -- reference value --
          pix0 = valid_ref[3] ? vid0[ref[0]][ci][ref[1]][ref[2]] : 0;

          // -- interpolate pixel value --
          bilin2d_interpolate(pix1, prop[1], prop[2], H, W, vid1[prop_i[0]][ci]);

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
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
    scalar_t weight, int* ref_patch, scalar_t* prop_patch,
    int ps, int pt, int dilation, bool reflect_bounds,
    int patch_offset, int ftr_start, int ftr_end,
    int* ref, scalar_t* prop, int* prop_i, bool* valid_ref, bool* valid_prop,
    bool valid, int T, int H, int W){

    scalar_t pix0,pix1,pix;
    scalar_t interp[2];
    scalar_t dDists;
    for (int pk = 0; pk < pt; pk++){

      // -- ref patch --

      ref[0] = bounds(ref_patch[0]+pk,T);
      valid_ref[0] = check_interval(ref[0],0,T);

      // -- prop patch --
      prop[0] = bounds(prop_patch[0]+pk,T);
      valid_prop[0] = check_interval(prop[0],0,T);

      for (int pi = 0; pi < ps; pi++){

        // -- ref patch --
        ref[1] = ref_patch[1]+dilation*(pi + patch_offset);
        ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,H);

        // -- prop patch --
        prop[1] = prop_patch[1]+dilation*(pi + patch_offset);
        prop[1] = reflect_bounds ? bounds(prop[1],H) : prop[1];
        valid_prop[1] = check_interval(prop[1],0,H);

        for (int pj = 0; pj < ps; pj++){
          
          // -- ref patch --
          ref[2] = ref_patch[2]+dilation*(pj + patch_offset);
          ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
          valid_ref[2] = check_interval(ref[2],0,W);

          // -- prop patch --
          prop[2] = prop_patch[2]+dilation*(pj + patch_offset);
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
          
          // -- set time --
          prop_i[0] = __float2int_rn(prop[0]);

          // -- fill each channel --
          for (int iftr = ftr_start; iftr < ftr_end; iftr++){
            
            // -- reference value --
            pix0 = vid0[ref[0]][iftr][ref[1]][ref[2]];
  
            // -- interpolate pixel value --
            bilin2d_interpolate(pix1, prop[1], prop[2], H, W, vid1[prop_i[0]][iftr]);

            // -- update vid0 --
            if (DIST_TYPE == 0){ // prod
              dDists = weight * pix1;
            }else if(DIST_TYPE == 1){ // l2 norm
              dDists = weight * 2 * (pix0 - pix1);
            }
            atomicAdd(&grad_vid0[ref[0]][iftr][ref[1]][ref[2]],dDists);

            // -- update vid1 --
            if (DIST_TYPE == 0){ // prod
              dDists = weight * pix0;
            }else if(DIST_TYPE == 1){ // l2 norm
              dDists = - dDists;
            }
            bilin2d_assign(dDists,prop[1],prop[2],H,W,grad_vid1[prop_i[0]][iftr]);

          }
        }
      }
    }

}


template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void update_bwd_bilin2d_vidflows(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid0,
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid1,
    // torch::TensorAccessor<scalar_t,5,torch::RestrictPtrTraits,int32_t> grad_flows,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
    // const torch::TensorAccessor<scalar_t,5,torch::RestrictPtrTraits,int32_t> flows,
    scalar_t* acc_dFlows,
    scalar_t weight, int* ref_patch, scalar_t* prop_patch,
    int ps, int pt, int dilation, int stride0,
    bool reflect_bounds, int patch_offset,
    int ftr_start, int ftr_end,
    int* ref, scalar_t* prop, int* prop_i,
    bool* valid_ref, bool* valid_prop, bool valid,
    int T, int H, int W){

    scalar_t pix0,pix1;
    scalar_t interp[2];
    scalar_t dDists;
    for (int pk = 0; pk < pt; pk++){

      // -- ref patch --

      ref[0] = bounds(ref_patch[0]+pk,T);
      valid_ref[0] = check_interval(ref[0],0,T);

      // -- prop patch --
      prop[0] = bounds(prop_patch[0]+pk,T);
      valid_prop[0] = check_interval(prop[0],0,T);

      for (int pi = 0; pi < ps; pi++){

        // -- ref patch --
        ref[1] = ref_patch[1]+dilation*(pi + patch_offset);
        ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,H);

        // -- prop patch --
        prop[1] = prop_patch[1]+dilation*(pi + patch_offset);
        prop[1] = reflect_bounds ? bounds(prop[1],H) : prop[1];
        valid_prop[1] = check_interval(prop[1],0,H);

        for (int pj = 0; pj < ps; pj++){
          
          // -- ref patch --
          ref[2] = ref_patch[2]+dilation*(pj + patch_offset);
          ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
          valid_ref[2] = check_interval(ref[2],0,W);

          // -- prop patch --
          prop[2] = prop_patch[2]+dilation*(pj + patch_offset);
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
          
          // -- bwd flow accumulation --
          scalar_t dDistAcc = static_cast<scalar_t>(0);

          // -- set time --
          prop_i[0] = __float2int_rn(prop[0]);

          // -- fill each channel --
          for (int iftr = ftr_start; iftr < ftr_end; iftr++){
            
            // -- reference value --
            pix0 = vid0[ref[0]][iftr][ref[1]][ref[2]];
  
            // -- interpolate pixel value --
            bilin2d_interpolate(pix1, prop[1], prop[2], H, W, vid1[prop_i[0]][iftr]);

            // -- update vid0 --
            if (DIST_TYPE == 0){ // prod
              dDists = weight * pix1;
            }else if(DIST_TYPE == 1){ // l2 norm
              dDists = weight * 2 * (pix0 - pix1);
            }
            atomicAdd(&grad_vid0[ref[0]][iftr][ref[1]][ref[2]],dDists);

            // -- update vid1 --
            if (DIST_TYPE == 0){ // prod
              dDists = weight * pix0;
            }else if(DIST_TYPE == 1){ // l2 norm
              dDists = -dDists;
            }
            bilin2d_assign(dDists,prop[1],prop[2],H,W,grad_vid1[prop_i[0]][iftr]);

            // -- update accumulated dflows --
            update_dFlows(acc_dFlows,dDists,prop[1],prop[2],H,W,vid1[prop_i[0]][iftr]);

          }

        }
      }
    }
}

