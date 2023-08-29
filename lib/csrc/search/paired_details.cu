
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

#include "shared_kernel.cu"
using namespace at;


/*********************************************

        Forward (Compute Distances 2d)

 *********************************************/


template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void compute_dist_2d(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame0,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame1,
  int* ref_patch, int* prop_patch, int* ref, int* prop,
  bool* valid_ref, bool* valid_prop,
  int ps, int dilation, bool reflect_bounds,
  int patch_offset, int* center_offsets, scalar_t invalid,
  int C, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t _dist){
                  
    
  for (int pi = 0; pi < ps; pi++){

    // -- ref height --
    ref[0] = (ref_patch[0]-center_offsets[0])+dilation*(pi + patch_offset);
    ref[0] = reflect_bounds ? bounds(ref[0],H) : ref[0];
    valid_ref[0] = check_interval(ref[0],0,H);

    // -- proposed height --
    prop[0] = (prop_patch[0]-center_offsets[1])+dilation*(pi + patch_offset);
    prop[0] = reflect_bounds ? bounds(prop[0],H) : prop[0];
    valid_prop[0] = check_interval(prop[0],0,H);

    for (int pj = 0; pj < ps; pj++){
      
      // -- ref width --
      ref[1] = (ref_patch[1]-center_offsets[2])+dilation*(pj + patch_offset);
      ref[1] = reflect_bounds ? bounds(ref[1],W) : ref[1];
      valid_ref[1] = check_interval(ref[1],0,W);

      // -- prop width --
      prop[1] = (prop_patch[1]-center_offsets[3])+dilation*(pj + patch_offset);
      prop[1] = reflect_bounds ? bounds(prop[1],W) : prop[1];
      valid_prop[1] = check_interval(prop[1],0,W);

      // -- ensure valid location --
      valid_ref[2] = true;
      valid_prop[2] = true;
      #pragma unroll
      for (int bool_idx=0; bool_idx<2; bool_idx++){
        valid_ref[2] = valid_ref[2] && valid_ref[bool_idx];
        valid_prop[2] = valid_prop[2] && valid_prop[bool_idx];
      }

      // -- fill each channel --
      for (int ci = 0; ci < C; ci++){

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


// template<typename scalar_t, int DIST_TYPE>
// __device__ __forceinline__ 
// void compute_dist_bilin2d_2d_asdf(scalar_t& dist,
//   const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame0,
//   const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame1,
//   int* ref_patch, scalar_t* prop_patch, int* ref, scalar_t* prop, int* prop_i,
//   bool* valid_ref, bool* valid_prop, int ps, int pt, int dilation,
//   bool reflect_bounds, int patch_offset, int* center_offsets){
//   return;
// }

template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void compute_dist_bilin2d_2d(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame0,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> frame1,
  int* ref_patch, scalar_t* prop_patch, int* ref, scalar_t* prop, int* prop_i,
  bool* valid_ref, bool* valid_prop,
  int ps, int dilation, bool reflect_bounds,
  int patch_offset, int* center_offsets, scalar_t invalid,
  int C, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t w){
                  
  scalar_t interp[2];
  for (int pi = 0; pi < ps; pi++){

    // -- ref height --
    ref[0] = (ref_patch[0]-center_offsets[0])+dilation*(pi + patch_offset);
    ref[0] = reflect_bounds ? bounds(ref[0],H) : ref[0];
    valid_ref[0] = check_interval(ref[0],0,H);

    // -- proposed height --
    prop[0] = (prop_patch[0]-center_offsets[1])+dilation*(pi + patch_offset);
    prop[0] = reflect_bounds ? bounds<scalar_t>(prop[0],H) : prop[0];
    valid_prop[0] = check_interval<scalar_t>(prop[0],0,H);

    for (int pj = 0; pj < ps; pj++){
      
      // -- ref width --
      ref[1] = (ref_patch[1]-center_offsets[2])+dilation*(pj + patch_offset);
      ref[1] = reflect_bounds ? bounds(ref[1],W) : ref[1];
      valid_ref[1] = check_interval(ref[1],0,W);

      // -- prop width --
      prop[1] = (prop_patch[1]-center_offsets[3])+dilation*(pj + patch_offset);
      prop[1] = reflect_bounds ? bounds<scalar_t>(prop[1],W) : prop[1];
      valid_prop[1] = check_interval<scalar_t>(prop[1],0,W);

      // -- ensure valid location --
      valid_ref[2] = true;
      valid_prop[2] = true;
      #pragma unroll
      for (int bool_idx=0; bool_idx<2; bool_idx++){
        valid_ref[2] = valid_ref[2] && valid_ref[bool_idx];
        valid_prop[2] = valid_prop[2] && valid_prop[bool_idx];
      }

      // -- fill each channel --
      for (int ci = 0; ci < C; ci++){

        // -- reference value --
        pix0 = valid_ref[2] ? frame0[ci][ref[0]][ref[1]] : 0;

        // -- interpolate pixel value --
        pix1 = 0;
        #pragma unroll
        for (int ix=0;ix<2;ix++){
          #pragma unroll
          for (int jx=0;jx<2;jx++){

            // -- interpolation weight --
            prop_i[0] = __float2int_rd(prop[0]+ix);
            interp[0] = max(0.,1-fabs(prop_i[0]-prop[0]));
            prop_i[1] = __float2int_rd(prop[1]+jx);
            interp[1] = max(0.,1-fabs(prop_i[1]-prop[1]));
            w = interp[0] * interp[1];

            // -- ensure legal bounds --
            prop_i[0] = bounds(prop_i[0],H);
            prop_i[1] = bounds(prop_i[1],W);

            // -- update --
            pix1 += valid_prop[2] ? w*frame1[ci][prop_i[0]][prop_i[1]] : 0;
          }
        }

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
    int ps, int pt, int dilation, bool reflect_bounds,
    int* center_offsets, int patch_offset,
    int iftr, int ftr_start, int ftr_end,
    int* ref, int* prop, bool* valid_ref, bool* valid_prop, bool valid,
    int H, int W, scalar_t pix0, scalar_t pix1, scalar_t pix, int i1){

    for (int pi = 0; pi < ps; pi++){

      // -- ref patch --
      ref[0] = (ref_patch[0]-center_offsets[0])+dilation*(pi + patch_offset);
      ref[0] = reflect_bounds ? bounds(ref[0],H) : ref[0];
      valid_ref[0] = check_interval(ref[0],0,H);

      // -- prop patch --
      prop[0] = (prop_patch[0]-center_offsets[1])+dilation*(pi + patch_offset);
      prop[0] = reflect_bounds ? bounds(prop[0],H) : prop[0];
      valid_prop[0] = check_interval(prop[0],0,H);

      for (int pj = 0; pj < ps; pj++){
        
        // -- ref patch --
        ref[1] = (ref_patch[1]-center_offsets[2])+dilation*(pj + patch_offset);
        ref[1] = reflect_bounds ? bounds(ref[1],W) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,W);

        // -- prop patch --
        prop[1] = (prop_patch[1]-center_offsets[3])+dilation*(pj + patch_offset);
        prop[1] = reflect_bounds ? bounds(prop[1],W) : prop[1];
        valid_prop[1] = check_interval(prop[1],0,W);

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
        for (iftr = ftr_start; iftr < ftr_end; iftr++){

          if (DIST_TYPE == 0){ // prod
            pix0 = weight*frame0[iftr][ref[0]][ref[1]];
            pix1 = weight*frame1[iftr][prop[0]][prop[1]];
            atomicAdd(&(grad_frame0[iftr][ref[0]][ref[1]]),pix1);
            atomicAdd(&(grad_frame1[iftr][prop[0]][prop[1]]),pix0);
          }else if(DIST_TYPE == 1){ // l2 norm
            pix0 = frame0[iftr][ref[0]][ref[1]];
            pix1 = frame1[iftr][prop[0]][prop[1]];
            pix = 2 * weight * (pix0 - pix1);
            atomicAdd(&grad_frame0[iftr][ref[0]][ref[1]],pix);
            atomicAdd(&grad_frame1[iftr][prop[0]][prop[1]],-pix);
          }

        } // features
      } // pj
    } // pi 

}




