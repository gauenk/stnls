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
#include "shared_tile_kernels.cu"


using namespace at;


template<typename scalar_t>
__device__ __forceinline__ 
void fill_non_local_patch_bilin2d(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> stack,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> counts,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid,
    scalar_t weight, int ps, int pt, int dilation, bool reflect_bounds,
    int* ref_patch, scalar_t* nl_patch, int* ref, scalar_t* nl, int* nl_i,
    bool* valid_ref, bool* valid_nl, bool valid,
    int* center_offsets, int patch_offset,
    int iftr, int ftr_start, int ftr_end,
    int T, int H, int W, scalar_t pix, int qi, int ki){

    for (int pk = 0; pk < pt; pk++){

      // -- ref patch --
      ref[0] = bounds(ref_patch[0]+pk,T);
      valid_ref[0] = check_interval(ref[0],0,T);

      // -- nl patch --
      nl[0] = bounds(nl_patch[0]+pk,T);
      valid_nl[0] = check_interval(nl[0],0,T);

      for (int pi = 0; pi < ps; pi++){

        // -- ref patch --
        ref[1] = (ref_patch[1]-center_offsets[1])+dilation*(pi + patch_offset);
        // ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,H);

        // -- nl patch --
        nl[1] = (nl_patch[1]-center_offsets[1])+dilation*(pi + patch_offset);
        nl[1] = reflect_bounds ? bounds(nl[1],H) : nl[1];
        valid_nl[1] = check_interval(nl[1],0,H);

        for (int pj = 0; pj < ps; pj++){
          
          // -- ref patch --
          ref[2] = (ref_patch[2]-center_offsets[3])+dilation*(pj + patch_offset);
          // ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
          valid_ref[2] = check_interval(ref[2],0,W);

          // -- nl patch --
          nl[2] = (nl_patch[2]-center_offsets[3])+dilation*(pj + patch_offset);
          nl[2] = reflect_bounds ? bounds(nl[2],W) : nl[2];
          valid_nl[2] = check_interval(nl[2],0,W);

          // -- ensure valid location --
          valid_ref[3] = true;
          valid_nl[3] = true;
          #pragma unroll
          for (int bool_idx=0; bool_idx<3; bool_idx++){
            valid_ref[3] = valid_ref[3] && valid_ref[bool_idx];
            valid_nl[3] = valid_nl[3] && valid_nl[bool_idx];
          }
          valid = valid_ref[3] && valid_nl[3];
          if (not valid) { continue; }
          
          // -- add count --
          if ((ki == 0) && (ftr_start == 0) && (valid_ref[3]) && (ref[0] == 0)){
            atomicAdd(&(counts[ref[1]][ref[2]]),1);
          }

          // -- fill each channel --
          nl_i[0] = __float2int_rn(round(nl[0]));
          for (iftr = ftr_start; iftr < ftr_end; iftr++){

            scalar_t pix = 0;
            scalar_t w = 0;
            #pragma unroll
            for (int ix=0;ix<2;ix++){
              #pragma unroll
              for (int jx=0;jx<2;jx++){

                // -- interpolation weight --
                nl_i[1] = __float2int_rd(nl[1]+ix);
                nl_i[2] = __float2int_rd(nl[2]+jx);
                w = max(0.,1-fabs(nl_i[1]-nl[1])) * max(0.,1-fabs(nl_i[2]-nl[2]));

                // -- legal --
                nl_i[1] = bounds(nl_i[1],H);
                nl_i[2] = bounds(nl_i[2],W);

                pix += w*vid[nl_i[0]][iftr][nl_i[1]][nl_i[2]];
              }
            }

            // scalar_t pix = vid[nl[0]][iftr][nl[1]][nl[2]];
            atomicAdd(&(stack[ref[0]][iftr][ref[1]][ref[2]]),weight*pix);
          }

        }
      }
    }

}

template<typename scalar_t>
__device__ __forceinline__ 
void fill_non_local_patch_bwd_bilin2d(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid,
    torch::TensorAccessor<scalar_t,2,torch::RestrictPtrTraits,int32_t> grad_weights,
    torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> grad_inds,
    // const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> counts,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_stack,
    // const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> stack,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid,
    scalar_t weight, int ps, int pt, int dilation, bool reflect_bounds,
    int* ref_patch, scalar_t* nl_patch, int* ref, scalar_t* nl, int* nl_i,
    bool* valid_ref, bool* valid_nl, bool valid,
    int* center_offsets, int patch_offset,
    int iftr, int ftr_start, int ftr_end,
    int T, int H, int W, scalar_t pix, int qi, int ki){

    for (int pk = 0; pk < pt; pk++){

      // -- ref patch --
      ref[0] = bounds(ref_patch[0]+pk,T);
      valid_ref[0] = check_interval(ref[0],0,T);

      // -- nl patch --
      nl[0] = bounds(nl_patch[0]+pk,T);
      valid_nl[0] = check_interval(nl[0],0,T);

      for (int pi = 0; pi < ps; pi++){

        // -- ref patch --
        ref[1] = (ref_patch[1]-center_offsets[0])+dilation*(pi + patch_offset);
        ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,H);

        // -- nl patch --
        nl[1] = (nl_patch[1]-center_offsets[1])+dilation*(pi + patch_offset);
        nl[1] = reflect_bounds ? bounds(nl[1],H) : nl[1];
        valid_nl[1] = check_interval(nl[1],0,H);

        for (int pj = 0; pj < ps; pj++){
          
          // -- ref patch --
          ref[2] = (ref_patch[2]-center_offsets[2])+dilation*(pj + patch_offset);
          ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
          valid_ref[2] = check_interval(ref[2],0,W);

          // -- nl patch --
          nl[2] = (nl_patch[2]-center_offsets[3])+dilation*(pj + patch_offset);
          nl[2] = reflect_bounds ? bounds(nl[2],W) : nl[2];
          valid_nl[2] = check_interval(nl[2],0,W);

          // -- ensure valid location --
          valid_ref[3] = true;
          valid_nl[3] = true;
          #pragma unroll
          for (int bool_idx=0; bool_idx<3; bool_idx++){
            valid_ref[3] = valid_ref[3] && valid_ref[bool_idx];
            valid_nl[3] = valid_nl[3] && valid_nl[bool_idx];
          }
          valid = valid_ref[3] && valid_nl[3];
          if (not valid) { continue; }
          
          // -- read count at pixel --
          // int count = counts[nl[1]][nl[2]];
          // int count = counts[ref[1]][ref[2]];

          // -- fill each channel --
          nl_i[0] = __float2int_rn(round(nl[0]));
          for (iftr = ftr_start; iftr < ftr_end; iftr++){

            // -- grad to BP --
            scalar_t grad_stack_pix = grad_stack[ref[0]][iftr][ref[1]][ref[2]];

            // -- handle continuous spatial indices --
            scalar_t igrad1 = 0;
            scalar_t igrad2 = 0;
            scalar_t pix = 0;
            scalar_t v = 0;
            scalar_t w = 0;
            scalar_t g1,g2;
            #pragma unroll
            for (int ix=0;ix<2;ix++){
              #pragma unroll
              for (int jx=0;jx<2;jx++){

                
                // -- interpolate weight --
                nl_i[1] = __float2int_rd(nl[1]+ix);
                nl_i[2] = __float2int_rd(nl[2]+jx);
                g1 = max(0.,1-fabs(nl[1]-nl_i[1]));
                g2 = max(0.,1-fabs(nl[2]-nl_i[2]));
                w = g1 * g2;

                // -- legalize inds --
                nl_i[1] = bounds(nl_i[1],H);
                nl_i[2] = bounds(nl_i[2],W);
                
                // -- compute update --
                v = vid[nl_i[0]][iftr][nl_i[1]][nl_i[2]];
                pix += w*v;

                // -- index grads --
                igrad1 += (nl[1] - nl_i[1]) < 0 ? g2*v : -g2*v;
                igrad2 += (nl[2] - nl_i[2]) < 0 ? g1*v : -g1*v;

                // -- update video --
                atomicAdd(&(grad_vid[nl_i[0]][iftr][nl_i[1]][nl_i[2]]),
                          w*grad_stack_pix*weight);
              }
            }

            // -- update dists --
            atomicAdd(&(grad_weights[qi][ki]),grad_stack_pix*pix);

            // -- update inds --
            atomicAdd(&(grad_inds[qi][ki][1]),grad_stack_pix*weight*igrad1);
            atomicAdd(&(grad_inds[qi][ki][2]),grad_stack_pix*weight*igrad2);

          }
        }
      }
    }

}

