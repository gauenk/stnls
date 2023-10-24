#include <cuda/std/type_traits>
#include <cstdio>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <math.h>
#include <ATen/ATen.h>
#include "../shared_kernel.cu"

using namespace at;


template<typename scalar_t>
__device__ __forceinline__ 
void fill_non_local_patch_int(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> stack,
    torch::TensorAccessor<int,2,torch::RestrictPtrTraits,int32_t> counts,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid,
    scalar_t weight, int ps, int pt, int dilation, bool reflect_bounds,
    int* ref_patch, int* nl_patch, int* ref, int* nl, 
    bool* valid_ref, bool* valid_nl, bool valid,
    int patch_offset, int iftr, int ftr_start, int ftr_end,
    int T, int H, int W, int qi, int ki){

  scalar_t pix;
    for (int pk = 0; pk < pt; pk++){

      // -- ref patch --
      ref[0] = bounds(ref_patch[0]+pk,T);
      valid_ref[0] = check_interval(ref[0],0,T);

      // -- nl patch --
      nl[0] = bounds(nl_patch[0]+pk,T);
      valid_nl[0] = check_interval(nl[0],0,T);

      for (int pi = 0; pi < ps; pi++){

        // -- ref patch --
        ref[1] = ref_patch[1]+dilation*(pi + patch_offset);
        // ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,H);

        // -- nl patch --
        nl[1] = nl_patch[1]+dilation*(pi + patch_offset);
        nl[1] = reflect_bounds ? bounds(nl[1],H) : nl[1];
        valid_nl[1] = check_interval(nl[1],0,H);

        for (int pj = 0; pj < ps; pj++){
          
          // -- ref patch --
          ref[2] = ref_patch[2]+dilation*(pj + patch_offset);
          // ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
          valid_ref[2] = check_interval(ref[2],0,W);

          // -- nl patch --
          nl[2] = nl_patch[2]+dilation*(pj + patch_offset);
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

          // -- add count --
          if ((ki == 0) && (ftr_start == 0) && (valid_ref[3]) && (ref[0] == 0)){
            atomicAdd(&(counts[ref[1]][ref[2]]),1);
          }
          
          // -- skip invalid pair --
          valid = valid_ref[3] && valid_nl[3];
          if (not valid) { continue; }

          // -- fill each channel --
          for (iftr = ftr_start; iftr < ftr_end; iftr++){
            pix = weight*vid[nl[0]][iftr][nl[1]][nl[2]];
            atomicAdd(&(stack[ref[0]][iftr][ref[1]][ref[2]]),pix);
          }

        }
      }
    }

}



template<typename scalar_t>
__device__ __forceinline__ 
void fill_non_local_patch_bwd_int(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid,
    torch::TensorAccessor<scalar_t,2,torch::RestrictPtrTraits,int32_t> grad_weights,
    torch::TensorAccessor<int,2,torch::RestrictPtrTraits,int32_t> counts,
    // torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> counts,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_stack,
    // const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> stack,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid,
    scalar_t weight, int ps, int pt, int dilation, bool reflect_bounds,
    int* ref_patch, int* nl_patch, int* ref, int* nl, 
    bool* valid_ref, bool* valid_nl, bool valid,
    int patch_offset, int iftr, int ftr_start, int ftr_end,
    int T, int H, int W, int qi, int ki){

  scalar_t pix;
    for (int pk = 0; pk < pt; pk++){

      // -- ref patch --
      ref[0] = bounds(ref_patch[0]+pk,T);
      valid_ref[0] = check_interval(ref[0],0,T);

      // -- nl patch --
      nl[0] = bounds(nl_patch[0]+pk,T);
      valid_nl[0] = check_interval(nl[0],0,T);

      for (int pi = 0; pi < ps; pi++){

        // -- ref patch --
        ref[1] = ref_patch[1]+dilation*(pi + patch_offset);
        // ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,H);

        // -- nl patch --
        nl[1] = nl_patch[1]+dilation*(pi + patch_offset);
        nl[1] = reflect_bounds ? bounds(nl[1],H) : nl[1];
        valid_nl[1] = check_interval(nl[1],0,H);

        for (int pj = 0; pj < ps; pj++){
          
          // -- ref patch --
          ref[2] = ref_patch[2]+dilation*(pj + patch_offset);
          // ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
          valid_ref[2] = check_interval(ref[2],0,W);

          // -- nl patch --
          nl[2] = nl_patch[2]+dilation*(pj + patch_offset);
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
          
          // -- fill each channel --
          scalar_t acc = 0;
          for (iftr = ftr_start; iftr < ftr_end; iftr++){
            scalar_t grad_stack_pix = grad_stack[ref[0]][iftr][ref[1]][ref[2]];//;/count;
            scalar_t pix = vid[nl[0]][iftr][nl[1]][nl[2]];///count;
            // scalar_t pix = stack[ref[0]][iftr][ref[1]][ref[2]];
            atomicAdd(&(grad_vid[nl[0]][iftr][nl[1]][nl[2]]),grad_stack_pix*weight);
            // atomicAdd(&(grad_weights[qi][ki]),1);
            acc += grad_stack_pix*pix;
          }
          atomicAdd(&(grad_weights[qi][ki]),acc);

        }
      }
    }

}


