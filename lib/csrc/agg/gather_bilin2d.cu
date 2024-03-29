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
void gather_patch_fwd_bilin2d(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> stack,
    torch::TensorAccessor<int,2,torch::RestrictPtrTraits,int32_t> counts,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid,
    scalar_t weight, int ps, int pt, int dilation, bool reflect_bounds,
    int* ref_patch, scalar_t* nl_patch, int* ref, scalar_t* nl, //int* nl_i,
    bool* valid_ref, bool* valid_nl, bool valid,
    int patch_offset, int ftr_start, int ftr_end,
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
        valid_ref[1] = check_interval(ref[1],0,H);

        // -- nl patch --
        nl[1] = nl_patch[1]+dilation*(pi + patch_offset);
        nl[1] = reflect_bounds ? bounds(nl[1],H) : nl[1];
        valid_nl[1] = check_interval(nl[1],0,H);

        for (int pj = 0; pj < ps; pj++){
          
          // -- ref patch --
          ref[2] = ref_patch[2]+dilation*(pj + patch_offset);
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
          // nl_i[0] = __float2int_rn(round(nl[0]));
          int ti = __float2int_rn(round(nl[0]));
          for (int iftr = ftr_start; iftr < ftr_end; iftr++){
            bilin2d_interpolate(pix,nl[1],nl[2],H,W,vid[ti][iftr]);
            atomicAdd(&(stack[ref[0]][iftr][ref[1]][ref[2]]),weight*pix);
          }

        }
      }
    }

}

template<typename scalar_t>
__device__ __forceinline__ 
void gather_patch_bwd_bilin2d(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid,
    torch::TensorAccessor<scalar_t,2,torch::RestrictPtrTraits,int32_t> grad_weights,
    torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> grad_inds,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_stack,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid,
    scalar_t weight, int ps, int pt, int dilation, bool reflect_bounds,
    int* ref_patch, scalar_t* nl_patch, int* ref, scalar_t* nl, int* nl_i,
    bool* valid_ref, bool* valid_nl, bool valid,
    int patch_offset, int iftr, int ftr_start, int ftr_end,
    int signH, int signW, int T, int H, int W, int qi, int ki){

  int sH,sW; // sign of height,width interpolation
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
        sH = check_interval(nl[1],0,H) ? 1 : -1;
        nl[1] = reflect_bounds ? bounds(nl[1],H) : nl[1];
        valid_nl[1] = check_interval(nl[1],0,H);

        for (int pj = 0; pj < ps; pj++){
          
          // -- ref patch --
          ref[2] = ref_patch[2]+dilation*(pj + patch_offset);
          // ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
          valid_ref[2] = check_interval(ref[2],0,W);

          // -- nl patch --
          nl[2] = nl_patch[2]+dilation*(pj + patch_offset);
          sW = check_interval(nl[2],0,W) ? 1 : -1;
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
          nl_i[0] = __float2int_rn(round(nl[0]));
          scalar_t acc_pix = 0;
          scalar_t iW = 0;
          scalar_t iH = 0;
          for (iftr = ftr_start; iftr < ftr_end; iftr++){

            // -- grad to BP --
            scalar_t grad_stack_pix = grad_stack[ref[0]][iftr][ref[1]][ref[2]];
            scalar_t weighted_grad = grad_stack_pix*weight;

            // -- handle continuous spatial indices --
            scalar_t igradH = 0;
            scalar_t igradW = 0;
            scalar_t pix = 0;
            scalar_t v = 0;
            scalar_t w = 0;
            scalar_t gH,gW;
            #pragma unroll
            for (int ix=0;ix<2;ix++){
              #pragma unroll
              for (int jx=0;jx<2;jx++){

                // -- interpolate weight --
                nl_i[1] = __float2int_rz(nl[1]+ix);
                nl_i[2] = __float2int_rz(nl[2]+jx);
                // nl_i[1] = __float2int_rd(floorf(nl[1]+ix));
                // nl_i[2] = __float2int_rd(floorf(nl[2]+jx));
                gH = max(0.,1-fabs(nl[1]-nl_i[1]));
                gW = max(0.,1-fabs(nl[2]-nl_i[2]));
                w = gH * gW;

                // -- check directions --
                bool leftH = ix==0;//(nl_i[1]-nl[1]) < 0;
                // bool rightH = (nl_i[1]-nl[1]) > 0;
                bool leftW = jx==0;//(nl_i[2]-nl[2]) < 0;
                // bool leftW = (nl_i[2]-nl[2]) < 0;
                // bool rightW = (nl_i[2]-nl[2]) > 0;

                // -- legalize inds --
                nl_i[1] = bounds(nl_i[1],H);
                nl_i[2] = bounds(nl_i[2],W);
                
                // -- read video --
                v = vid[nl_i[0]][iftr][nl_i[1]][nl_i[2]];

                // -- dist grad --
                pix += w*v;

                // -- index grad --
                // igradW += leftW ? -gH*v : (rightW ? gH*v : 0); // dF[0]/dF[0]; A(0)
                // igradH += leftH ? -gW*v : (rightH ? gW*v : 0); // dF[0]/dF[0]; A(0)
                igradW += leftW ? -gH*v : gH*v; // dF[0]/dF[0]; A(0) 
                igradH += leftH ? -gW*v : gW*v; // dF[0]/dF[0]; A(0)
                
                // -- video grad --
                atomicAdd(&(grad_vid[nl_i[0]][iftr][nl_i[1]][nl_i[2]]),w*weighted_grad);

              }
            }

            // -- update dists --
            acc_pix += grad_stack_pix*pix;
            // acc_pix += grad_stack_pix;//pix;
            // atomicAdd(&(grad_weights[qi][ki]),grad_stack_pix*pix);
            // atomicAdd(&(grad_weights[qi][ki]),static_cast<scalar_t>(1));
            iW += grad_stack_pix*weight*sW*igradW;
            iH += grad_stack_pix*weight*sH*igradH;

          }

          // -- update dists --
          atomicAdd(&(grad_weights[qi][ki]),acc_pix);
          
          // -- update inds --
          atomicAdd(&(grad_inds[qi][ki][1]),iH*signH);
          atomicAdd(&(grad_inds[qi][ki][2]),iW*signW);


        }
      }
    }

}

