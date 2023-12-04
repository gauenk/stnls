
#include "../shared_kernel.cu"

template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void compute_dist_int(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
  int* ref_patch, int* prop_patch, int* ref, int* prop,
  bool* valid_ref, bool* valid_prop, int* q_offs,
  int ps, int pt, int dilation, bool reflect_bounds,
  int patch_offset, scalar_t invalid,
  int T, int C, int qH, int qW, int kH, int kW){
                  
  scalar_t pix0,pix1,_dist;
  for (int pk = 0; pk < pt; pk++){

    // -- reference time --
    ref[0] = bounds(ref_patch[0] + pk,T);
    valid_ref[0] = check_interval(ref[0],0,T);

    // -- proposed time --
    prop[0] = bounds(prop_patch[0] + pk,T);
    valid_prop[0] = check_interval(prop[0],0,T);
    
    for (int pi = 0; pi < ps; pi++){

      // -- ref height --
      ref[1] = ref_patch[1]+q_offs[0]+dilation*(pi + patch_offset);
      ref[1] = reflect_bounds ? bounds(ref[1],qH) : ref[1];
      valid_ref[1] = check_interval(ref[1],0,qH);

      // -- proposed height --
      prop[1] = prop_patch[1]+dilation*(pi + patch_offset);
      prop[1] = reflect_bounds ? bounds(prop[1],kH) : prop[1];
      valid_prop[1] = check_interval(prop[1],0,kH);

      for (int pj = 0; pj < ps; pj++){
        
        // -- ref width --
        ref[2] = ref_patch[2]+q_offs[1]+dilation*(pj + patch_offset);
        ref[2] = reflect_bounds ? bounds(ref[2],qW) : ref[2];
        valid_ref[2] = check_interval(ref[2],0,qW);

        // -- prop width --
        prop[2] = prop_patch[2]+dilation*(pj + patch_offset);
        prop[2] = reflect_bounds ? bounds(prop[2],kW) : prop[2];
        valid_prop[2] = check_interval(prop[2],0,kW);

        // -- ensure valid location --
        valid_ref[3] = true;
        valid_prop[3] = true;
        #pragma unroll
        for (int bool_idx=0; bool_idx<3; bool_idx++){
          valid_ref[3] = valid_ref[3] && valid_ref[bool_idx];
          valid_prop[3] = valid_prop[3] && valid_prop[bool_idx];
        }

        bool valid = valid_ref[3] && valid_prop[3];
        if (not valid) { continue; }

        // -- fill each channel --
        for (int ci = 0; ci < C; ci++){

          // -- get data --
          pix0 = valid_ref[3] ? vid0[ref[0]][ci][ref[1]][ref[2]] : (scalar_t)0.;
          pix1 = valid_prop[3] ? vid1[prop[0]][ci][prop[1]][prop[2]] : (scalar_t)0.;

          // -- compute dist --
          if(DIST_TYPE == 0){ // product
            dist += pix0 * pix1;
          }else if(DIST_TYPE == 1){ // l2
            _dist = (pix0 - pix1);
            dist += _dist*_dist;
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
void update_bwd_patch_int(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid0,
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid1,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
    scalar_t weight, int* ref_patch, int* prop_patch,
    int ps, int pt, int dilation, bool reflect_bounds,
    int patch_offset, int iftr, int ftr_start, int ftr_end,
    int* ref, int* prop, bool* valid_ref, bool* valid_prop,
    bool valid, int T, int qH, int qW, int kH, int kW){

    scalar_t pix0,pix1,pix;

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
        ref[1] = reflect_bounds ? bounds(ref[1],qH) : ref[1];
        valid_ref[1] = check_interval(ref[1],0,qH);

        // -- prop patch --
        prop[1] = prop_patch[1]+dilation*(pi + patch_offset);
        prop[1] = reflect_bounds ? bounds(prop[1],kH) : prop[1];
        valid_prop[1] = check_interval(prop[1],0,kH);

        for (int pj = 0; pj < ps; pj++){
          
          // -- ref patch --
          ref[2] = ref_patch[2]+dilation*(pj + patch_offset);
          ref[2] = reflect_bounds ? bounds(ref[2],qW) : ref[2];
          valid_ref[2] = check_interval(ref[2],0,qW);

          // -- prop patch --
          prop[2] = prop_patch[2]+dilation*(pj + patch_offset);
          prop[2] = reflect_bounds ? bounds(prop[2],kW) : prop[2];
          valid_prop[2] = check_interval(prop[2],0,kW);

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
          
          // -- fill each channel --
          for (iftr = ftr_start; iftr < ftr_end; iftr++){
            if (DIST_TYPE == 0){ // prod
              pix0 = weight*vid0[ref[0]][iftr][ref[1]][ref[2]];
              pix1 = weight*vid1[prop[0]][iftr][prop[1]][prop[2]];
              atomicAdd(&(grad_vid0[ref[0]][iftr][ref[1]][ref[2]]),pix1);
              atomicAdd(&(grad_vid1[prop[0]][iftr][prop[1]][prop[2]]),pix0);
            }else if(DIST_TYPE == 1){ // l2 norm
              pix0 = vid0[ref[0]][iftr][ref[1]][ref[2]];
              pix1 = vid1[prop[0]][iftr][prop[1]][prop[2]];
              pix = 2 * weight * (pix0 - pix1);
              atomicAdd(&grad_vid0[ref[0]][iftr][ref[1]][ref[2]],pix);
              atomicAdd(&grad_vid1[prop[0]][iftr][prop[1]][prop[2]],-pix);
            }
          }
        }
      }
    }

}



