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
void compute_dist_bilin3d(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
  int* ref_patch, scalar_t* prop_patch, int* ref, scalar_t* prop, int* prop_i,
  bool* valid_ref, bool* valid_prop,
  int ps, int pt, int dilation, bool reflect_bounds,
  int patch_offset, int* center_offsets, scalar_t invalid,
  int T, int C, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t _dist){
                  
  scalar_t interp[3];
  for (int pk = 0; pk < pt; pk++){

    // -- reference time --
    ref[0] = bounds(ref_patch[0] + pk,T);
    valid_ref[0] = check_interval(ref[0],0,T);

    // -- proposed time [an actual decimal this time] --
    prop[0] = bounds(prop_patch[0] + pk,T);
    valid_prop[0] = check_interval(prop[0],0,T);
    
    for (int pi = 0; pi < ps; pi++){

      // -- ref height --
      ref[1] = (ref_patch[1]-center_offsets[0])+dilation*(pi + patch_offset);
      ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
      valid_ref[1] = check_interval(ref[1],0,H);

      // -- proposed height --
      prop[1] = (prop_patch[1]-center_offsets[1])+dilation*(pi + patch_offset);
      prop[1] = reflect_bounds ? bounds(prop[1],H) : prop[1];
      valid_prop[1] = check_interval(prop[1],0,H);

      for (int pj = 0; pj < ps; pj++){
        
        // -- ref width --
        ref[2] = (ref_patch[2]-center_offsets[2])+dilation*(pj + patch_offset);
        ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];
        valid_ref[2] = check_interval(ref[2],0,W);

        // -- prop width --
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


        // -- fill each channel --
        for (int ci = 0; ci < C; ci++){

          // -- reference value --
          pix0 = valid_ref[3] ? vid0[ref[0]][ci][ref[1]][ref[2]] : 0;

          // -- interpolate pixel value --
          pix1 = 0;
          scalar_t w = 0;
#pragma unroll
          for (int tx=0;tx<2;tx++){
            prop_i[0] = __float2int_rn(prop[0]+tx);
            interp[0] = max(0.,1-fabs(prop_i[0]-prop[0]));
#pragma unroll
          for (int ix=0;ix<2;ix++){
            prop_i[1] = __float2int_rd(prop[1]+ix);
            interp[1] = max(0.,1-fabs(prop_i[1]-prop[1]));
#pragma unroll
            for (int jx=0;jx<2;jx++){
              prop_i[2] = __float2int_rd(prop[2]+jx);
              interp[2] = max(0.,1-fabs(prop_i[2]-prop[2]));

              // -- compute weight --
              w = 1;
#pragma unroll
              for (int _widx=0;_widx<3;_widx++){
                w *= interp[_widx];
              }

              // -- ensure legal bounds --
              prop_i[0] = bounds(prop_i[0],T);
              prop_i[1] = bounds(prop_i[1],H);
              prop_i[2] = bounds(prop_i[2],W);

              // -- update --
              pix1 += valid_prop[3] ? w*vid1[prop_i[0]][ci][prop_i[1]][prop_i[2]] : 0;
            }
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
void update_bwd_patch_bilin3d(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid0,
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid1,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
    scalar_t weight, int* ref_patch, scalar_t* prop_patch,
    int ps, int pt, int dilation, bool reflect_bounds,
    int* center_offsets, int patch_offset,
    int iftr, int ftr_start, int ftr_end,
    int* ref, scalar_t* prop, int* prop_i,
    bool* valid_ref, bool* valid_prop, bool valid,
    int T, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t pix, int i1){
    scalar_t interp[3];

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

          // -- fill each channel --
          for (iftr = ftr_start; iftr < ftr_end; iftr++){
            
            // -- reference value --
            pix0 = vid0[ref[0]][iftr][ref[1]][ref[2]];
  
            // -- interpolate pixel value --
            pix1 = 0;
            scalar_t w = 0;
           #pragma unroll
            for (int tx=0;tx<2;tx++){
            prop_i[0] = __float2int_rn(prop[0]+tx);
            interp[0] = max(0.,1-fabs(prop_i[0]-prop[0]));
            #pragma unroll
            for (int ix=0;ix<2;ix++){
              prop_i[1] = __float2int_rd(prop[1]+ix);
              interp[1] = max(0.,1-fabs(prop_i[1]-prop[1]));
              #pragma unroll
              for (int jx=0;jx<2;jx++){
                prop_i[2] = __float2int_rd(prop[2]+jx);
                interp[2] = max(0.,1-fabs(prop_i[2]-prop[2]));

                // -- interpolation weight --
                w = 1;
#pragma unroll
                for (int _widx=0;_widx<3;_widx++){
                  w *= interp[_widx];
                }

                // -- ensure legal bounds --
                prop_i[0] = bounds(prop_i[0],T);
                prop_i[1] = bounds(prop_i[1],H);
                prop_i[2] = bounds(prop_i[2],W);

                // -- update pixel --
                pix1 += w*vid1[prop_i[0]][iftr][prop_i[1]][prop_i[2]];
              }
            }}

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
            for (int tx=0;tx<2;tx++){
            prop_i[0] = __float2int_rn(prop[0]+tx);
            interp[0] = max(0.,1-fabs(prop_i[0]-prop[0]));

            #pragma unroll
            for (int ix=0;ix<2;ix++){
              prop_i[1] = __float2int_rd(prop[1]+ix);
              interp[1] = max(0.,1-fabs(prop_i[1]-prop[1]));

              #pragma unroll
              for (int jx=0;jx<2;jx++){
                prop_i[2] = __float2int_rd(prop[2]+jx);
                interp[2] = max(0.,1-fabs(prop_i[2]-prop[2]));


                // -- interpolation weight --
                w = 1;
#pragma unroll
                for (int _widx=0;_widx<3;_widx++){
                  w *= interp[_widx];
                }

                // -- update gradient --
                if (DIST_TYPE == 0){ // prod
                  atomicAdd(&(grad_vid1[prop_i[0]][iftr][prop_i[1]][prop_i[2]]),
                            w*wpix0);
                }else if(DIST_TYPE == 1){ // l2 norm
                  atomicAdd(&grad_vid1[prop_i[0]][iftr][prop_i[1]][prop_i[2]],-w*pix);
                }
              }
            }}

          }
        }
      }
    }

}

template<typename scalar_t>
__device__ __forceinline__ 
void update_bwd_flows_bilin3d(
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
    scalar_t delta_f = prop[0] - __int2float_rn(ref[0]);
    if (fabs(delta_f) < 1e-10){ return; } // no temporal offset
    bool is_fwd = delta_f > 0;
    int direction = is_fwd ? 1 : -1; //  fwd or bwd
    int prop_time = is_fwd ? __float2int_ru(prop[0]) : __float2int_rd(prop[0]);
    int delta_t = abs(prop_time - ref[0]);
    int src_t = prop_time - direction;

    // -- select tensor --
    auto flow = is_fwd ? fflow : bflow;
    auto g_flow = is_fwd ? grad_fflow : grad_bflow;

    // -- init incs --
    scalar_t w;
    scalar_t interm[3],interm_n[3];
    scalar_t grad0 = 0;
    scalar_t grad1 = 0;
    scalar_t g0,g1,g2,v0,v1;
    int tx;

    // -- setup --
#pragma unroll
    for (int idx=0;idx<3;idx++){
      interm[idx] = __int2float_rd(ref[idx]);
      interm_n[idx] = interm[idx];
    }

    // -- compute gradient across time --
    for (int _tx = 0; _tx < delta_t; _tx++){

      scalar_t grad0_t = 0;
      scalar_t grad1_t = 0;

      // tx = direction * _tx;
      v0,v1 = 0,0;
      interm[0] = interm[0]+direction;
      // prop_i[0] = __float2int_rd(interm[0]);
      interm[1] = interm_n[1];
      interm[2] = interm_n[2];

#pragma unroll
      for (int tx=0;tx<2;tx++){
        prop_i[0] = __float2int_rn(interm[0]+tx);
        g0 = max(0.,1-fabs(prop_i[1]-interm[0]));
#pragma unroll
      for (int ix=0;ix<2;ix++){
        prop_i[1] = __float2int_rd(interm[1]+ix);
        g1 = max(0.,1-fabs(prop_i[1]-interm[1]));
#pragma unroll
        for (int jx=0;jx<2;jx++){
          prop_i[2] = __float2int_rd(interm[2]+jx);
          g2 = max(0.,1-fabs(prop_i[2]-interm[2]));

          // -- compute direction --
          bool left0 = (prop_i[1]-interm[1]) < 0;
          bool left1 = (prop_i[2]-interm[2]) < 0;

          // -- ensure legal inds --
          prop_i[0] = bounds(prop_i[0],T);
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
      }}

      // -- accumulate across time --
      grad0 += _tx > 1 ? grad0*grad0_t : grad0_t;
      grad1 += _tx > 1 ? grad1*grad1_t : grad1_t;
      // grad0 += 1;
      // grad1 += 1;

    }

    // -- udpate gradient --
#pragma unroll
    for (int tx=0;tx<2;tx++){
      prop_i[0] = __float2int_rd(prop[0]+tx);
      g0 = max(0.,1-fabs(prop_i[0]-prop[0]));
#pragma unroll
    for (int ix=0;ix<2;ix++){
      prop_i[1] = __float2int_rd(prop[1]+ix);
      g1 = max(0.,1-fabs(prop_i[1]-prop[1]));
#pragma unroll
      for (int jx=0;jx<2;jx++){
        prop_i[2] = __float2int_rd(prop[2]+jx);
        g2 = max(0.,1-fabs(prop_i[2]-prop[2]));

        // -- finish weights --
        w = g0 * g1 * g2;

        // -- ensure legal bounds --
        prop_i[0] = bounds(prop_i[0],T);
        prop_i[1] = bounds(prop_i[1],H);
        prop_i[2] = bounds(prop_i[2],W);

        // -- update --
        atomicAdd(&g_flow[prop_i[0]][0][prop_i[1]][prop_i[2]],\
                  w*iweight[1]*grad0);
        atomicAdd(&g_flow[prop_i[0]][1][prop_i[1]][prop_i[2]],\
                  w*iweight[2]*grad1);
      }
    }
    }

}


template<typename scalar_t>
__device__ __forceinline__ 
void update_bwd_offsets_bilin3d(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_offsets,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> fflow,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> bflow,
    scalar_t* iweight, int*ref_patch, scalar_t* prop, int* prop_i,
    bool* valid_ref, bool* valid_prop, bool valid,
    int T, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t pix, int i1,
    int ws, int wt, int stride1, bool full_ws, bool full_ws_time){

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    // -- determine (st,wi,wj) --
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    // -- rounding removes offset values --
    for (int _idx = 0; _idx < 3; _idx++){
      prop_i[_idx] = __float2int_rn(round(prop[_idx]));
    }

    // -- find frame anchor --
    int frame_anchor[3];
    frame_anchor[0] = ref_patch[0];
    frame_anchor[1] = ref_patch[1];
    frame_anchor[2] = ref_patch[2];

    // -- search region offsets --
    int wsOff_h,wsOff_w;
    int wsHalf = (ws)/2;
    // int wsHalf = (ws)/2;
    int wsMax = stride1*(ws-1-wsHalf);
    set_search_offsets(wsOff_h, wsOff_w, ref_patch[1], ref_patch[2], stride1,
                       wsHalf, wsHalf, wsMax, wsMax, H, W, full_ws);

    // -- temporal search bounds --
    int t_shift,t_max;
    set_time_range(t_max,t_shift,ref_patch[0],T,wt);

    // -- compute temporal offset --
    int ST = 2*wt+1;
    int prev_ti = ref_patch[0];
    int t_inc = 0;
    bool swap_dir = false;
    int dir = 0;
    int st_i;
    for(st_i = 0; st_i < ST; st_i++){
      if (frame_anchor[0] == prop_i[0]){break;}

      // -- increment frame index --
      increment_frame(frame_anchor[0],prev_ti,t_inc,swap_dir,dir,ref_patch[0],t_max);

      // -- possibly reset (frame_anchor <- reference_patch) --
      reset_centers(frame_anchor,ref_patch,swap_dir);

      // -- compute offset with optical flow --
      update_centers<scalar_t>(frame_anchor[1],frame_anchor[2],dir,H,W,
                               fflow[prev_ti],bflow[prev_ti]);
      
      // -- search region offsets --
      set_search_offsets(wsOff_h,wsOff_w, frame_anchor[1], frame_anchor[2], stride1,
                         wsHalf, wsHalf, wsMax, wsMax, H, W, full_ws_time);

    }

    // -- compute spatial offset --
    int wi = prop_i[1] - ref_patch[1];
    int wj = prop_i[2] - ref_patch[2];

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    // -- update offset gradient --
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
    grad_offsets[st_i][wi][wj][0] = iweight[0];
    grad_offsets[st_i][wi][wj][1] = iweight[1]; 
    grad_offsets[st_i][wi][wj][2] = iweight[2];

}

      
