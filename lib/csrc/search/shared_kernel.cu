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
template< class T, class U >
inline constexpr bool is_same_v = cuda::std::is_same<T, U>::value;

using namespace at;

//#define LAUNCH_KERNEL(kernel, dist_type, full_ws, ...)    \
  
template<typename dtype=int>
__device__ __forceinline__ dtype bounds(dtype val, int lim ){
  dtype vval = val;
  if (val < 0){
    vval = -val; // want ("-1" -> "1") _not_ ("-1" -> "0")
  }else if (val >= lim){
    vval = 2*(lim-1)-val; // want ("H" -> "H-2") _not_ ("H" -> "H-1")
  }
  return vval;
}


template<typename itype=int>
__device__ __forceinline__ 
void get_pixel_loc(itype* pix,  int qindex, int tmp, int stride0,
                   int nW0, int nHW0, int H, int W){
  int nH_index;
  if (is_same_v<itype,int>){
    tmp = qindex;
    pix[0] = tmp / nHW0;
    tmp = (tmp - pix[0]*nHW0); 
    nH_index = tmp / nW0;
    pix[1] = (nH_index*stride0) % H;
    tmp = tmp - nH_index*nW0;
    pix[2] = ((tmp % nW0) * stride0) % W;
  }else{
    tmp = qindex;
    pix[0] = round(tmp/nHW0);
    tmp = (tmp - pix[0]*nHW0); 
    nH_index = tmp / nW0;
    pix[1] = round((nH_index*stride0) % H);
    tmp = tmp - nH_index*nW0;
    pix[2] = round(((tmp % nW0) * stride0) % W);
  }
}


template<typename itype=int>
__device__ __forceinline__
bool check_interval(itype val, int lower, int upper){
  return (val >= lower) && (val < upper);
}

template<typename itype=int>
__device__ __forceinline__
void check_bounds(bool& valid_anchor, itype* loc3d, int T, int H, int W){
  valid_anchor = check_interval<itype>(loc3d[0],0,T);
  valid_anchor = valid_anchor && check_interval<itype>(loc3d[1],0,H);
  valid_anchor = valid_anchor && check_interval<itype>(loc3d[2],0,W);
}


// template<typename itype=int>
__device__ __forceinline__
void set_search_offsets(int& wsOff_h, int& wsOff_w, int hi, int wi, int stride1,
                        int wsHalf_h, int wsHalf_w, int wsMax_h, int wsMax_w,
                        int H, int W, bool full_ws){
    if(full_ws){
      wsOff_h = (hi-max(hi-stride1*wsHalf_h,0))/stride1;
      wsOff_w = (wi-max(wi-stride1*wsHalf_w,0))/stride1;
      if ((hi+wsMax_h) >= H){
        wsOff_h+=(hi+wsMax_h-min(hi+stride1*wsMax_h,H-1)-1)/stride1 + 1;
      }
      if ((wi+wsMax_w) >= W){
        wsOff_w+=(wi+wsMax_w-min(wi+stride1*wsMax_w,W-1)-1)/stride1 + 1;
      }
    }else{
      wsOff_h = wsHalf_h;
      wsOff_w = wsHalf_w;
    }
}

__device__ __forceinline__
void set_search_minmax(int& wsMax, int& wsMin, int wsOff,
                       int ws, int stride1, bool set_bool){
  if (set_bool){
    wsMax = stride1*(ws-1-wsOff);
    wsMin = -stride1*wsOff;
  }
}

__device__ __forceinline__
void set_time_range(int& t_max, int t_shift, int ti, int T, int wt){
    t_shift = min(0,ti - wt) + max(0,ti + wt - (T-1));
    // t_min = max(ti - wt - t_shift,0);
    t_max = min(T-1,ti + wt - t_shift);
}

template<typename itype=int>
__device__ __forceinline__
void increment_frame(itype& n_ti, int& prev_ti, int& t_inc,
                     bool& swap_dir, int& dir, int ti, int t_max){
  prev_ti = is_same_v<itype,int> ? n_ti : __float2int_rn(round(n_ti));
  n_ti += t_inc;
  swap_dir = n_ti > t_max; // max(t_max) == (T-1), a legal index.
  t_inc = (t_inc == 0) ? 1 : t_inc; // set after tindex == 0, forward first
  t_inc = swap_dir ? -1 : t_inc;
  n_ti = swap_dir ? ti-1 : n_ti;
  prev_ti = swap_dir ? ti : prev_ti;
  dir = max(-1,min(1,int(n_ti) - ti));
}

template<typename itype=int>
__device__ __forceinline__ 
void reset_centers(itype* prop_patch, int* ref_patch, bool swap_dir){
  if(is_same_v<itype,int>){
    prop_patch[1] = swap_dir ? ref_patch[1] : prop_patch[1];
    prop_patch[2] = swap_dir ? ref_patch[2] : prop_patch[2];
  }else{
    prop_patch[1] = swap_dir ? __int2float_rn(ref_patch[1]) : prop_patch[1];
    prop_patch[2] = swap_dir ? __int2float_rn(ref_patch[2]) : prop_patch[2];
  }
}

template<typename scalar_t, typename itype=int>
__device__ __forceinline__ 
void update_centers(itype& hj_center, itype& wj_center, int dir, int H, int W,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> fflow,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> bflow){

  // -- fixed so we can read both --
  itype hj_tmp = hj_center;
  itype wj_tmp = wj_center;

  // -- optical flow --
  if (dir != 0){

    // -- access flows --
    auto flow = dir > 0 ? fflow : bflow;


    if(is_same_v<itype,int>){

      // -- simple rounding if "int" --
      wj_center = int(1.*wj_center + flow[0][hj_tmp][wj_tmp] + 0.5);
      hj_center = int(1.*hj_center + flow[1][hj_tmp][wj_tmp] + 0.5);

      // -- wrap around boarders --
      wj_center = max(0,min(W-1,(int)wj_center));
      hj_center = max(0,min(H-1,(int)hj_center));

    }else{


      // // -- simple rounding if "int" --
      // wj_center = round(wj_center + flow[0][hj_tmp][wj_tmp]);
      // hj_center = round(hj_center + flow[1][hj_tmp][wj_tmp]);

      // // -- wrap around boarders --
      // wj_center = max(0.,min(1.*W-1,(float)wj_center));
      // hj_center = max(0.,min(1.*H-1,(float)hj_center));

      // -- weighted average of neighbors --
      float weight = 0;
      int hj = 0, wj = 0;
      for (int i=0;i<2;i++){
        for (int j=0;j<2;j++){

          // -- compute int locaion with weight --
          hj = __float2int_rd(hj_tmp + i);
          wj = __float2int_rd(wj_tmp + j);
          weight = max(0.,1-fabs(hj-hj_tmp)) * max(0.,1-fabs(wj-wj_tmp));

          // -- ensure legal boudns --
          hj = bounds(hj,H);
          wj = bounds(wj,W);

          // -- update with shift --
          wj_center = wj_center + weight*flow[0][hj][wj];
          hj_center = hj_center + weight*flow[1][hj][wj];
        }
      }

      // -- wrap around boarders --
      wj_center = max(0.,min(1.*W-1,(float)wj_center));
      hj_center = max(0.,min(1.*H-1,(float)hj_center));

    }

  }
}

template<typename itype=int>
__device__ __forceinline__ 
void set_search_patch(itype* prop, itype* frame_anchor,
                      int stride1, int ws_i, int ws_j, int wsOff_h,
                      int wsOff_w, int search_abs){
  prop[0] = frame_anchor[0];
  if (search_abs){
    prop[1] = stride1 * ws_i;
    prop[2] = stride1 * ws_j;
  }else{
    prop[1] = frame_anchor[1] + stride1 * (ws_i - wsOff_h);
    prop[2] = frame_anchor[2] + stride1 * (ws_j - wsOff_w);
  }
}

template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void compute_dist(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
  int* ref_patch, int* prop_patch, int* ref, int* prop,
  bool* valid_ref, bool* valid_prop,
  int ps, int pt, int dilation, bool reflect_bounds,
  int patch_offset, int* center_offsets, scalar_t invalid,
  int T, int C, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t _dist){
                  
  for (int pk = 0; pk < pt; pk++){

    // -- reference time --
    ref[0] = bounds(ref_patch[0] + pk,T);
    valid_ref[0] = check_interval(ref[0],0,T);

    // -- proposed time --
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
void compute_dist_v2(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
  int* ref_patch, int* prop_patch, int* ref, int* prop,
  bool* valid_ref, bool* valid_prop,
  int ps, int pt, int dilation, bool reflect_bounds,
  int patch_offset, int* center_offsets, scalar_t invalid,
  int iftr, int ftr_start, int ftr_end,
  int T, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t _dist){

  for (int pk = 0; pk < pt; pk++){

    // -- reference time --
    ref[0] = bounds(ref_patch[0] + pk,T);
    valid_ref[0] = check_interval(ref[0],0,T);

    // -- proposed time --
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
        for (iftr = ftr_start; iftr < ftr_end; iftr++){

          // -- get data --
          pix0 = valid_ref[3] ? vid0[ref[0]][iftr][ref[1]][ref[2]] : (scalar_t)0.;
          pix1 = valid_prop[3] ? vid1[prop[0]][iftr][prop[1]][prop[2]] : (scalar_t)0.;

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
void update_bwd_patch(
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid0,
    torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid1,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
    const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
    // torch::TensorAccessor<int,3,torch::RestrictPtrTraits,int32_t> count0,
    // torch::TensorAccessor<int,3,torch::RestrictPtrTraits,int32_t> count1,
    scalar_t weight, int* ref_patch, int* prop_patch,
    int ps, int pt, int dilation, bool reflect_bounds,
    int* center_offsets, int patch_offset,
    int iftr, int ftr_start, int ftr_end,
    int* ref, int* prop, bool* valid_ref, bool* valid_prop, bool valid,
    int T, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t pix, int i1){

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
            if (DIST_TYPE == 0){ // prod
              pix0 = weight*vid0[ref[0]][iftr][ref[1]][ref[2]];
              pix1 = weight*vid1[prop[0]][iftr][prop[1]][prop[2]];
              atomicAdd(&(grad_vid0[ref[0]][iftr][ref[1]][ref[2]]),pix1);
              atomicAdd(&(grad_vid1[prop[0]][iftr][prop[1]][prop[2]]),pix0);
            }else if(DIST_TYPE == 1){ // l2 norm
              pix0 = valid_ref[3] ? vid0[ref[0]][iftr][ref[1]][ref[2]] : (scalar_t)0.;
              pix1 = valid_prop[3] ? vid1[prop[0]][iftr][prop[1]][prop[2]] : (scalar_t)0.;
              pix = 2 * weight * (pix0 - pix1);
              atomicAdd(&grad_vid0[ref[0]][iftr][ref[1]][ref[2]],pix);
              atomicAdd(&grad_vid1[prop[0]][iftr][prop[1]][prop[2]],-pix);
            }
          }
        }
      }
    }

}





