#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
using namespace at;

#define LAUNCH_KERNEL(kernel, dist_type, full_ws, ...)\
  

// #define WS_LOOP(ws_h_per_thread, ws_w_per_thread, ws_h, ws_w,\
//                 thread_H,blockdim_H, thread_W, blockdim_W)\
//   for(int _wh = 0; _wh < ws_h_per_thread; _wh++)\
//       wh = thread_H + blockdim_H*_wh;\
//       if (wh >= ws_h){ continue; }\
//       for(int _ww = 0; _ww < ww_h_per_thread; _ww++){\
//          wh = thread_W + blockdim_W*_ww;\
//          if (ww >= ws_w){ continue; }\


__device__ __forceinline__ int bounds(int val, int lim ){
  int vval = val;
  if (val < 0){
    vval = -val;
  }else if (val >= lim){
    vval = 2*(lim-1) - val;
  }
  return vval;
}

__device__ __forceinline__ 
void get_pixel_loc(int& ti, int& hi, int& wi,  int qindex,
                  int i_mod, int stride0, int nW0, int nHW0, int H, int W){
  i_mod = qindex % nHW0;
  ti = qindex / nHW0;
  wi = ((i_mod % nW0) * stride0) % W ;
  hi = ((i_mod / nW0) * stride0) % H;
}

__device__ __forceinline__
bool check_interval(int val, int lower, int upper){
  return (val >= lower) && (val < upper);
}
__device__ __forceinline__
void check_bounds(bool& valid_anchor, int ti, int hi, int wi, int T, int H, int W){
  valid_anchor = check_interval(ti,0,T);
  valid_anchor = valid_anchor && check_interval(hi,0,H);
  valid_anchor = valid_anchor && check_interval(wi,0,W);
}


__device__ __forceinline__
void set_time_range(int& t_min, int t_shift, int ti, int T, int wt){
    t_shift = min(0,ti - wt) + max(0,ti + wt - (T-1));
    t_min = max(ti - wt - t_shift,0);
    // t_max = min(T-1,ti + wt - t_shift);
}

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
void increment_frame(int& n_ti, int& prev_ti, int& t_inc,
                     bool& swap_dir, int& dir, int ti, int t_min){
  prev_ti = n_ti;
  n_ti += t_inc;
  swap_dir = n_ti < t_min;
  t_inc = (t_inc == 0) ? -1 : t_inc; // set after tindex == 0
  t_inc = swap_dir ? 1 : t_inc;
  n_ti = swap_dir ? ti+1 : n_ti;
  prev_ti = swap_dir ? ti : prev_ti;
  dir = max(-1,min(1,n_ti - ti));
}

__device__ __forceinline__ 
void reset_centers(int& hj_center, int& wj_center, bool swap_dir, int wi, int hi){
  wj_center = swap_dir ? wi : wj_center;
  hj_center = swap_dir ? hi : hj_center;
}

template<typename scalar_t>
__device__ __forceinline__ 
void update_centers(int& hj_center, int& wj_center, int dir, int H, int W,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> fflow,
  const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> bflow){

  // -- optical flow --
  if (dir != 0){

    // -- access flows --
    if (dir > 0 ){
      wj_center = int(1.*wj_center + fflow[0][hj_center][wj_center] + 0.5);
      hj_center = int(1.*hj_center + fflow[1][hj_center][wj_center] + 0.5);
    }else{
      wj_center = int(1.*wj_center + bflow[0][hj_center][wj_center] + 0.5);
      hj_center = int(1.*hj_center + bflow[1][hj_center][wj_center] + 0.5);
    }

    // -- rounding --
    wj_center = int(max(0,min(W-1,int(wj_center))));
    hj_center = int(max(0,min(H-1,int(hj_center))));
  }
}

__device__ __forceinline__ 
void set_search_patch(int& n_hi, int& n_wi, int hj_center, int wj_center,
                      int stride1, int ws_i, int ws_j, int wsOff_h,
                      int wsOff_w, int search_abs){
  if (search_abs){
    n_hi = stride1 * ws_i;
    n_wi = stride1 * ws_j;
  }else{
    n_hi = hj_center + stride1 * (ws_i - wsOff_h);
    n_wi = wj_center + stride1 * (ws_j - wsOff_w);
  }
}

template<typename scalar_t, int DIST_TYPE>
__device__ __forceinline__ 
void compute_dist(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
    scalar_t v_pix, scalar_t n_pix, int F,
    int ti, int hi, int wi, int n_ti, int n_hi, int n_wi,
    int vH, int vW, int vT, int nH, int nW, int nT,
    bool vvalid_t,bool vvalid_h,bool vvalid_w, bool vvalid,
    bool nvalid_t, bool nvalid_h, bool nvalid_w, bool nvalid,
    int H, int W, int T, int pt, int ps, int dilation, int adj, int psHalf,
    bool reflect_bounds, int off_H0, int off_W0, int off_H1, int off_W1){
  
  for (int pk = 0; pk < pt; pk++){
    // -- anchor time --
    vT = bounds(ti + pk,T);
    vvalid_t = (vT < T) && (vT >= 0);

    // -- proposed time --
    nT = bounds(n_ti + pk,T);
    nvalid_t = (nT < T) && (nT >= 0);
    
    for (int pi = 0; pi < ps; pi++){
      // -- anchor H --
      vH = (hi-off_H0) + dilation*(pi - psHalf + adj);
      vH = reflect_bounds ? bounds(vH,H) : vH;
      vvalid_h = (vH < H) && (vH >= 0);
      
      // -- propose H --
      nH = (n_hi-off_H1) + dilation*(pi - psHalf + adj);
      nH = reflect_bounds ? bounds(nH,H) : nH;
      nvalid_h = (nH < H) && (nH >= 0);


      for (int pj = 0; pj < ps; pj++){
        
        // -- anchor W --
        vW = (wi-off_W0) + dilation*(pj - psHalf + adj);
        vW = reflect_bounds ? bounds(vW,W) : vW;
        vvalid_w = (vW < W) && (vW >= 0);

        // -- propose W --
        nW = (n_wi-off_W1) + dilation*(pj - psHalf + adj);
        nW = reflect_bounds ? bounds(nW,W) : nW;
        nvalid_w = (nW < W) && (nW >= 0);

        // -- check valid --
        vvalid = vvalid_t && vvalid_h && vvalid_w;
        nvalid = nvalid_t && nvalid_h && nvalid_w;

        // -- all channels --
        for (int ci = 0; ci < F; ci++){

          // -- get data --
          v_pix = vvalid ? vid0[vT][ci][vH][vW] : (scalar_t)0.;
          n_pix = nvalid ? vid1[nT][ci][nH][nW] : (scalar_t)0.;

          // -- compute dist --
          if(DIST_TYPE == 0){ // product
            dist += v_pix * n_pix;
          }else if(DIST_TYPE == 1){ // l2
            scalar_t _dist = (v_pix - n_pix);
            dist += _dist * _dist;
          }else{ // error
            dist = -10000000;
          }
        }
      }
    }
  }

}
