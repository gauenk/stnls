
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
using namespace at;
// enum DIST_TYPE{ L2, PROD };


/****************************

       Inline Functions

****************************/

inline __host__ __device__ int bounds(int val, int lim ){
  int vval = val;
  if (val < 0){
    vval = -val;
  }else if (val >= lim){
    vval = 2*(lim-1) - val;
  }
  return vval;
}

inline __host__ __device__
void get_pixel_loc(int& ti, int& hi, int& wi,  int qindex,
                  int i_mod, int stride0, int nW0, int nHW0, int H, int W){
  i_mod = qindex % nHW0;
  ti = qindex / nHW0;
  wi = ((i_mod % nW0) * stride0) % W ;
  hi = ((i_mod / nW0) * stride0) % H;
}

inline __host__ __device__
void check_bounds(bool& valid_anchor, int ti, int hi, int wi, int T, int H, int W){
  valid_anchor = (ti < T) && (ti >= 0);
  valid_anchor = valid_anchor && (hi < H) && (hi >= 0);
  valid_anchor = valid_anchor && (wi < W) && (wi >= 0);
}

inline __host__ __device__
void set_time_range(int& t_min, int t_shift, int ti, int T, int wt){
    t_shift = min(0,ti - wt) + max(0,ti + wt - (T-1));
    t_min = max(ti - wt - t_shift,0);
    // t_max = min(T-1,ti + wt - t_shift);
}

inline __host__ __device__
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


inline __host__ __device__
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

inline __host__ __device__
void reset_centers(int& hj_center, int& wj_center, bool swap_dir, int wi, int hi){
  wj_center = swap_dir ? wi : wj_center;
  hj_center = swap_dir ? hi : hj_center;
}

template<typename scalar_t>
inline __host__ __device__
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

inline __host__ __device__
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
inline __host__ __device__
void compute_dist(scalar_t& dist,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
  const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
    // const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid0,
    // const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid1,
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


/****************************

       Forward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void search_with_heads_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> inds,
    int qshift, int stride0, int nH0, int nW0,
    int off_H0, int off_W0, int off_H1, int off_W1,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int dilation, int stride1,
    bool use_adj, bool reflect_bounds, bool search_abs,
    bool full_ws, int ws_h_iters, int ws_w_iters, int q_per_thread){

  // shapes
  int B = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int nHW0 = nH0 * nW0;
  int nqueries = dists.size(2);
  int st = dists.size(3);

  // constants
  // float nan = __int_as_float(0xffe00000);
  float invalid = __int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }

  // offsets
  int psHalf = (ps)/2;
  int wsHalf_h = (ws_h)/2;
  int wsHalf_w = (ws_w)/2;
  int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  int wsMax_w = stride1*(ws_w-1-wsHalf_w);
  int adj = use_adj ? psHalf : 0;

  // -- time indices --
  int t_shift;
  int t_min;

  // cuda index
  int bi = blockIdx.x;
  int head = blockIdx.y;
  int q_start = blockIdx.z*q_per_thread;
  int blkDimX = blockDim.x; // num threads in x-block
  int blkDimY = blockDim.y; // num threads in y-block
  int cu_tidX = threadIdx.x;
  int cu_tidY = threadIdx.y;
  int qi,ws_i,ws_j;

  // accumulate time offsets
  int t_inc = 0;
  int dir = 0;
  bool swap_dir = false;
  int hj_center,wj_center;
  int prev_ti;

  // decls
  int ti,hi,wi;
  int n_ti,n_hi,n_wi;
  int vH,vW,vT,nH,nW,nT;
  bool valid,vvalid,nvalid;
  bool valid_anchor,valid_n;
  bool vvalid_t,vvalid_h,vvalid_w;
  bool nvalid_t,nvalid_h,nvalid_w;
  bool eq_dim;//eq_ti,eq_hi,eq_wi,
  int wsOff_h,wsOff_w;
  scalar_t dist,v_pix,n_pix;
  int qindex,i_mod;

  for (int q_index = 0; q_index < q_per_thread; q_index++){

    //---------------------------
    //       Anchor Pixel
    //---------------------------

    // -- block start --
    qi = q_start + q_index;
    if (qi >= nqueries){ continue; }
    qindex = qi + qshift;

    // -- pixel location from query index --
    get_pixel_loc(ti,hi,wi,qindex,i_mod,stride0,nW0,nHW0,H,W);

    // -- check bounds of pixel location --
    check_bounds(valid_anchor,ti,hi,wi,T,H,W);

    // -- temporal search bounds --
    set_time_range(t_min,t_shift,ti,T,wt);

    // -- search region offsets --
    set_search_offsets(wsOff_h,wsOff_w, hi, wi, stride1, wsHalf_h,
                       wsHalf_w, wsMax_h, wsMax_w, H, W, full_ws);

    // -- init search params --
    t_inc = 0;
    n_ti = ti;
    prev_ti = ti;
    hj_center = hi;
    wj_center = wi;
    swap_dir = false;
    dir = 0;

    for(int st_i = 0; st_i < st; st_i++){

      // ---------------------------------------
      //       compute search center
      // ---------------------------------------

      // -- increment frame index --
      increment_frame(n_ti,prev_ti,t_inc,swap_dir,dir,ti,t_min);
      // -- possibly reset (hj_center,wj_center) --
      reset_centers(hj_center,wj_center,swap_dir,wi,hi);
      // -- compute offset with optical flow --
      update_centers<scalar_t>(hj_center,wj_center,dir,H,W,
                               fflow[bi][prev_ti],bflow[bi][prev_ti]);
      
      // ---------------------------------------
      //          spatial searching
      // ---------------------------------------
  
      // -- loop over search space --
      for (int _xi = 0; _xi < ws_h_iters; _xi++){
        ws_i = cu_tidX + blkDimX*_xi;
        if (ws_i >= ws_h){ continue; }
        for (int _yi = 0; _yi < ws_w_iters; _yi++){
          ws_j = cu_tidY + blkDimY*_yi;
          if (ws_j >= ws_w){ continue; }
  
          // -- compute proposed location --
          set_search_patch(n_hi,n_wi,hj_center,wj_center,stride1,
                           ws_i,ws_j,wsOff_h,wsOff_w,search_abs);

          // -- check bounds of pixel location --
          check_bounds(valid_n,n_ti,n_hi,n_wi,T,H,W);
          valid = valid_n && valid_anchor;

          // -- init dist --
          dist = 0;

          //  -- compute patch difference --
          if (valid){
            compute_dist<scalar_t,DIST_TYPE>(dist,vid0[bi][head],vid1[bi][head],
                         v_pix, n_pix, F,
                         ti,hi,wi,n_ti,n_hi,n_wi,
                         vH, vW, vT, nH, nW, nT,
                         vvalid_t,vvalid_h,vvalid_w, vvalid,
                         nvalid_t,nvalid_h,nvalid_w, nvalid,
                         H,W,T,pt,ps,dilation,adj,psHalf,
                         reflect_bounds,off_H0,off_W0,off_H1,off_W1);
          }

          // -- assignent --
          if (!valid){ dist = invalid; }
          dists[bi][head][qi][st_i][ws_i][ws_j] = dist;
          inds[bi][head][qi][st_i][ws_i][ws_j][0] = n_ti;
          inds[bi][head][qi][st_i][ws_i][ws_j][1] = n_hi;
          inds[bi][head][qi][st_i][ws_i][ws_j][2] = n_wi;
          
        }
      }
    }
  }
}

void search_with_heads_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int qstart, int stride0, int nH0, int nW0,
    int off_H0, int off_W0, int off_H1, int off_W1,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int dilation, int stride1, bool use_adj,
    bool reflect_bounds, bool search_abs, bool full_ws, int dist_type){

    // # -- launch params --
    // w_threads = min(ws,32)
    // nthreads = (w_threads,w_threads)
    // ws_iters = (ws-1)//w_threads + 1
    // nblocks = (nq-1)//batches_per_block+1
    // fprintf(stdout,"nH0,nW0: %d,%d\n",nH0,nW0);

   // fprintf(stdout,"qstart, nqueries: %d,%d\n",qstart,nqueries);
   // launch params
   // our many (too many?) registers limit the number of threads
   // -- threads --
   int nheads = dists.size(1);
   int nqueries = dists.size(2);
   int ws_h_threads = std::min(ws_h,32);
   int ws_w_threads = std::min(ws_w,32);
   int ws_h_iters = ((ws_h-1)/ws_h_threads) + 1;
   int ws_w_iters = ((ws_w-1)/ws_w_threads) + 1;
   dim3 nthreads(ws_h_threads,ws_w_threads);


   // -- nblocks --
   int bsize = vid0.size(0);
   int q_per_thread = 2;
   int nquery_blocks = ((nqueries - 1) / q_per_thread) + 1;
   dim3 nblocks(bsize,nheads,nquery_blocks);

   // fprintf(stdout,"ps,pt,nH0,nW0,wt,chnls,stride0,ws_h,ws_w: %d,%d,%d,%d,%d,%d,%d,%d,%d\n",ps,pt,nH0,nW0,wt,chnls,stride0,ws_h,ws_w);
   // fprintf(stdout,"bsize,nheads,nquery_blocks: %d,%d,%d\n",
   //         bsize,nheads,nquery_blocks);
   // fprintf(stdout,"q_per_thread,nquery_blocks,ws_h_threads,ws_w_threads: %d,%d,%d,%d\n",
   //         q_per_thread,nquery_blocks,ws_h_threads,ws_w_threads);
   // fprintf(stdout,"reflect_bounds,search_abs,full_ws,anchor_self,use_self: %d,%d,%d,%d,%d\n",
   //         reflect_bounds,search_abs,full_ws,anchor_self,use_self);
   // fprintf(stdout,"ws_h_iters,ws_w_iters,ws_h,ws_w: %d,%d,%d,%d,\n",ws_h_iters,ws_w_iters,ws_h,ws_w);
    
   // launch kernel
   if (dist_type == 0){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                                  "search_with_heads_forward_kernel", ([&] {
       search_with_heads_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
            qstart, stride0, nH0, nW0,
            off_H0, off_W0, off_H1, off_W1,
            ps, pt, ws_h, ws_w, wt, dilation, stride1,
            use_adj, reflect_bounds, search_abs, full_ws,
            ws_h_iters, ws_w_iters, q_per_thread);
          }));
   }else if(dist_type == 1){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                                  "search_with_heads_forward_kernel", ([&] {
       search_with_heads_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
            qstart, stride0, nH0, nW0,
            off_H0, off_W0, off_H1, off_W1,
            ps, pt, ws_h, ws_w, wt, dilation, stride1,
            use_adj, reflect_bounds, search_abs, full_ws,
            ws_h_iters, ws_w_iters, q_per_thread);
          }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void search_with_heads_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int qstart, int stride0, int nH0, int nW0,
    int off_H0, int off_W0, int off_H1, int off_W1,
    int ps, int pt, int dilation, bool use_adj, bool reflect_bounds,
    int q_per_thread, int npt, int cpt) {

  // -- shape --
  int bs = grad_dists.size(0);
  int nq = grad_dists.size(2);
  int k =  grad_dists.size(3);
  // int bs = vid0.size(0);
  int T = vid0.size(2);
  int colors = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int nHW0 = nH0 * nW0;

  // -- fwd decl registers --
  int ti,hi,wi;
  int tj,hj,wj;
  int tk,hk,wk;
  int tk_a,hk_a,wk_a;
  // bool valid_hj,valid_wj;
  // bool valid_hk,valid_wk;
  bool valid,valid_j,valid_k;
  // float 
  scalar_t weight,pix0,pix1,pix;

  // -- declare constants --
  int psHalf = ps/2;
  int adj = use_adj ? psHalf : 0;

  // -- limits --
  int i0_max = inds.size(2); // nq
  int i1_max = inds.size(3); // k

  // -- get indices --
  int bi = blockIdx.x;
  int i0_start = q_per_thread * (threadIdx.x + blockDim.x * blockIdx.y);
  int i1_start = threadIdx.y * npt;
  int c0_start = threadIdx.z * cpt;
  int head = blockIdx.z;

  // -- get block limits --
  int i0_end = min(i0_start + q_per_thread,i0_max);
  int i1_end = min(i1_start + npt,i1_max);
  int c0_end = min(c0_start + cpt,colors);

  // -- color offset --
  int c0 = 0;
  int c0_dist = c0_end - c0_start;
  int c0_offset = 0;
  // if (threadIdx.x == 0){
  //   printf("c0_dist,c0_start,c0_end: %d,%d,%d\n",c0_dist,c0_start,c0_end);
  // }

  // -- each region --
  for (int i0=i0_start; i0 < i0_end; i0++){

    int qindex = i0 + qstart;
    int i_mod = qindex % nHW0;
    tk_a = qindex / nHW0;
    wk_a = ((i_mod % nW0) * stride0) % W ;
    hk_a = ((i_mod / nW0) * stride0) % H;
    c0_offset = __float2int_rd(c0_dist * rand_nums[i0][0][0]);
    // printf("c0_offset: %d\n",c0_offset);

    // k neighbors
    for (int i1=i1_start; i1 < i1_end; i1++){
      ti = inds[bi][head][i0][i1][0];
      hi = inds[bi][head][i0][i1][1];
      wi = inds[bi][head][i0][i1][2];
      weight = grad_dists[bi][head][i0][i1];

      for (int pk = 0; pk < pt; pk++){
        for (int pi = 0; pi < ps; pi++){
          for (int pj = 0; pj < ps; pj++){
            

            // -- anchor patch --
            hk = (hk_a-off_H0) + dilation*(pi - psHalf + adj);
            hk = reflect_bounds ? bounds(hk,H) : hk;
            wk = (wk_a-off_W0) + dilation*(pj - psHalf + adj);
            wk = reflect_bounds ? bounds(wk,W) : wk;
            tk = reflect_bounds ? bounds(tk_a+pk,T) : tk_a+pk;

            // -- proposed location --
            hj = (hi-off_H1) + dilation*(pi - psHalf + adj);
            hj = reflect_bounds ? bounds(hj,H) : hj;
            wj = (wi-off_W1) + dilation*(pj - psHalf + adj);
            wj = reflect_bounds ? bounds(wj,W) : wj;
            tj = reflect_bounds ? bounds(ti+pk,T) : ti+pk;

            // -- assess if valid --
            valid_j = (hj >= 0) && (hj < H);
            valid_j = valid_j && (wj >= 0) && (wj < W);
            // valid_j = valid_hj && valid_wj;

            valid_k = (hk >= 0) && (hk < H);
            valid_k = valid_k && (wk >= 0) && (wk < W);
            // valid_k = valid_hk && valid_wk;

            // __syncthreads();
            valid = valid_j && valid_k;
            for (int _c0 = c0_start; _c0 < c0_end; _c0++){
              c0 = (_c0 + c0_offset) % c0_dist + c0_start;
              if(valid){
                pix0 = vid0[bi][head][tk][c0][hk][wk];
                pix1 = vid1[bi][head][tj][c0][hj][wj];
                if (DIST_TYPE == 0){
                  pix0 = weight*pix0;
                  pix1 = weight*pix1;
                  grad_vid1[bi][head][tj][c0][hj][wj] += pix0;
                  grad_vid0[bi][head][tk][c0][hk][wk] += pix1;
                }else if(DIST_TYPE == 1){
                  pix = 2 * weight * (pix0 - pix1);
                  grad_vid1[bi][head][tj][c0][hj][wj] -= pix;
                  grad_vid0[bi][head][tk][c0][hk][wk] += pix;
                }
              }
            }
          }
        }
      }
    }
  }
}

void search_with_heads_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int qstart, int stride0, int nH0, int nW0,
    int off_H0, int off_W0, int off_H1, int off_W1,
    int ps, int pt, int dilation,
    bool use_adj, bool reflect_bounds, bool use_rand,
    bool exact, int dist_type) {

  // -- unpack --
  int bsize = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int colors = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int nqueries = inds.size(2);
  int k = inds.size(3);
  assert(pt == 1);

  // -- compute number of neighbor threads --
  int npt = 4;
  int neigh_nthreads = (k-1) / npt + 1;
  if (neigh_nthreads > 32){
    neigh_nthreads = 32;
    npt = (k-1)/neigh_nthreads + 1;
  }
  if (exact){
    neigh_nthreads = 1;
    npt = k;
  }

  // -- compute number of color threads --
  int cpt = exact ? 1 : colors;
  int color_nthreads = (colors - 1)/cpt + 1;

  // -- compute number of blocks --
  //    [think: parallelization over "nqueries"]
  int q_per_thread = 2;
  int query_nthreads = 32;
  int total_per_block = q_per_thread * query_nthreads;
  int nquery_blocks = ((nqueries - 1) / total_per_block) + 1;
  if (exact){
    q_per_thread = nqueries;
    query_nthreads = 1;
    nquery_blocks = 1;
  }
  dim3 nblocks(bsize,nquery_blocks,HD);

  // -- launch params --
  dim3 nthreads(query_nthreads, neigh_nthreads, color_nthreads);

  // -- info --
  // fprintf(stdout,
  //         "query_nthreads, neigh_nthreads, color_nthreads: %d,%d,%d\n",
  //         query_nthreads, neigh_nthreads, color_nthreads);
  // fprintf(stdout,"nblocks: %d\n",nblocks);
  // fprintf(stdout,"q_per_thread,npt,cpt: %d,%d,%d\n",q_per_thread,npt,cpt);
  // fprintf(stdout,"off_H0,off_W0,off_H1,off_W1: %d,%d,%d,%d\n",
  //         off_H0,off_W0,off_H1,off_W1);
  // fprintf(stdout,"use_adj,use_reflect: %d,%d\n",
  //         use_adj,reflect_bounds);
  // fprintf(stdout,"ps,pt,dil: %d,%d,%d\n",ps,pt,dilation);

  // -- allocate random values --
  auto cu_index = grad_vid0.device().index();
  auto options = torch::TensorOptions().device(torch::kCUDA,
                                               cu_index).dtype(torch::kFloat32);
  torch::Tensor rand_nums;
  if (use_rand){
    rand_nums = torch::rand({nqueries,1,1},options);
  }else{
    rand_nums = torch::zeros({nqueries,1,1},options);
  }


  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                               "search_with_heads_backward_kernel", ([&] {
    search_with_heads_backward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          qstart, stride0, nH0, nW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, use_adj, reflect_bounds, q_per_thread, npt, cpt);
    }));
  }else if (dist_type == 1){ // l2
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                               "search_with_heads_backward_kernel", ([&] {
    search_with_heads_backward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          qstart, stride0, nH0, nW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, use_adj, reflect_bounds, q_per_thread, npt, cpt);
    }));
  }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");    }
}


