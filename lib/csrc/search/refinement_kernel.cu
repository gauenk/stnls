
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include "shared_kernel.cu"
using namespace at;

/****************************

       Forward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void refinement_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> qinds,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> inds,
    int wr_h, int wr_w, int ws_h2, int ws_w2,
    int ps, int pt, int k, int stride0, int stride1, int dilation,
    int q_shift, int nH0, int nW0, int nHW0, bool reflect_bounds, bool full_ws,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1,
    int q_per_thread, int k_per_thread, int wr_h_per_thread, int wr_w_per_thread){

  // -- unpack shapes --
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int Q = dists.size(2);
  int K = qinds.size(3);

  // -- invalid constant --
  float invalid = __int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }

  // -- search region offsets --
  int psHalf = (ps)/2;
  int wrHalf_h = (wr_h)/2;
  int wrHalf_w = (wr_w)/2;
  int wrOff_h = wrHalf_h;
  int wrOff_w = wrHalf_w;
  int wrMax_h = stride1*(wr_h-1-wrOff_h);
  int wrMax_w = stride1*(wr_w-1-wrOff_w);
  int wrMin_h = -stride1 * wrOff_h;
  int wrMin_w = -stride1 * wrOff_h;
  int adj = use_adj ? psHalf : 0;

  // -- cuda index --
  int ibatch = blockIdx.x;
  int ihead = blockIdx.y;
  int q_start = blockIdx.z*q_per_thread;
  int qi,si,wh,ww;
  int qindex,i_mod;

  // decls
  int ti,hi,wi;
  int n_ti,n_hi,n_wi;
  int vH,vW,vT,nH,nW,nT;
  bool valid_anchor,valid_n,valid;
  bool vvalid_t,vvalid_h,vvalid_w,vvalid;
  bool nvalid_t,nvalid_h,nvalid_w,nvalid;
  scalar_t dist,v_pix,n_pix;

  for (int q_index = 0; q_index < q_per_thread; q_index++){

    //---------------------------
    //       Anchor Pixel
    //---------------------------

    // -- block start --
    qi = q_start + q_index;
    if (qi >= Q){ continue; }
    qindex = qi + q_shift;

    // -- pixel location from query index --
    get_pixel_loc(ti,hi,wi,qindex,i_mod,stride0,nW0,nHW0,H,W);

    // -- check bounds of pixel location --
    check_bounds(valid_anchor,ti,hi,wi,T,H,W);

    // -- search region offsets --
    set_search_offsets(wrOff_h,wrOff_w, hi, wi, stride1, wrHalf_h,
                       wrHalf_w, wrMax_h, wrMax_w, H, W, full_ws);
    set_search_minmax(wrMax_h, wrMin_h, wrOff_h, wr_h, stride1, full_ws);
    set_search_minmax(wrMax_w, wrMin_w, wrOff_w, wr_w, stride1, full_ws);

    // ---------------------------------------
    //     for each neighbor in k_search
    // ---------------------------------------
    for(int _si = 0; _si < k_per_thread; _si++){
      si = threadIdx.x + blockDim.x*_si;
      if (si >= K){ continue; }

      // -- unpack base -- 
      int n_ti = qinds[ibatch][ihead][qi][si][0]; // no search
      int hj_center = qinds[ibatch][ihead][qi][si][1];
      int wj_center = qinds[ibatch][ihead][qi][si][2];

      // ---------------------------------------
      //     for each position to search
      // ---------------------------------------
      for(int _wh = 0; _wh < wr_h_per_thread; _wh++){
        wh = threadIdx.y + blockDim.y*_wh;
        if (wh >= wr_h){ continue; }

        for(int _ww = 0; _ww < wr_w_per_thread; _ww++){
          ww = threadIdx.z + blockDim.z*_ww;
          if (ww >= wr_w){ continue; }

          // --------------------
          //      init dists
          // --------------------
          dist = 0;

          // ----------------------
          //    spatial center
          // ----------------------
          n_hi = (hj_center) + stride1 * (wh - wrOff_h);
          n_wi = (wj_center) + stride1 * (ww - wrOff_w);

          // -- check bounds of pixel location --
          check_bounds(valid_n,n_ti,n_hi,n_wi,T,H,W);
          // valid_n = valid_n && check_interval(n_hi-hi,wrMin_h,wrMax_h);
          // valid_n = valid_n && check_interval(n_wi-wi,wrMin_w,wrMax_w);
          // check_search_bounds(abs(n_hi-hi),wrMin_h,wrMax_h);
          // check_search_bounds(abs(n_hi-hi),wrMin_h,wrMax_h);
          // set_search_minmax(
          // valid_n = (abs(n_hi - hi)/stride1 <= ws_h2) && valid_n;
          // valid_n = (abs(n_wi - wi)/stride1 <= ws_w2) && valid_n;
          valid = valid_n && valid_anchor;

          //  -- compute patch difference --
          if (valid){
            compute_dist<scalar_t,DIST_TYPE>(dist,
                         vid0[ibatch][ihead],vid1[ibatch][ihead],
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
          dists[ibatch][ihead][qi][si][wh][ww] = dist;
          inds[ibatch][ihead][qi][si][wh][ww][0] = n_ti;
          inds[ibatch][ihead][qi][si][wh][ww][1] = n_hi;
          inds[ibatch][ihead][qi][si][wh][ww][2] = n_wi;

        } //  ww
      } // wh
    } // si
  } // qi
} // fxn

void refinement_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws_h, int ws_w, int ps, int k, int dist_type, int stride0, int stride1,
    int dilation, int pt, int q_shift, bool reflect_bounds, bool full_ws,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1){

   // -- num threads --
   int nheads = dists.size(1);
   int nqueries = dists.size(2);
   int ksearch = inds.size(3);
   int wr_h = dists.size(4);
   int wr_w = dists.size(5);
   int ksearch_threads = std::min(ksearch,12);
   int wr_h_threads = std::min(wr_h,8);
   int wr_w_threads = std::min(wr_w,8);
   int k_per_thread = ((ksearch-1)/ksearch_threads)+1;
   int wr_h_per_thread = ((wr_h-1)/wr_h_threads) + 1;
   int wr_w_per_thread = ((wr_w-1)/wr_w_threads) + 1;
   dim3 nthreads(ksearch_threads,wr_h_threads,wr_w_threads);

   int batchsize = vid0.size(0);
   int rem_blocks = (65535-1)/nheads+1;
   int q_per_thread = 2;
   int nquery_blocks = ((nqueries - 1) / q_per_thread) + 1;
   nquery_blocks = min(nquery_blocks,rem_blocks);
   q_per_thread = ((nqueries - 1) / nquery_blocks) + 1;
   dim3 nblocks(batchsize,nheads,nquery_blocks);

   // -- derived quantities --
   int H = vid0.size(4);
   int W = vid0.size(5);
   int nH0 = (H-1)/stride0+1;
   int nW0 = (W-1)/stride0+1;
   int nHW0 = nH0 * nW0;
   int ws_h2 = ws_h/2;
   int ws_w2 = ws_w/2;

   // launch kernel
   if (dist_type == 0){
     AT_DISPATCH_FLOATING_TYPES(vid0.type(),"refinement_forward_kernel", ([&] {
          refinement_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          qinds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
          wr_h, wr_w, ws_h2, ws_w2, ps, pt, k, stride0, stride1, dilation,
          q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws,
          use_adj, off_H0, off_W0, off_H1, off_W1,
          q_per_thread, k_per_thread, wr_h_per_thread, wr_w_per_thread);
        }));
   }else if (dist_type == 1){
     AT_DISPATCH_FLOATING_TYPES(vid0.type(),"refinement_forward_kernel", ([&] {
          refinement_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          qinds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
          wr_h, wr_w, ws_h2, ws_w2, ps, pt, k, stride0, stride1, dilation,
          q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws,
          use_adj, off_H0, off_W0, off_H1, off_W1,
          q_per_thread, k_per_thread, wr_h_per_thread, wr_w_per_thread);
        }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}


