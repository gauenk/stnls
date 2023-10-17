
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
// #include "shared_kernel.cu"
#include "nls_int.cu"
using namespace at;

/****************************

       Forward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void refinement_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> flows,
    torch::PackedTensorAccessor32<scalar_t,8,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,9,torch::RestrictPtrTraits> inds,
    int wr, int ws, int ps, int pt, int K, int stride0, int stride1, int dilation,
    bool reflect_bounds, bool full_ws, bool restrict_radius, int patch_offset,
    int q_per_thread, int k_per_thread, int wr_per_thread){

  // -- unpack shapes --
  int HD = vid0.size(1);
  int HD_f = flows.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int nH = dists.size(3);
  int nW = dists.size(4);
  int nHW = nH*nW;
  int Q = T*nHW;
  int Ks = inds.size(5);
  int ti,nh,nw;

  // -- invalid constant --
  scalar_t invalid = (scalar_t)__int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }

  // -- search window params --
  int wrHalf = (wr-1)/2;
  int wrMax = stride1*(wr-1-wrHalf);
  int wrOff_h,wrOff_w;

  // -- cuda index --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int ihead_f = ihead % HD_f;
  int q_start = blockIdx.x*q_per_thread;
  int qi,ki,wh,ww;

  // -- fwd decls --
  int prop_center[2];
  int prop_patch[3];
  int prop_pix[3];
  int ref_patch[3];
  int ref_pix[3];
  bool valid;
  bool valid_prop[4];
  bool valid_ref[4];
  scalar_t dist;

  for (int q_index = 0; q_index < q_per_thread; q_index++){

    //---------------------------
    //       Anchor Pixel
    //---------------------------

    // -- block start --
    qi = q_start + q_index;
    if (qi >= Q){ continue; }

    // -- pixel location from query index --
    get_pixel_loc(ref_patch,qi,stride0,nW,nHW,H,W);
    ti = ref_patch[0];
    nh = ref_patch[1]/stride0;
    nw = ref_patch[2]/stride0;

    // -- check bounds of pixel location --
    check_bounds(valid_ref[3],ref_patch,T,H,W);

    // -- search region offsets --
    set_search_offsets(wrOff_h, wrOff_w,
                       ref_patch[0], ref_patch[1], stride1,
                       wrHalf, wr, H, W, full_ws);

    // -- [unused] set search bounds for [optionally] expanded region --
    // if (restrict_radius){
    //   set_search_minmax(wrMax_h, wrMin_h, wrOff_h, wr_h, stride1, full_ws);
    //   set_search_minmax(wrMax_w, wrMin_w, wrOff_w, wr_w, stride1, full_ws);
    // }

    // ---------------------------------------
    //     for each neighbor in k_search
    // ---------------------------------------
    for(int _ki = 0; _ki < k_per_thread; _ki++){
      ki = threadIdx.x + blockDim.x*_ki;
      if (ki >= K){ continue; }

      // -- unpack base -- 
      prop_patch[0] = flows[ibatch][ihead_f][ti][nh][nw][ki][0]; // no search
      prop_center[0] = flows[ibatch][ihead_f][ti][nh][nw][ki][1];
      prop_center[1] = flows[ibatch][ihead_f][ti][nh][nw][ki][2];

      // ---------------------------------------
      //     for each position to search
      // ---------------------------------------
      for(int _wh = 0; _wh < wr_per_thread; _wh++){
        wh = threadIdx.y + blockDim.y*_wh;
        if (wh >= wr){ continue; }

        for(int _ww = 0; _ww < wr_per_thread; _ww++){
          ww = threadIdx.z + blockDim.z*_ww;
          if (ww >= wr){ continue; }

          // --------------------
          //      init dists
          // --------------------
          dist = 0;

          // ----------------------
          //    spatial center
          // ----------------------
          prop_patch[1] = prop_center[0] + stride1 * (wh - wrOff_h);
          prop_patch[2] = prop_center[1] + stride1 * (ww - wrOff_w);

          // -- check bounds of pixel location --
          check_bounds(valid_prop[3],prop_patch,T,H,W);
          valid = valid_ref[3] && valid_prop[3];

          //  -- compute patch difference --
          if (valid){
            compute_dist_int<scalar_t,DIST_TYPE>(dist,
                         vid0[ibatch][ihead],vid1[ibatch][ihead],
                         ref_patch, prop_patch, 
                         ref_pix, prop_pix, valid_ref, valid_prop,
                         ps,pt,dilation,reflect_bounds,
                         patch_offset,invalid,T,F,H,W);
          }

          // -- assignent --
          if (!valid){ dist = invalid; }
          dists[ibatch][ihead][ti][nh][nw][ki][wh][ww] = dist;
          inds[ibatch][ihead][ti][nh][nw][ki][wh][ww][0] = prop_patch[0];
          inds[ibatch][ihead][ti][nh][nw][ki][wh][ww][1] = prop_patch[1];
          inds[ibatch][ihead][ti][nh][nw][ki][wh][ww][2] = prop_patch[2];

        } //  ww
      } // wh
    } // ki
  } // qi
} // fxn

void refinement_int_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor flows, torch::Tensor dists, torch::Tensor inds,
    int ws, int ps, int k, int stride0, int stride1, int dilation, int pt,
    bool restrict_radius, bool reflect_bounds, bool full_ws,
    int patch_offset,  int dist_type){

   // dists.shape = (B,HD,T,nH,nW,K,wr,wr)
   // inds.shape = (B,HD,T,nH,nW,K,wr,wr,3)
   // flows.shape = (B,HD,T,nH,nW,K,3)

   // -- num threads --
   int B = dists.size(0);
   int HD = dists.size(1);
   int T = dists.size(2);
   int nH = dists.size(3);
   int nW = dists.size(4);
   int Ks = dists.size(5);
   int wr = dists.size(6);

   int Q = T*nH*nW;
   int Ks_threads = std::min(Ks,9);
   int k_per_thread = ((Ks-1)/Ks_threads)+1;
   int wr_threads = std::min(wr,5);
   int wr_per_thread = ((wr-1)/wr_threads) + 1;
   // int wr_w_threads = std::min(wr_w,8);
   // int wr_w_per_thread = ((wr_w-1)/wr_w_threads) + 1;
   dim3 nthreads(Ks_threads,wr_threads,wr_threads);

   int q_per_thread = 2;
   int nquery_blocks = ((Q - 1) / q_per_thread) + 1;
   dim3 nblocks(nquery_blocks,B,HD);

   // int rem_blocks = (65535-1)/HD;
   // int q_per_thread = 2;
   // int nquery_blocks = ((Q - 1) / q_per_thread) + 1;
   // nquery_blocks = min(nquery_blocks,rem_blocks);
   // q_per_thread = ((Q - 1) / nquery_blocks) + 1;
   // dim3 nblocks(nquery_blocks,B,HD);

   // launch kernel
   if (dist_type == 0){
     AT_DISPATCH_FLOATING_TYPES(vid0.type(),"refinement_forward_kernel", ([&] {
          refinement_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          flows.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,8,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,9,torch::RestrictPtrTraits>(),
          wr, ws, ps, pt, k, stride0, stride1, dilation,
          restrict_radius, reflect_bounds, full_ws, patch_offset, 
          q_per_thread, k_per_thread, wr_per_thread);
        }));
   }else if (dist_type == 1){
     AT_DISPATCH_FLOATING_TYPES(vid0.type(),"refinement_forward_kernel", ([&] {
          refinement_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          flows.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,8,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,9,torch::RestrictPtrTraits>(),
          wr, ws, ps, pt, k, stride0, stride1, dilation,
          restrict_radius, reflect_bounds, full_ws, patch_offset,
          q_per_thread, k_per_thread, wr_per_thread);
        }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}

