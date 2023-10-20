
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
__global__ void quadref_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    // const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> deno0,
    // const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> deno1,
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
  int C = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int Q = dists.size(2);
  // int K = qinds.size(3); // == 5?
  const K = 4;

  // -- invalid constant --
  scalar_t invalid = (scalar_t)__int_as_float(0x7f800000);
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
  wh,ww = 0,0
  int qindex,qindex_tmp;

  // -- fwd decls --
  // int centers[5][2];
  // int prop_patch[3];
  // int prop_pix[3];
  int patches[5][3];
  int pixels[5][3];
  int locs[5][3];
  // int ref_patch[3];
  // int ref_pix[3];
  bools valids[5][3];
  bool valid;
  // bool valids_anchor[5];
  // bool valid_prop[5][4];
  // bool valid_ref[5][4];
  scalar_t pixels[5];
  scalar_t dist,_dist;

  // int ti,hi,wi;
  // int n_ti,n_hi,n_wi;
  // int vH,vW,vT,nH,nW,nT;
  // bool valid_anchor,valid_n,valid;
  // bool vvalid_t,vvalid_h,vvalid_w,vvalid;
  // bool nvalid_t,nvalid_h,nvalid_w,nvalid;

  // -- cleaner code --
  int center_offsets[4] = {off_H0,off_W0,off_H1,off_W1};
  int patch_offset = psHalf + adj;


  for (int q_index = 0; q_index < q_per_thread; q_index++){

    //---------------------------
    //       Anchor Pixel
    //---------------------------

    // -- block start --
    qi = q_start + q_index;
    if (qi >= Q){ continue; }
    qindex = qi + q_shift;

    // -- pixel location from query index --
    get_pixel_loc(patches[0],qindex,qindex_tmp,stride0,nW0,nHW0,H,W);

    // -- check bounds of pixel location --
    check_bounds(valid,patches[0],T,H,W);

    // -- search region offsets --
    set_search_offsets(wrOff_h,wrOff_w, patches[0][1], patches[0][2],
                       stride1, wrHalf_h, wrHalf_w, wrMax_h, wrMax_w, H, W, full_ws);

    // -- [unused] set search bounds for [optionally] expanded region --
    // set_search_minmax(wrMax_h, wrMin_h, wrOff_h, wr_h, stride1, full_ws);
    // set_search_minmax(wrMax_w, wrMin_w, wrOff_w, wr_w, stride1, full_ws);

    // ---------------------------------------
    //     for each neighbor in k_search
    // ---------------------------------------
    for(int _si = 0; _si < k_per_thread; _si++){
      si = threadIdx.x + blockDim.x*_si;
      if (si >= K){ continue; }

      // -- unpack base --
#pragma unroll
      for (int k=1; k < K; k++){
        patches[k][0] = qinds[ibatch][ihead][qi][k][0]; // no search
        patches[k][1] = qinds[ibatch][ihead][qi][k][1];
        patches[k][2] = qinds[ibatch][ihead][qi][k][2];
      }

      // --------------------
      //      init dists
      // --------------------
      dist = 0;

      // -- check bounds of pixel location --
// #pragma unroll
//       for (int k=0; k < K; k++){
//         check_bounds(valid_prop[3],prop_patch,T,H,W);
//         valid = valid_ref[3] && valid_prop[3];
//       }

      //  -- compute patch difference --
      if (valid){
        compute_quad_dist<scalar_t,DIST_TYPE>(dist,
                     vid0[ibatch][ihead],vid1[ibatch][ihead],
                     patches, locs, pixels, valids,
                     ps,pt,dilation,reflect_bounds,
                     patch_offset,center_offsets,invalid,
                     T,C,H,W,pixes,_dist);
      }

      // -- assignent --
      if (!valid){ dist = invalid; }
#pragma unroll
      dists[ibatch][ihead][qi][0][wh][ww] = dist;
      for (int k=0; k < K; k++){
        inds[ibatch][ihead][qi][k][wh][ww][0] = patches[k][0];
        inds[ibatch][ihead][qi][k][wh][ww][1] = patches[k][1];
        inds[ibatch][ihead][qi][k][wh][ww][2] = patches[k][2];
      }

    } // si
  } // qi
} // fxn

void quadref_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    // const torch::Tensor deno0, const torch::Tensor deno1,
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
   int ksearch_threads = 1;//std::min(ksearch,12);
   int wr_h_threads = std::min(wr_h,8);
   int wr_w_threads = std::min(wr_w,8);
   int k_per_thread = ksearch;//((ksearch-1)/ksearch_threads)+1;
   assert(ksearch == 5);
   assert(wr_h == 1);
   assert(wr_w == 1);
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
     AT_DISPATCH_FLOATING_TYPES(vid0.type(),"quadref_forward_kernel", ([&] {
          quadref_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          // deno0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          // deno1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          qinds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
          wr_h, wr_w, ws_h2, ws_w2, ps, pt, k, stride0, stride1, dilation,
          q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws,
          use_adj, off_H0, off_W0, off_H1, off_W1,
          q_per_thread, wr_h_per_thread, wr_w_per_thread);
        }));
   }else if (dist_type == 1){
     AT_DISPATCH_FLOATING_TYPES(vid0.type(),"quadref_forward_kernel", ([&] {
          quadref_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          // deno0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          // deno1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          qinds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
          wr_h, wr_w, ws_h2, ws_w2, ps, pt, k, stride0, stride1, dilation,
          q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws,
          use_adj, off_H0, off_W0, off_H1, off_W1,
          q_per_thread, wr_h_per_thread, wr_w_per_thread);
        }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}



/****************************

       Backward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void quadref_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    // const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> deno0,
    // const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> deno1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int q_shift, int stride0, int nH0, int nW0, int nHW0,
    int off_H0, int off_W0, int off_H1, int off_W1,
    int ps, int pt, int dilation, bool use_adj, bool reflect_bounds,
    int q_per_thread, int neigh_per_thread, int chnls_per_thread) {

  // -- shape --
  int B = grad_dists.size(0);
  int Q = grad_dists.size(2);
  // int K =  grad_dists.size(3);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int C = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  const int K = 4;

  // -- fwd decl registers --
  int patches[K][3];
  int pixels_i[K][3];
  scalar_t pixels_v[K];
  // int locs[K][3];
  bool valids[K][4];
  // int ref_patch[3];
  // int prop_patch[3];
  // int ref[3];
  // int prop[3];
  // bool valid_ref[4];
  // bool valid_prop[4];
  int qindex,qindex_tmp;
  bool valid;
  scalar_t weight,pix;

  // -- declare constants --
  int psHalf = ps/2;
  int adj = use_adj ? psHalf : 0;

  // -- limits --
  int i0_max = inds.size(2); // nq
  int i1_max = 1;//inds.size(3); // k

  // -- get indices --
  int ibatch = blockIdx.x / HD;
  int ihead = blockIdx.x - ibatch*HD;
  int i0_start = q_per_thread * (threadIdx.x + blockDim.x * blockIdx.y);
  int i1_start = threadIdx.y * neigh_per_thread;
  int c0_start = blockIdx.z * chnls_per_thread;

  // -- get block limits --
  int i0_end = min(i0_start + q_per_thread,i0_max);
  int i1_end = min(i1_start + neigh_per_thread,i1_max);
  int c0_end = min(c0_start + chnls_per_thread,C);

  // -- color offset --
  int c0 = 0;
  int c0_dist = c0_end - c0_start;
  int c0_offset = 0;

  // -- cleaner code --
  int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};
  int patch_offset = psHalf + adj;

  // -- each region --
  for (int i0=i0_start; i0 < i0_end; i0++){

    // -- full-resolution video query index --
    qindex = i0 + q_shift;

    // -- pixel location from query index --
    get_pixel_loc(ref_patch,qindex,qindex_tmp,stride0,nW0,nHW0,H,W);

    // -- channel access offset --
    c0_offset = __float2int_rd(c0_dist * rand_nums[i0][0][0]);

    // -- k neighbors --
    for (int i1=i1_start; i1 < i1_end; i1++){

      // -- proposed location --
#pragma unroll
      for (int k = 0; k < K; k++){
        patches[k][0] = inds[ibatch][ihead][i0][k][0];
        patches[k][1] = inds[ibatch][ihead][i0][k][1];
        patches[k][2] = inds[ibatch][ihead][i0][k][2];
      }
      weight = grad_dists[ibatch][ihead][i0][0];

      // -- update patch --
      // torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid0,
      // torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> grad_vid1,
      // const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid0,
      // const torch::TensorAccessor<scalar_t,4,torch::RestrictPtrTraits,int32_t> vid1,
      // scalar_t weight, int* ref_patch, int* prop_patch,
      // int ps, int pt, int dilation, bool reflect_bounds,
      // int* center_offsets, int patch_offset,
      // int c0, int c0_start, int c0_end, int c0_offset, int c0_dist,
      // int* ref, int* prop, bool* valid_ref, bool* valid_prop, bool valid,
      // int T, int C, int H, int W, scalar_t pix0, scalar_t pix1, scalar_t pix){

      update_bwd_quad_patch<scalar_t,DIST_TYPE>(
                           grad_vid0[ibatch][ihead],grad_vid1[ibatch][ihead],
                           vid0[ibatch][ihead],vid1[ibatch][ihead],
                           // deno0[ibatch][ihead],deno1[ibatch][ihead],
                           weight,patches,pixels_i,pixels_v,valids,valid,
                           ps,pt,dilation,reflect_bounds,
                           center_offsets,patch_offset,
                           c0,c0_start,c0_end,c0_offset,c0_dist,
                           T,C,H,W,pix);
    }
  }
}

void quadref_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    // torch::Tensor deno0, torch::Tensor deno1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1,
    bool use_rand, bool exact, int dist_type,
    int queries_per_thread, int neigh_per_thread, int channel_groups) {

  // -- unpack --
  int B = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int C = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int nqueries = inds.size(2);
  int K = inds.size(3);
  int BHD = B*HD;
  int nHW0 = nH0 * nW0;
  assert(pt == 1);

  // -- compute number of neighbor threads --
  int neigh_nthreads = (K-1) / neigh_per_thread + 1;
  if (exact){
    neigh_nthreads = 1;
    neigh_per_thread = K;
  }

  // -- compute number of color blocks --
  channel_groups = (channel_groups > 0) ? channel_groups : C;
  channel_groups = std::min(channel_groups,C);
  int chnls_nblocks = exact ? C : channel_groups;
  int chnls_per_thread = (C-1) / chnls_nblocks + 1;

  // -- compute number of blocks --
  int MAX_NTHREADS = 28*32;
  int query_nthreads = std::max(MAX_NTHREADS/neigh_nthreads,1);
  int queries_per_block = queries_per_thread * query_nthreads;
  int nblocks_queries = ((nqueries - 1) / queries_per_block) + 1;
  if (exact){
    queries_per_thread = nqueries;
    query_nthreads = 1;
    nblocks_queries = 1;
  }

  // -- launch params --
  dim3 nblocks(BHD,nblocks_queries,chnls_nblocks);
  dim3 nthreads(query_nthreads, neigh_nthreads);//, chnls_nthreads);

  // -- view launch info --
  // fprintf(stdout,"BHD,nblocks_queries,chnls_nblocks: %d,%d,%d\n",
  //         BHD,nblocks_queries,chnls_nblocks);
  // fprintf(stdout,"query_nthreads,neigh_nthreads: %d,%d\n",
  //         query_nthreads,neigh_nthreads);

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
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"quadref_backward_kernel", ([&] {
    quadref_backward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          // deno0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          // deno1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, use_adj, reflect_bounds,
          queries_per_thread, neigh_per_thread, chnls_per_thread);
    }));
  }else if (dist_type == 1){ // l2
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"quadref_backward_kernel", ([&] {
    quadref_backward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          // deno0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          // deno1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, use_adj, reflect_bounds,
          queries_per_thread, neigh_per_thread, chnls_per_thread);
    }));
  }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");    }
}

