
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
__global__ void non_local_search_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> inds,
    int ws_h, int ws_w, int wt, int ps, int pt,
    int stride0, int stride1, int dilation,
    int q_shift, int nH0, int nW0, int nHW0,
    bool reflect_bounds, bool full_ws, bool search_abs,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1,
    int q_per_thread, int ws_h_per_thread, int ws_w_per_thread){

  // -- unpack shape --
  int B = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int C = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int Q = dists.size(2);
  int ST = dists.size(3);

  // -- invalid constant --
  float invalid = __int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }

  // -- search region offsets --
  int psHalf = (ps)/2;
  int wsHalf_h = (ws_h)/2;
  int wsHalf_w = (ws_w)/2;
  int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  int wsMax_w = stride1*(ws_w-1-wsHalf_w);
  int adj = use_adj ? psHalf : 0;
  int wsOff_h,wsOff_w;

  // -- time indices --
  int t_shift;
  int t_max;

  // -- cuda index --
  int ibatch = blockIdx.x;
  int ihead = blockIdx.y;
  int q_start = blockIdx.z*q_per_thread;
  int qi,ws_i,ws_j;

  // accumulate time offsets
  int t_inc = 0;
  int dir = 0;
  int prev_ti = -1;
  bool swap_dir = false;

  // decls
  int ref_patch[3];
  int prop_patch[3];
  int frame_anchor[3];
  int ref_pix[3];
  int prop_pix[3];
  bool valid;
  bool valid_ref_patch,valid_prop_patch;
  bool valid_ref[4];
  bool valid_prop[4];

  // -- cleaner code --
  int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};
  int patch_offset = psHalf + adj;

  // -- indexing --
  int qindex,qindex_tmp;
  scalar_t dist,pix0,pix1,_dist;

  for (int q_index = 0; q_index < q_per_thread; q_index++){

    //---------------------------
    //       Anchor Pixel
    //---------------------------

    // -- block start --
    qi = q_start + q_index;
    if (qi >= Q){ continue; }
    qindex = qi + q_shift;

    // -- pixel location from query index --
    get_pixel_loc(ref_patch,qindex,qindex_tmp,stride0,nW0,nHW0,H,W);

    // -- check bounds of pixel location --
    check_bounds(valid_ref_patch,ref_patch,T,H,W);

    // -- search region offsets --
    set_search_offsets(wsOff_h,wsOff_w, ref_patch[1], ref_patch[2], stride1,
                       wsHalf_h, wsHalf_w, wsMax_h, wsMax_w, H, W, full_ws);

    // -- temporal search bounds --
    set_time_range(t_max,t_shift,ref_patch[0],T,wt);

    // -- init search params --
    frame_anchor[0] = ref_patch[0];
    frame_anchor[1] = ref_patch[1];
    frame_anchor[2] = ref_patch[2];
    prev_ti = ref_patch[0];
    t_inc = 0;
    swap_dir = false;
    dir = 0;

    // -- search across time --
    for(int st_i = 0; st_i < ST; st_i++){

      // ---------------------------------------
      //       compute search center
      // ---------------------------------------

      // -- increment frame index --
      increment_frame(frame_anchor[0],prev_ti,t_inc,swap_dir,dir,ref_patch[0],t_max);

      // -- possibly reset (frame_anchor <- reference_patch) --
      reset_centers(frame_anchor,ref_patch,swap_dir);

      // -- compute offset with optical flow --
      update_centers<scalar_t>(frame_anchor[1],frame_anchor[2],dir,H,W,
                               fflow[ibatch][prev_ti],bflow[ibatch][prev_ti]);
      
      // ---------------------------------------
      //          spatial searching
      // ---------------------------------------
  
      // -- search across space --
      for (int _xi = 0; _xi < ws_h_per_thread; _xi++){
        ws_i = threadIdx.x + blockDim.x*_xi;
        if (ws_i >= ws_h){ continue; }
        for (int _yi = 0; _yi < ws_w_per_thread; _yi++){
          ws_j = threadIdx.y + blockDim.y*_yi;
          if (ws_j >= ws_w){ continue; }
  
          // -- compute proposed location --
          set_search_patch(prop_patch,frame_anchor,stride1,
                           ws_i,ws_j,wsOff_h,wsOff_w,search_abs);
          check_bounds(valid_prop_patch,prop_patch,T,H,W);
          valid = valid_ref_patch && valid_prop_patch;

          // -- init dist --
          dist = 0;

          //  -- compute patch difference --
          if (valid){
            compute_dist<scalar_t,DIST_TYPE>(dist,
                         vid0[ibatch][ihead],vid1[ibatch][ihead],
                         ref_patch, prop_patch, 
                         ref_pix, prop_pix, valid_ref, valid_prop,
                         ps,pt,dilation,reflect_bounds,
                         patch_offset,center_offsets,invalid,
                         T,C,H,W,pix0,pix1,_dist);
          }

          // -- assignent --
          if (!valid){ dist = invalid; }
          dists[ibatch][ihead][qi][st_i][ws_i][ws_j] = dist;
          inds[ibatch][ihead][qi][st_i][ws_i][ws_j][0] = prop_patch[0];
          inds[ibatch][ihead][qi][st_i][ws_i][ws_j][1] = prop_patch[1];
          inds[ibatch][ihead][qi][st_i][ws_i][ws_j][2] = prop_patch[2];
          
        }
      }
    }
  }
}

void non_local_search_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int wt, int ps, int k, int dist_type,
    int stride0, int stride1, int dilation, int pt, int q_shift,
    bool reflect_bounds, bool full_ws, bool search_abs,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1){


// void non_local_search_forward_cuda(
//     const torch::Tensor vid0, const torch::Tensor vid1,
//     const torch::Tensor fflow, const torch::Tensor bflow,
//     torch::Tensor dists, torch::Tensor inds,
//     int q_shift, int stride0, int nH0, int nW0,
//     int off_H0, int off_W0, int off_H1, int off_W1,
//     int ps, int pt, int ws_h, int ws_w, int wt,
//     int dilation, int stride1, bool use_adj,
//     bool reflect_bounds, bool search_abs, bool full_ws, int dist_type){

    // # -- launch params --
    // w_threads = min(ws,32)
    // nthreads = (w_threads,w_threads)
    // ws_iters = (ws-1)//w_threads + 1
    // nblocks = (nq-1)//batches_per_block+1
    // fprintf(stdout,"nH0,nW0: %d,%d\n",nH0,nW0);

   // launch params
   // our many (too many?) registers limit the number of threads

   // -- derived quantities --
   int H = vid0.size(4);
   int W = vid0.size(5);
   int nH0 = (H-1)/stride0+1;
   int nW0 = (W-1)/stride0+1;
   int nHW0 = nH0 * nW0;

   // -- threads --
   int nheads = dists.size(1);
   int nqueries = dists.size(2);
   int st = dists.size(3);
   int ws_h = dists.size(4);
   int ws_w = dists.size(5);
   int ws_h_threads = std::min(ws_h,27);
   int ws_w_threads = std::min(ws_w,27);
   int ws_h_per_thread = ((ws_h-1)/ws_h_threads) + 1;
   int ws_w_per_thread = ((ws_w-1)/ws_w_threads) + 1;
   dim3 nthreads(ws_h_threads,ws_w_threads);

   // -- nblocks --
   int B = vid0.size(0);
   int HD = vid0.size(1);
   int q_per_thread = 4;
   int nquery_blocks = ((nqueries - 1) / q_per_thread) + 1;
   dim3 nblocks(B,HD,nquery_blocks);

   // fprintf(stdout,"q_shift, nqueries: %d,%d\n",q_shift,nqueries);
   // fprintf(stdout,"dilation, reflect_bounds: %d,%d\n",dilation, reflect_bounds);
   // fprintf(stdout,"stride0,stride1: %d,%d\n",stride0,stride1);
   // fprintf(stdout,"full_ws: %d\n",full_ws);

   // fprintf(stdout,"Q: %d\n",nqueries);
   // fprintf(stdout,"ps,pt,nH0,nW0,wt,chnls,stride0,ws_h,ws_w: %d,%d,%d,%d,%d,%d,%d,%d,%d\n",ps,pt,nH0,nW0,wt,chnls,stride0,ws_h,ws_w);
   // fprintf(stdout,"bsize,nheads,nquery_blocks: %d,%d,%d\n",
   //         bsize,nheads,nquery_blocks);
   // fprintf(stdout,"q_per_thread,nquery_blocks,ws_h_threads,ws_w_threads: %d,%d,%d,%d\n",
   //         q_per_thread,nquery_blocks,ws_h_threads,ws_w_threads);
   // fprintf(stdout,"reflect_bounds,search_abs,full_ws,anchor_self,use_self: %d,%d,%d,%d,%d\n",
   //         reflect_bounds,search_abs,full_ws,anchor_self,use_self);
   // fprintf(stdout,"ws_h_per_thread,ws_w_per_thread,ws_h,ws_w: %d,%d,%d,%d,\n",ws_h_per_thread,ws_w_per_thread,ws_h,ws_w);
    
    // int q_shift, int stride0, int nH0, int nW0,
    // int off_H0, int off_W0, int off_H1, int off_W1,
    // int ps, int pt, int ws_h, int ws_w, int wt,
    // int dilation, int stride1,
    // bool use_adj, bool reflect_bounds, bool search_abs,
    // bool full_ws, int ws_h_per_thread, int ws_w_per_thread, int q_per_thread){

   // launch kernel
   if (dist_type == 0){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),"non_local_search_forward_kernel", ([&] {
       non_local_search_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
            ws_h, ws_w, wt, ps, pt, stride0, stride1, dilation, 
            q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws, search_abs,
            use_adj, off_H0, off_W0, off_H1, off_W1,
            q_per_thread, ws_h_per_thread, ws_w_per_thread);
          }));
   }else if(dist_type == 1){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),"non_local_search_forward_kernel", ([&] {
       non_local_search_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
            ws_h, ws_w, wt, ps, pt, stride0, stride1, dilation, 
            q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws, search_abs,
            use_adj, off_H0, off_W0, off_H1, off_W1,
            q_per_thread, ws_h_per_thread, ws_w_per_thread);
          }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void non_local_search_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
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
  int K =  grad_dists.size(3);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int C = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);

  // -- fwd decl registers --
  int ref_patch[3];
  int prop_patch[3];
  int ref[3];
  int prop[3];
  bool valid_ref[4];
  bool valid_prop[4];
  int qindex,qindex_tmp;
  bool valid;
  scalar_t weight,pix0,pix1,pix;

  // -- declare constants --
  int psHalf = ps/2;
  int adj = use_adj ? psHalf : 0;

  // -- limits --
  int i0_max = inds.size(2); // nq
  int i1_max = inds.size(3); // k

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
      prop_patch[0] = inds[ibatch][ihead][i0][i1][0];
      prop_patch[1] = inds[ibatch][ihead][i0][i1][1];
      prop_patch[2] = inds[ibatch][ihead][i0][i1][2];
      weight = grad_dists[ibatch][ihead][i0][i1];

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

      update_bwd_patch<scalar_t,DIST_TYPE>(
                       grad_vid0[ibatch][ihead],grad_vid1[ibatch][ihead],
                       vid0[ibatch][ihead],vid1[ibatch][ihead],
                       weight,ref_patch,prop_patch,
                       ps,pt,dilation,reflect_bounds,
                       center_offsets,patch_offset,
                       c0,c0_start,c0_end,c0_offset,c0_dist,
                       ref,prop,valid_ref,valid_prop,valid,
                       T,C,H,W,pix0,pix1,pix);
    }
  }
}

void non_local_search_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
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

  // -- compute number of color threads --
  channel_groups = (channel_groups > 0) ? channel_groups : C;
  channel_groups = std::min(channel_groups,C);
  int chnls_nblocks = exact ? C : channel_groups;
  int chnls_per_thread = (C-1) / chnls_nblocks + 1;
  // int chnls_nblocks = exact ? C : channel_groups;
  // int chnls_per_thread = C;

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
  // fprintf(stdout,"queries_per_thread,neigh_per_thread,chnls_per_thread: %d,%d,%d\n",
  //         queries_per_thread,neigh_per_thread,chnls_per_thread);
  // fprintf(stdout,"nblocks_queries,chnls_nblocks,BHD: %d,%d,%d\n",
  //         nblocks_queries,chnls_nblocks,BHD);

  // int ibatch = blockIdx.x / HD;
  // int ihead = blockIdx.x - ibatch*HD;
  // int i0_start = q_per_thread * (threadIdx.x + blockDim.x * blockIdx.y);
  // int i1_start = threadIdx.y * neigh_per_thread;
  // int c0_start = blockIdx.z * chnls_per_thread;

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
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"non_local_search_backward_kernel", ([&] {
    non_local_search_backward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, use_adj, reflect_bounds,
          queries_per_thread, neigh_per_thread, chnls_per_thread);
    }));
  }else if (dist_type == 1){ // l2
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"non_local_search_backward_kernel", ([&] {
    non_local_search_backward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
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


