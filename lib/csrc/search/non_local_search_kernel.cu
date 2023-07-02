
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
    bool reflect_bounds, bool full_ws, bool full_ws_time,
    bool search_abs, int patch_offset,
    int off_H0, int off_W0, int off_H1, int off_W1,
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
  // int psHalf = (ps)/2;
  int wsHalf_h = (ws_h)/2;
  int wsHalf_w = (ws_w)/2;
  int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  int wsMax_w = stride1*(ws_w-1-wsHalf_w);
  // int adj = use_adj ? psHalf : 0;
  int wsOff_h,wsOff_w;

  // -- time indices --
  int t_shift;
  int t_max;

  // -- cuda index --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int q_start = blockIdx.x*q_per_thread;
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
      
      // -- search region offsets --
      set_search_offsets(wsOff_h,wsOff_w, frame_anchor[1], frame_anchor[2], stride1,
			 wsHalf_h, wsHalf_w, wsMax_h, wsMax_w, H, W, full_ws_time);

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
    bool reflect_bounds, bool full_ws, bool full_ws_time,
    bool search_abs, bool use_adj,
    int off_H0, int off_W0, int off_H1, int off_W1){

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
   int q_per_thread = 2;
   int nquery_blocks = ((nqueries - 1) / q_per_thread) + 1;
   dim3 nblocks(nquery_blocks,B,HD);

   // -- share --
   int psHalf = ps/2;
   int adj = use_adj ? psHalf : 0;
   // int patch_offset = adj - psHalf;
   int patch_offset = adj - psHalf;

   // -- viz --
   // fprintf(stdout,"ws_h,ws_w: %d,%d\n",ws_h,ws_w);
   // fprintf(stdout,"nquery_blocks,B,HD: %d,%d,%d\n",nquery_blocks,B,HD);

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
            q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws, full_ws_time,
            search_abs, patch_offset, off_H0, off_W0, off_H1, off_W1,
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
            q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws, full_ws_time,
            search_abs, patch_offset, off_H0, off_W0, off_H1, off_W1,
            q_per_thread, ws_h_per_thread, ws_w_per_thread);
          }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}


/****************************

      Forward Pass (v2)

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void non_local_search_forward_v2_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> inds,
    int ws_h, int ws_w, int wt, int ps, int pt,
    int stride0, int stride1, int dilation,
    int q_shift, int nH0, int nW0, int nHW0,
    bool reflect_bounds, bool full_ws, bool full_ws_time,
    bool search_abs, int patch_offset,
    int off_H0, int off_W0, int off_H1, int off_W1,
    int q_per_thread, int ftrs_per_thread){

  // -- unpack shape --
  int nbatch = vid0.size(0);
  int nheads = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
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
  // int psHalf = (ps)/2;
  int wsHalf_h = (ws_h)/2;
  int wsHalf_w = (ws_w)/2;
  int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  int wsMax_w = stride1*(ws_w-1-wsHalf_w);
  // int adj = use_adj ? psHalf : 0;
  int wsOff_h,wsOff_w;

  // -- time indices --
  int t_shift;
  int t_max;

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

  // -- indexing --
  int qi,qindex,qindex_tmp;
  scalar_t dist,pix0,pix1,_dist;

  // -- location to fill --
  int q_start = (blockIdx.x*blockDim.x+threadIdx.x)*q_per_thread;
  int ws_i = blockIdx.y / ws_h;
  int ws_j = blockIdx.y % ws_h;
  int ihead = blockIdx.z/nbatch;
  int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;

  // -- feature chunk --
  int iftr;
  int ftr_start = threadIdx.z * ftrs_per_thread;
  int ftr_end = min(F,ftr_start + ftrs_per_thread);

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
      
      // -- search region offsets --
      set_search_offsets(wsOff_h,wsOff_w, frame_anchor[1], frame_anchor[2], stride1,
                         wsHalf_h, wsHalf_w, wsMax_h, wsMax_w, H, W, full_ws_time);

      // ---------------------------------------
      //          spatial searching
      // ---------------------------------------
  
      // -- compute proposed location --
      set_search_patch(prop_patch,frame_anchor,stride1,
                       ws_i,ws_j,wsOff_h,wsOff_w,search_abs);
      check_bounds(valid_prop_patch,prop_patch,T,H,W);
      valid = valid_ref_patch && valid_prop_patch;

      // -- init dist --
      dist = 0;

      //  -- compute patch difference --
      if (valid){

        compute_dist_v2<scalar_t,DIST_TYPE>(dist,
                                            vid0[ibatch][ihead],vid1[ibatch][ihead],
                                            ref_patch, prop_patch, 
                                            ref_pix, prop_pix, valid_ref, valid_prop,
                                            ps,pt,dilation,reflect_bounds,
                                            patch_offset,center_offsets,invalid,
                                            iftr,ftr_start,ftr_end,
                                            T,H,W,pix0,pix1,_dist);

      }

      // -- assignent --
      if (!valid){ dist = invalid; }
      atomicAdd(&dists[ibatch][ihead][qi][st_i][ws_i][ws_j],dist);
      inds[ibatch][ihead][qi][st_i][ws_i][ws_j][0] = prop_patch[0];
      inds[ibatch][ihead][qi][st_i][ws_i][ws_j][1] = prop_patch[1];
      inds[ibatch][ihead][qi][st_i][ws_i][ws_j][2] = prop_patch[2];

    }
  }
}

void non_local_search_forward_v2_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int wt, int ps, int k, int dist_type,
    int stride0, int stride1, int dilation, int pt, int q_shift,
    bool reflect_bounds, bool full_ws, bool full_ws_time,
    bool search_abs, bool use_adj,
    int off_H0, int off_W0, int off_H1, int off_W1){

   // -- derived quantities --
   int nftrs = vid0.size(3);
   int H = vid0.size(4);
   int W = vid0.size(5);
   int nH0 = (H-1)/stride0+1;
   int nW0 = (W-1)/stride0+1;
   int nHW0 = nH0 * nW0;

   // -- threads per block --
   int MAX_THREADS = 760;
   int xdim = 330;
   int fpb = 2;//MAX_THREADS/16; // num of nftrs per block
   int ftrs_per_thread = ((nftrs - 1)/fpb) + 1; // num of nftrs per thread
   dim3 nthreads(xdim,fpb);
    
   // -- blocks per grid --
   int nbatch = dists.size(0);
   int nheads = dists.size(1);
   int nqueries = dists.size(2);
   int st = dists.size(3);
   int ws_h = dists.size(4);
   int ws_w = dists.size(5);
   int ws2 = ws_h * ws_w;
   // int ws_h_threads = std::min(ws_h,27);
   // int ws_w_threads = std::min(ws_w,27);
   // int ws_h_per_thread = ((ws_h-1)/ws_h_threads) + 1;
   // int ws_w_per_thread = ((ws_w-1)/ws_w_threads) + 1;
   int q_per_thread = 1;
   // int nquery_blocks = ((nqueries - 1) / q_per_thread) + 1;
   dim3 nblocks(1,ws2,nheads*nbatch);
   nblocks.x = ceil(double(nqueries)/double(q_per_thread*nthreads.x));

   // -- share --
   int psHalf = ps/2;
   int adj = use_adj ? psHalf : 0;
   int patch_offset = adj - psHalf;
   // int patch_offset = psHalf - adj;

   // -- viz --
   // fprintf(stdout,"ws_h,ws_w: %d,%d\n",ws_h,ws_w);
   // fprintf(stdout,"nquery_blocks,B,HD: %d,%d,%d\n",nquery_blocks,B,HD);

   // launch kernel
   if (dist_type == 0){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),"non_local_search_forward_v2_kernel", ([&] {
       non_local_search_forward_v2_kernel<scalar_t,0><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
            ws_h, ws_w, wt, ps, pt, stride0, stride1, dilation, 
            q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws, full_ws_time,
            search_abs, patch_offset, off_H0, off_W0, off_H1, off_W1,
            q_per_thread,ftrs_per_thread);
          }));
   }else if(dist_type == 1){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),"non_local_search_forward_v2_kernel", ([&] {
       non_local_search_forward_v2_kernel<scalar_t,1><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
            ws_h, ws_w, wt, ps, pt, stride0, stride1, dilation, 
            q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws, full_ws_time,
            search_abs, patch_offset, off_H0, off_W0, off_H1, off_W1,
            q_per_thread,ftrs_per_thread);
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
    // torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> count0,
    // torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> count1,
    int q_shift, int stride0, int nH0, int nW0, int nHW0,
    int off_H0, int off_W0, int off_H1, int off_W1,
    int ps, int pt, int dilation, int patch_offset,
    bool reflect_bounds, int ftrs_per_thread) {

  // -- shape --
  int nbatch = grad_dists.size(0);
  int Q = grad_dists.size(2);
  int K =  grad_dists.size(3);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
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
  int iftr;


  // -- location to fill --
  int i0 = blockIdx.x*blockDim.x+threadIdx.x;
  int i1 = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/nbatch;
  int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;

  // -- feature chunk --
  int ftr_start = threadIdx.z * ftrs_per_thread;
  int ftr_end = min(F,ftr_start + ftrs_per_thread);

  // -- cleaner code --
  int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};

  // -- each region --
  if ((i0 < Q) && (i1 < K)){

    // -- full-resolution video query index --
    qindex = i0 + q_shift;

    // -- pixel location from query index --
    get_pixel_loc(ref_patch,qindex,qindex_tmp,stride0,nW0,nHW0,H,W);

    // -- proposed location --
    prop_patch[0] = inds[ibatch][ihead][i0][i1][0];
    prop_patch[1] = inds[ibatch][ihead][i0][i1][1];
    prop_patch[2] = inds[ibatch][ihead][i0][i1][2];
    weight = grad_dists[ibatch][ihead][i0][i1];

    // -- add count --
    // if (i1 == 0){
    //   if (ftr_start == 0){
    //     atomicAdd(&(count0[ibatch][ihead][ref[0]][ref[1]][ref[2]]),1);
    //   }
    //   // if (valid_prop[3]){
    //   //   atomicAdd(&(count1[ibatch][ihead][prop[0]][prop[1]][prop[2]]),1);
    //   // }
    // }


    // -- update patch --
    update_bwd_patch<scalar_t,DIST_TYPE>(
                     grad_vid0[ibatch][ihead],grad_vid1[ibatch][ihead],
                     vid0[ibatch][ihead],vid1[ibatch][ihead],
                     // count0[ibatch][ihead],count1[ibatch][ihead],
                     weight,ref_patch,prop_patch,
                     ps,pt,dilation,reflect_bounds,
                     center_offsets,patch_offset,
                     iftr,ftr_start,ftr_end,
                     ref,prop,valid_ref,valid_prop,valid,
                     T,H,W,pix0,pix1,pix,i1);

  }
}

void non_local_search_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds, bool use_adj,
    int off_H0, int off_W0, int off_H1, int off_W1, int dist_type) {

  // -- unpack --
  int B = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int nqueries = inds.size(2);
  int K = inds.size(3);
  int BHD = B*HD;
  int nHW0 = nH0 * nW0;
  assert(pt == 1);

  // -- launch parameters --
  int nbatch = grad_dists.size(0);
  int nheads = grad_dists.size(1);
  int nq = grad_dists.size(2);
  int k = grad_dists.size(3);
  int ftr_threads = min(15,F);
  dim3 threadsPerBlock(10,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, nheads*nbatch);
  blocksPerGrid.x = ceil(double(nq)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
  int ftrs_per_thread = (F-1)/ftr_threads+1;

  // -- shared --
  int psHalf = ps/2;
  int adj = use_adj ? psHalf : 0;
  int patch_offset = adj - psHalf;
  // int patch_offset = psHalf - adj;

 

  // -- allocate counts --
  // auto options = torch::TensorOptions()
  //   .dtype(torch::kInt32)
  //   .layout(torch::kStrided)
  //   .device(torch::kCUDA, grad_vid0.device().index());
  // auto count0 = torch::zeros({B,HD,T,H,W},options);
  // auto count1 = torch::zeros({B,HD,T,H,W},options);

  // -- view launch info --
  // fprintf(stdout,"BHD,nblocks_queries,chnls_nblocks: %d,%d,%d\n",
  //         BHD,nblocks_queries,chnls_nblocks);
  // fprintf(stdout,"query_nthreads,neigh_nthreads: %d,%d\n",
  //         query_nthreads,neigh_nthreads);


  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"non_local_search_backward_kernel", ([&] {
    non_local_search_backward_kernel<scalar_t,0><<<blocksPerGrid, threadsPerBlock>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          // count0.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          // count1.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);
    }));
  }else if (dist_type == 1){ // l2
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"non_local_search_backward_kernel", ([&] {
    non_local_search_backward_kernel<scalar_t,1><<<blocksPerGrid, threadsPerBlock>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          // count0.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          // count1.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);
    }));
  }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");    }

  // -- normalize --
  // count0 = count0.view({B, HD, T, 1, H, W});
  // count1 = count1.view({B, HD, T, 1, H, W});
  // grad_vid0 /= count0;
  // grad_vid1 /= count1;

}


