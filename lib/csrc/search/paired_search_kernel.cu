
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include "paired_details.cu"

using namespace at;


/****************************

       Forward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void paired_search_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame0,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> flow,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> inds,
    int ws_h, int ws_w, int ps,
    int stride0, int stride1, int dilation,
    int q_shift, int nH0, int nW0,
    bool reflect_bounds, bool full_ws, bool full_ws_time,
    int patch_offset, int off_H0, int off_W0, int off_H1, int off_W1,
    int q_per_thread, int ws_h_per_thread, int ws_w_per_thread){

  // -- unpack shape --
  int B = frame0.size(0);
  int HD = frame0.size(1);
  int C = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);
  int Q = dists.size(2);

  // -- invalid constant --
  float invalid = __int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }

  // -- search region offsets --
  // int psHalf = (ps)/2;
  int wsHalf_h = (ws_h-1)/2;
  int wsHalf_w = (ws_w-1)/2;
  // int adj = use_adj ? psHalf : 0;
  int wsOff_h,wsOff_w;
  // int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  // int wsMax_w = stride1*(ws_w-1-wsHalf_w);


  // -- cuda index --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int q_start = blockIdx.x*q_per_thread;
  int qi,ws_i,ws_j;

  // decls
  int ref_patch[2];
  int prop_patch[2];
  int frame_anchor[2];
  int ref_pix[2];
  int prop_pix[2];
  bool valid;
  bool valid_ref_patch,valid_prop_patch;
  bool valid_ref[3];
  bool valid_prop[3];

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
    get_pixel_loc_2d(ref_patch,qindex,qindex_tmp,stride0,nW0,H,W);
    check_bounds_2d(valid_ref_patch,ref_patch,H,W);

    // -- assign to reference --
    frame_anchor[0] = ref_patch[0];
    frame_anchor[1] = ref_patch[1];

    // -- compute frame offsets with flow --
    update_centers_flow(frame_anchor[0],frame_anchor[1],H,W,flow[ibatch]);

    // -- search region offsets --
    set_search_offsets(wsOff_h, wsOff_w, frame_anchor[0], frame_anchor[1], stride1,
                       wsHalf_h, wsHalf_w, ws_h, ws_w, H, W, full_ws_time);

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
        prop_patch[0] = frame_anchor[0] + stride1 * (ws_i - wsOff_h);
        prop_patch[1] = frame_anchor[1] + stride1 * (ws_j - wsOff_w);
        check_bounds_2d(valid_prop_patch,prop_patch,H,W);
        valid = valid_ref_patch && valid_prop_patch;

        // -- init dist --
        dist = 0;

        //  -- compute patch difference --
        if (valid){

          compute_dist_2d<scalar_t,DIST_TYPE>(dist,
                       frame0[ibatch][ihead],frame1[ibatch][ihead],
                       ref_patch, prop_patch, 
                       ref_pix, prop_pix, valid_ref, valid_prop,
                       ps,dilation,reflect_bounds,
                       patch_offset,center_offsets,invalid,
                       C,H,W,pix0,pix1,_dist);

        }

        // -- assignent --
        if (!valid){ dist = invalid; }
        dists[ibatch][ihead][qi][ws_i][ws_j] = dist;
        inds[ibatch][ihead][qi][ws_i][ws_j][0] = prop_patch[0];
        inds[ibatch][ihead][qi][ws_i][ws_j][1] = prop_patch[1];
          
      }
    }
  }
}

void paired_search_forward_cuda(
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow, torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int dist_type,
    int stride0, int stride1, int dilation, int q_shift,
    bool reflect_bounds, bool full_ws, bool full_ws_time,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1){

   // -- derived quantities --
   int B = frame0.size(0);
   int HD = frame0.size(1);
   int H = frame0.size(3);
   int W = frame0.size(4);
   int nH0 = (H-1)/stride0+1;
   int nW0 = (W-1)/stride0+1;

   // -- threads --
   int nheads = dists.size(1);
   int nqueries = dists.size(2);
   int ws_h = dists.size(3);
   int ws_w = dists.size(4);
   int ws_h_threads = std::min(ws_h,27);
   int ws_w_threads = std::min(ws_w,27);
   int ws_h_per_thread = ((ws_h-1)/ws_h_threads) + 1;
   int ws_w_per_thread = ((ws_w-1)/ws_w_threads) + 1;
   dim3 nthreads(ws_h_threads,ws_w_threads);

   // -- nblocks --
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
       AT_DISPATCH_FLOATING_TYPES(frame0.type(),"paired_search_forward_kernel", ([&] {
       paired_search_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
            frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            flow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
            ws_h, ws_w, ps, stride0, stride1, dilation, 
            q_shift, nH0, nW0, reflect_bounds, full_ws, full_ws_time,
            patch_offset, off_H0, off_W0, off_H1, off_W1,
            q_per_thread, ws_h_per_thread, ws_w_per_thread);
          }));
   }else if(dist_type == 1){
       AT_DISPATCH_FLOATING_TYPES(frame0.type(),"paired_search_forward_kernel", ([&] {
       paired_search_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
            frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            flow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
            ws_h, ws_w, ps, stride0, stride1, dilation, 
            q_shift, nH0, nW0, reflect_bounds, full_ws, full_ws_time,
            patch_offset, off_H0, off_W0, off_H1, off_W1,
            q_per_thread, ws_h_per_thread, ws_w_per_thread);
          }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}


/**********************************

      Forward Pass (Bilin2d)

**********************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void paired_search_bilin2d_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame0,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> flow,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> inds,
    int ws_h, int ws_w, int ps,
    int stride0, float _stride1, int dilation,
    int q_shift, int nH0, int nW0,
    bool reflect_bounds, bool full_ws, bool full_ws_time,
    int patch_offset, int off_H0, int off_W0, int off_H1, int off_W1,
    int q_per_thread, int ws_h_per_thread, int ws_w_per_thread){

  // -- unpack shape --
  int B = frame0.size(0);
  int HD = frame0.size(1);
  int C = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);
  int Q = dists.size(2);
  scalar_t stride1 = static_cast<scalar_t>(_stride1);

  // -- invalid constant --
  float _invalid = __int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    _invalid = -_invalid;
  }
  scalar_t invalid = static_cast<scalar_t>(_invalid);

  // -- search region offsets --
  // int psHalf = (ps)/2;
  // int wsHalf_h = (ws_h)/2;
  // int wsHalf_w = (ws_w)/2;
  // int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  // int wsMax_w = stride1*(ws_w-1-wsHalf_w);
  // int adj = use_adj ? psHalf : 0;

  // int wsHalf_h = (ws_h-1)/2;
  // int wsHalf_w = (ws_w-1)/2;
  // int wsOff_h,wsOff_w;

  scalar_t wsHalf_h = trunc((ws_h)/2);
  scalar_t wsHalf_w = trunc((ws_w)/2);
  scalar_t wsOff_h,wsOff_w;

  // -- cuda index --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int q_start = blockIdx.x*q_per_thread;
  int qi,ws_i,ws_j;

  // decls
  int ref_patch[2];
  scalar_t prop_patch[2];
  scalar_t frame_anchor[2];
  int ref_pix[2];
  scalar_t prop_pix[2];
  int prop_i[2];
  bool valid;
  bool valid_ref_patch,valid_prop_patch;
  bool valid_ref[3];
  bool valid_prop[3];

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
    get_pixel_loc_2d(ref_patch,qindex,qindex_tmp,stride0,nW0,H,W);
    check_bounds_2d(valid_ref_patch,ref_patch,H,W);

    // -- assign to reference --
    frame_anchor[0] = __int2float_rn(ref_patch[0]);
    frame_anchor[1] = __int2float_rn(ref_patch[1]);

    // -- compute frame offsets with flow --
    update_centers_flow(frame_anchor[0],frame_anchor[1],H,W,flow[ibatch]);


    // -- search region offsets --
    set_search_offsets<scalar_t>(wsOff_h, wsOff_w,
                                 frame_anchor[0], frame_anchor[1],
                                 stride1, wsHalf_h, wsHalf_w,
                                 ws_h, ws_w, H, W, full_ws_time);

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
        prop_patch[0] = frame_anchor[0] + stride1 * (ws_i - wsOff_h);
        prop_patch[1] = frame_anchor[1] + stride1 * (ws_j - wsOff_w);
        check_bounds_2d(valid_prop_patch,prop_patch,H,W);
        valid = valid_ref_patch && valid_prop_patch;

        // -- init dist --
        dist = 0;

        //  -- compute patch difference --
        if (valid){
          compute_dist_bilin2d_2d<scalar_t,DIST_TYPE>(dist,
                       frame0[ibatch][ihead],
                       frame1[ibatch][ihead],
                       ref_patch, prop_patch,
                       ref_pix, prop_pix, prop_i,
                       valid_ref, valid_prop,
                       ps,dilation,reflect_bounds,
                       patch_offset,center_offsets,invalid,
                       C,H,W,pix0,pix1,_dist);
        }

        // -- assignent --
        if (!valid){ dist = invalid; }
        dists[ibatch][ihead][qi][ws_i][ws_j] = dist;
        inds[ibatch][ihead][qi][ws_i][ws_j][0] = prop_patch[0];
        inds[ibatch][ihead][qi][ws_i][ws_j][1] = prop_patch[1];
          
      }
    }
  }
}

void paired_search_bilin2d_forward_cuda(
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow, torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int dist_type,
    int stride0, float stride1, int dilation, int q_shift,
    bool reflect_bounds, bool full_ws, bool full_ws_time,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1){

   // -- derived quantities --
   int B = frame0.size(0);
   int HD = frame0.size(1);
   int H = frame0.size(3);
   int W = frame0.size(4);
   int nH0 = (H-1)/stride0+1;
   int nW0 = (W-1)/stride0+1;

   // -- threads --
   int nheads = dists.size(1);
   int nqueries = dists.size(2);
   int ws_h = dists.size(3);
   int ws_w = dists.size(4);
   int ws_h_threads = std::min(ws_h,27);
   int ws_w_threads = std::min(ws_w,27);
   int ws_h_per_thread = ((ws_h-1)/ws_h_threads) + 1;
   int ws_w_per_thread = ((ws_w-1)/ws_w_threads) + 1;
   dim3 nthreads(ws_h_threads,ws_w_threads);

   // -- nblocks --
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
       AT_DISPATCH_FLOATING_TYPES(frame0.type(),
                                  "paired_search_bilin2d_forward_kernel", ([&] {
       paired_search_bilin2d_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
            frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            flow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            ws_h, ws_w, ps, stride0, stride1, dilation, 
            q_shift, nH0, nW0, reflect_bounds, full_ws, full_ws_time,
            patch_offset, off_H0, off_W0, off_H1, off_W1,
            q_per_thread, ws_h_per_thread, ws_w_per_thread);
          }));
   }else if(dist_type == 1){
       AT_DISPATCH_FLOATING_TYPES(frame0.type(),
                                  "paired_search_bilin2d_forward_kernel", ([&] {
       paired_search_bilin2d_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
            frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            flow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            ws_h, ws_w, ps, stride0, stride1, dilation, 
            q_shift, nH0, nW0, reflect_bounds, full_ws, full_ws_time,
            patch_offset, off_H0, off_W0, off_H1, off_W1,
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
__global__ void paired_search_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_frame0,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_frame1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame0,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int q_shift, int stride0, int nH0, int nW0,
    int off_H0, int off_W0, int off_H1, int off_W1,
    int ps, int pt, int dilation, int patch_offset,
    bool reflect_bounds, int ftrs_per_thread) {

  // -- shape --
  int nbatch = grad_dists.size(0);
  int Q = grad_dists.size(2);
  int K =  grad_dists.size(3);
  int HD = frame0.size(1);
  int F = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);

  // -- fwd decl registers --
  int ref_patch[2];
  int prop_patch[2];
  int ref[2];
  int prop[2];
  bool valid_ref[3];
  bool valid_prop[3];
  int qindex,qindex_tmp;
  bool valid;
  scalar_t weight,pix0,pix1,pix;
  int iftr;
  int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};

  // -- location to fill --
  int i0 = blockIdx.x*blockDim.x+threadIdx.x;
  int i1 = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/nbatch;
  int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;

  // -- feature chunk --
  int ftr_start = threadIdx.z * ftrs_per_thread;
  int ftr_end = min(F,ftr_start + ftrs_per_thread);

  // -- each region --
  if ((i0 < Q) && (i1 < K)){

    // -- full-resolution video query index --
    qindex = i0 + q_shift;

    // -- pixel location from query index --
    get_pixel_loc_2d(ref_patch,qindex,qindex_tmp,stride0,nW0,H,W);

    // -- proposed location --
    prop_patch[0] = inds[ibatch][ihead][i0][i1][0];
    prop_patch[1] = inds[ibatch][ihead][i0][i1][1];
    prop_patch[2] = inds[ibatch][ihead][i0][i1][2];
    weight = grad_dists[ibatch][ihead][i0][i1];

    // -- update patch --
    update_bwd_patch_2d<scalar_t,DIST_TYPE>(
                     grad_frame0[ibatch][ihead],grad_frame1[ibatch][ihead],
                     frame0[ibatch][ihead],frame1[ibatch][ihead],
                     weight,ref_patch,prop_patch,
                     ps,pt,dilation,reflect_bounds,
                     center_offsets,patch_offset,
                     iftr,ftr_start,ftr_end,
                     ref,prop,valid_ref,valid_prop,valid,
                     H,W,pix0,pix1,pix,i1);

  }
}

void paired_search_backward_cuda(
    torch::Tensor grad_frame0, torch::Tensor grad_frame1,
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds, bool use_adj,
    int off_H0, int off_W0, int off_H1, int off_W1, int dist_type) {

  // -- unpack --
  int B = frame0.size(0);
  int HD = frame0.size(1);
  int F = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);
  int nqueries = inds.size(2);
  int K = inds.size(3);
  int BHD = B*HD;
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

  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(frame0.type(),"paired_search_backward_kernel", ([&] {
    paired_search_backward_kernel<scalar_t,0><<<blocksPerGrid, threadsPerBlock>>>(
          grad_frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);
    }));
  }else if (dist_type == 1){ // l2
    AT_DISPATCH_FLOATING_TYPES(frame0.type(),"paired_search_backward_kernel", ([&] {
    paired_search_backward_kernel<scalar_t,1><<<blocksPerGrid, threadsPerBlock>>>(
          grad_frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);
    }));
  }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");    }


}



/****************************

       Backward Bilinear-2d

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void paired_search_bilin2d_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_frame0,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_frame1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame0,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
    int q_shift, int stride0, int nH0, int nW0,
    int off_H0, int off_W0, int off_H1, int off_W1,
    int ps, int pt, int dilation, int patch_offset,
    bool reflect_bounds, int ftrs_per_thread) {

  // -- shape --
  int nbatch = grad_dists.size(0);
  int Q = grad_dists.size(2);
  int K =  grad_dists.size(3);
  int HD = frame0.size(1);
  int F = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);

  // -- fwd decl registers --
  int ref_patch[2];
  int prop_patch[2];
  int ref[2];
  int prop[2];
  bool valid_ref[3];
  bool valid_prop[3];
  int qindex,qindex_tmp;
  bool valid;
  scalar_t weight,pix0,pix1,pix;
  int iftr;
  int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};

  // -- location to fill --
  int i0 = blockIdx.x*blockDim.x+threadIdx.x;
  int i1 = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/nbatch;
  int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;

  // -- feature chunk --
  int ftr_start = threadIdx.z * ftrs_per_thread;
  int ftr_end = min(F,ftr_start + ftrs_per_thread);

  // -- each region --
  if ((i0 < Q) && (i1 < K)){

    // -- full-resolution video query index --
    qindex = i0 + q_shift;

    // -- pixel location from query index --
    get_pixel_loc_2d(ref_patch,qindex,qindex_tmp,stride0,nW0,H,W);

    // -- proposed location --
    prop_patch[0] = inds[ibatch][ihead][i0][i1][0];
    prop_patch[1] = inds[ibatch][ihead][i0][i1][1];
    prop_patch[2] = inds[ibatch][ihead][i0][i1][2];
    weight = grad_dists[ibatch][ihead][i0][i1];

    // -- update patch --
    update_bwd_patch_2d<scalar_t,DIST_TYPE>(
                     grad_frame0[ibatch][ihead],grad_frame1[ibatch][ihead],
                     frame0[ibatch][ihead],frame1[ibatch][ihead],
                     weight,ref_patch,prop_patch,
                     ps,pt,dilation,reflect_bounds,
                     center_offsets,patch_offset,
                     iftr,ftr_start,ftr_end,
                     ref,prop,valid_ref,valid_prop,valid,
                     H,W,pix0,pix1,pix,i1);

  }
}

void paired_search_bilin2d_backward_cuda(
    torch::Tensor grad_frame0, torch::Tensor grad_frame1,
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds, bool use_adj,
    int off_H0, int off_W0, int off_H1, int off_W1, int dist_type) {

  // -- unpack --
  int B = frame0.size(0);
  int HD = frame0.size(1);
  int F = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);
  int nqueries = inds.size(2);
  int K = inds.size(3);
  int BHD = B*HD;
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

  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(frame0.type(),
                               "paired_search_bilin2d_backward_kernel", ([&] {
    paired_search_bilin2d_backward_kernel<scalar_t,0><<<blocksPerGrid, threadsPerBlock>>>(
          grad_frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);
    }));
  }else if (dist_type == 1){ // l2
    AT_DISPATCH_FLOATING_TYPES(frame0.type(),
                               "paired_search_bilin2d_backward_kernel", ([&] {
    paired_search_bilin2d_backward_kernel<scalar_t,1><<<blocksPerGrid, threadsPerBlock>>>(
          grad_frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);
    }));
  }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");    }


}


