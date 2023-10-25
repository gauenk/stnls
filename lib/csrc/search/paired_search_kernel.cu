
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
__global__ void paired_search_int_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame0,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> flow,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> inds,
    int ws, int ps, int stride0, int stride1, int dilation,
    bool reflect_bounds, bool full_ws, int patch_offset,
    int q_per_thread, int ws_per_thread){

  // -- unpack shape --
  int B = frame0.size(0);
  int HD_frame = frame0.size(1);
  int HD_flow = flow.size(1);
  int HD_search = inds.size(1);
  int C = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);
  int Q = dists.size(2);
  int HD = max(HD_frame,HD_flow);

  // -- invalid constant --
  scalar_t invalid = (scalar_t)__int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }


  // -- search region offsets --
  // int psHalf = (ps)/2;
  int wsHalf = (ws-1)/2;
  // int wsHalf_w = (ws_w)/2;
  // int adj = use_adj ? psHalf : 0;
  int wsOff_h,wsOff_w;
  // int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  // int wsMax_w = stride1*(ws_w-1-wsHalf_w);

  // -- cuda index --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int ihead_fr = ihead % HD_frame;
  int ihead_fl = ihead % HD_flow;
  int ihead_sr = ihead % HD_search;
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

  // -- indexing --
  scalar_t dist;

  for (int q_index = 0; q_index < q_per_thread; q_index++){


    //---------------------------
    //       Anchor Pixel
    //---------------------------

    // -- block start --
    qi = q_start + q_index;
    if (qi >= Q){ continue; }

    // -- pixel location from query index --
    get_pixel_loc_2d(ref_patch,qi,stride0,H,W);
    int nh = ref_patch[0]/stride0;
    int nw = ref_patch[1]/stride0;
    check_bounds_2d(valid_ref_patch,ref_patch,H,W);

    // -- assign to reference --
    frame_anchor[0] = ref_patch[0] + flow[ibatch][ihead_fl][1][nh][nw];
    frame_anchor[1] = ref_patch[1] + flow[ibatch][ihead_fl][0][nh][nw];
    frame_anchor[0] = bounds(frame_anchor[0],H);
    frame_anchor[1] = bounds(frame_anchor[1],W);

    // -- search region offsets --
    set_search_offsets(wsOff_h, wsOff_w,
                       frame_anchor[0], frame_anchor[1], stride1,
                       wsHalf, ws, H, W, full_ws);

    // ---------------------------------------
    //          spatial searching
    // ---------------------------------------

    // -- search across space --
    for (int _xi = 0; _xi < ws_per_thread; _xi++){
      ws_i = threadIdx.x + blockDim.x*_xi;
      if (ws_i >= ws){ continue; }
      for (int _yi = 0; _yi < ws_per_thread; _yi++){
        ws_j = threadIdx.y + blockDim.y*_yi;
        if (ws_j >= ws){ continue; }

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
                       frame0[ibatch][ihead_fr],frame1[ibatch][ihead_fr],
                       ref_patch, prop_patch, 
                       ref_pix, prop_pix, valid_ref, valid_prop,
                       ps,dilation,reflect_bounds,
                       patch_offset,invalid,C,H,W);

        }

        // -- assignent --
        if (!valid){ dist = invalid; }
        dists[ibatch][ihead_sr][qi][ws_i][ws_j] = dist;
        inds[ibatch][ihead_sr][qi][ws_i][ws_j][0] = prop_patch[0]-ref_patch[0];
        inds[ibatch][ihead_sr][qi][ws_i][ws_j][1] = prop_patch[1]-ref_patch[1];
          
      }
    }
  }
}

void paired_search_int_forward_cuda(
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow, torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, int stride1, int dilation,
    bool reflect_bounds, bool full_ws, int patch_offset, int dist_type){

   // -- derived quantities --
   int B = frame0.size(0);
   int HD_frame = frame0.size(1);
   int HD_flow = flow.size(1);
   int H = frame0.size(3);
   int W = frame0.size(4);
   // int nH0 = (H-1)/stride0+1;
   int HD = max(HD_frame,HD_flow);

   // -- threads --
   int nqueries = dists.size(2);
   int ws = dists.size(3);
   int ws_threads = std::min(ws,25);
   int ws_per_thread = ((ws-1)/ws_threads) + 1;
   dim3 nthreads(ws_threads,ws_threads);

   // -- nblocks --
   int q_per_thread = 2;
   int nquery_blocks = ((nqueries - 1) / q_per_thread) + 1;
   dim3 nblocks(nquery_blocks,B,HD);

   // -- share --
   // int psHalf = ps/2;
   // int adj = use_adj ? psHalf : 0;
   // // int patch_offset = adj - psHalf;
   // int patch_offset = adj - psHalf;

   // -- viz --
   // fprintf(stdout,"ws_h,ws_w: %d,%d,%d,%d\n",ws_h,ws_w,ws_h_threads,ws_h_per_thread);
   // fprintf(stdout,"nquery_blocks,B,HD: %d,%d,%d\n",nquery_blocks,B,HD);


   // launch kernel
   if (dist_type == 0){
       AT_DISPATCH_FLOATING_TYPES(frame0.type(),"paired_search_int_forward_kernel", ([&] {
       paired_search_int_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
            frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            flow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
            ws, ps, stride0, stride1, dilation, reflect_bounds, full_ws,
            patch_offset, q_per_thread, ws_per_thread);
          }));
   }else if(dist_type == 1){
       AT_DISPATCH_FLOATING_TYPES(frame0.type(),"paired_search_int_forward_kernel", ([&] {
       paired_search_int_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
            frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            flow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
            ws, ps, stride0, stride1, dilation, reflect_bounds, full_ws,
            patch_offset, q_per_thread, ws_per_thread);
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
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> flow,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> inds,
    int ws, int ps, int stride0, float _stride1, int dilation,
    bool reflect_bounds, bool full_ws, int patch_offset,
    int q_per_thread, int ws_per_thread){

  // -- unpack shape --
  int B = frame0.size(0);
  int HD_frame = frame0.size(1);
  int HD_flow = flow.size(1);
  int HD_search = inds.size(1);
  int C = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);
  int Q = dists.size(2);
  int HD = max(HD_frame,HD_flow);
  scalar_t stride1 = static_cast<scalar_t>(_stride1);

  // -- invalid constant --
  scalar_t invalid = (scalar_t)__int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }

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
  scalar_t wsHalf = trunc((ws-1)/2);
  scalar_t wsOff_h,wsOff_w;

  // -- cuda index --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int ihead_fr = ihead % HD_frame;
  int ihead_fl = ihead % HD_flow;
  int ihead_sr = ihead % HD_search;
  int q_start = blockIdx.x*q_per_thread;
  int qi,ws_i,ws_j;

  // decls
  int ref_patch[2];
  scalar_t prop_patch[2];
  scalar_t frame_anchor[2];
  int ref_pix[2];
  scalar_t prop_pix[2];
  // int prop_i[2];
  bool valid;
  bool valid_ref_patch,valid_prop_patch;
  bool valid_ref[3];
  bool valid_prop[3];

  // -- indexing --
  scalar_t dist,pix0,pix1;

  for (int q_index = 0; q_index < q_per_thread; q_index++){


    //---------------------------
    //       Anchor Pixel
    //---------------------------

    // -- block start --
    qi = q_start + q_index;
    if (qi >= Q){ continue; }

    // -- pixel location from query index --
    get_pixel_loc_2d(ref_patch,qi,stride0,H,W);
    check_bounds_2d(valid_ref_patch,ref_patch,H,W);
    int nh = ref_patch[0]/stride0;
    int nw = ref_patch[1]/stride0;

    // -- compute frame offsets with flow --
    frame_anchor[0] = ref_patch[0]+flow[ibatch][ihead_fl][1][nh][nw];
    frame_anchor[1] = ref_patch[1]+flow[ibatch][ihead_fl][0][nh][nw];
    frame_anchor[0] = bounds(frame_anchor[0],H);
    frame_anchor[1] = bounds(frame_anchor[1],W);

    // -- search region offsets --
    set_search_offsets(wsOff_h, wsOff_w,
                       frame_anchor[0], frame_anchor[1], stride1,
                       wsHalf, ws, H, W, full_ws);

    // ---------------------------------------
    //          spatial searching
    // ---------------------------------------

    // -- search across space --
    for (int _xi = 0; _xi < ws_per_thread; _xi++){
      ws_i = threadIdx.x + blockDim.x*_xi;
      if (ws_i >= ws){ continue; }
      for (int _yi = 0; _yi < ws_per_thread; _yi++){
        ws_j = threadIdx.y + blockDim.y*_yi;
        if (ws_j >= ws){ continue; }

        // -- compute proposed location --
        prop_patch[0] = frame_anchor[0] + stride1 * (ws_i - wsOff_h);
        prop_patch[1] = frame_anchor[1] + stride1 * (ws_j - wsOff_w);
        check_bounds_2d<scalar_t>(valid_prop_patch,prop_patch,H,W);
        valid = valid_ref_patch && valid_prop_patch;


        // -- init dist --
        dist = 0;
        // Z = 0;

        //  -- compute patch difference --
        if (valid){
          compute_dist_bilin2d_2d<scalar_t,DIST_TYPE>(dist,
                       frame0[ibatch][ihead_fr],frame1[ibatch][ihead_fr],
                       ref_patch, prop_patch, ref_pix, prop_pix,// prop_i,
                       valid_ref, valid_prop, ps,dilation,reflect_bounds,
                       patch_offset,invalid,C,H,W);
          // dist /= Z;
        }


        // -- assignent --
        if (!valid){ dist = invalid; }
        dists[ibatch][ihead_sr][qi][ws_i][ws_j] = dist;
        inds[ibatch][ihead_sr][qi][ws_i][ws_j][0] = prop_patch[0]-ref_patch[0];
        inds[ibatch][ihead_sr][qi][ws_i][ws_j][1] = prop_patch[1]-ref_patch[1];
        // inds[ibatch][ihead_fl][qi][ws_i][ws_j][0] = frame_anchor[0];
        // inds[ibatch][ihead_fl][qi][ws_i][ws_j][1] = frame_anchor[1];
          
      }
    }
  }
}

void paired_search_bilin2d_forward_cuda(
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow, torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, float stride1, int dilation,
    bool reflect_bounds, bool full_ws, int patch_offset, int dist_type){

   // -- derived quantities --
   int B = frame0.size(0);
   int HD_frame = frame0.size(1);
   int HD_flow = flow.size(1);
   int H = frame0.size(3);
   int W = frame0.size(4);
   // int nH0 = (H-1)/stride0+1;
   int HD = max(HD_frame,HD_flow);

   // -- threads --
   int nqueries = dists.size(2);
   int ws = dists.size(3);
   int ws_threads = std::min(ws,25);
   int ws_per_thread = ((ws-1)/ws_threads) + 1;
   dim3 nthreads(ws_threads,ws_threads);

   // -- nblocks --
   int q_per_thread = 2;
   int nquery_blocks = ((nqueries - 1) / q_per_thread) + 1;
   dim3 nblocks(nquery_blocks,B,HD);

   // -- share --
   // int psHalf = ps/2;
   // int adj = use_adj ? psHalf : 0;
   // // int patch_offset = adj - psHalf;
   // int patch_offset = adj - psHalf;

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
            flow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            ws, ps, stride0, stride1, dilation, reflect_bounds, full_ws,
            patch_offset, q_per_thread, ws_per_thread);
          }));
   }else if(dist_type == 1){
       AT_DISPATCH_FLOATING_TYPES(frame0.type(),
                                  "paired_search_bilin2d_forward_kernel", ([&] {
       paired_search_bilin2d_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
            frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            flow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            ws, ps, stride0, stride1, dilation, reflect_bounds, full_ws,
            patch_offset, q_per_thread, ws_per_thread);
          }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void paired_search_int_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_frame0,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_frame1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame0,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int stride0, int ps, int dilation, int patch_offset,
    bool reflect_bounds, int ftrs_per_thread) {

  // -- shape --
  int nbatch = grad_dists.size(0);
  int Q = grad_dists.size(2);
  int K =  grad_dists.size(3);
  int HD_frame = frame0.size(1);
  int HD_flow = grad_dists.size(1);
  int F = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);
  int HD = max(HD_frame,HD_flow);

  // -- fwd decl registers --
  int ref_patch[2];
  int prop_patch[2];
  int ref[2];
  int prop[2];
  bool valid_ref[3];
  bool valid_prop[3];
  bool valid;
  scalar_t weight,pix0,pix1,pix;
  // int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};


  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/nbatch;
  int ihead_fr = ihead % HD_frame;
  int ihead_fl = ihead % HD_flow;
  int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;

  // -- feature chunk --
  int ftr_start = threadIdx.z * ftrs_per_thread;
  int ftr_end = min(F,ftr_start + ftrs_per_thread);

  // -- each region --
  if ((qi < Q) && (ki < K)){

    // -- pixel location from query index --
    get_pixel_loc_2d(ref_patch,qi,stride0,H,W);

    // -- proposed location --
    prop_patch[0] = ref_patch[0] + inds[ibatch][ihead_fl][qi][ki][0];
    prop_patch[1] = ref_patch[1] + inds[ibatch][ihead_fl][qi][ki][1];
    weight = grad_dists[ibatch][ihead_fl][qi][ki];

    // -- update patch --
    update_bwd_patch_2d<scalar_t,DIST_TYPE>(
                     grad_frame0[ibatch][ihead_fr],
                     grad_frame1[ibatch][ihead_fr],
                     frame0[ibatch][ihead_fr],
                     frame1[ibatch][ihead_fr],
                     weight,ref_patch,prop_patch,
                     ps,dilation,reflect_bounds,
                     patch_offset,ftr_start,ftr_end,
                     ref,prop,valid_ref,valid_prop,valid,
                     H,W,pix0,pix1);

  }
}

void paired_search_int_backward_cuda(
    torch::Tensor grad_frame0, torch::Tensor grad_frame1,
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int stride0, int ps, int dilation, bool reflect_bounds,
    int patch_offset, int dist_type) {


  // -- unpack --
  int B = frame0.size(0);
  int HD_frame = frame0.size(1);
  int HD_flow = grad_dists.size(1);
  int F = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);
  int HD = max(HD_frame,HD_flow);
  int nqueries = inds.size(2);
  int K = inds.size(3);
  int BHD = B*HD;

  // -- launch parameters --
  int nbatch = grad_dists.size(0);
  int nq = grad_dists.size(2);
  int k = grad_dists.size(3);
  int ftr_threads = min(1,F);
  dim3 threadsPerBlock(128,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, nbatch*HD);
  blocksPerGrid.x = ceil(double(nq)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
  int ftrs_per_thread = (F-1)/ftr_threads+1;

  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(frame0.type(),"paired_search_backward_kernel", ([&] {
    paired_search_int_backward_kernel<scalar_t,0><<<blocksPerGrid, threadsPerBlock>>>(
          grad_frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          stride0, ps, dilation, patch_offset, reflect_bounds,
          ftrs_per_thread);
    }));
  }else if (dist_type == 1){ // l2
    AT_DISPATCH_FLOATING_TYPES(frame0.type(),"paired_search_backward_kernel", ([&] {
    paired_search_int_backward_kernel<scalar_t,1><<<blocksPerGrid, threadsPerBlock>>>(
          grad_frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          stride0, ps, dilation, patch_offset, reflect_bounds,
          ftrs_per_thread);
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
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_flow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame0,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> frame1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> flow,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_inds,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
    int stride0, int ps, int dilation, int patch_offset, bool reflect_bounds) {

  // -- shape --
  int nbatch = grad_dists.size(0);
  int Q = grad_dists.size(2);
  int K =  grad_dists.size(3);
  int HD_frame = frame0.size(1);
  int HD_flow = grad_flow.size(1);
  int HD_search = inds.size(1);
  int F = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);
  int HD = max(HD_frame,HD_flow);

  // -- fwd decl registers --
  int ref_patch[2];
  scalar_t prop_patch[2];
  // int ref[2];
  // scalar_t prop[2];
  // int prop_i[2];
  bool valid_ref[3];
  bool valid_prop[3];
  bool valid;
  scalar_t weight;
  scalar_t iweight[2];
  // int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};

  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/nbatch;
  int ihead_fr = ihead % HD_frame;
  int ihead_fl = ihead % HD_flow;
  int ihead_sr = ihead % HD_search;
  int ibatch = (blockIdx.z-ihead*nbatch);

  // -- feature chunk --
  // int ftr_start = 0;//threadIdx.z * ftrs_per_thread;
  // int ftr_end = F;//min(F,ftr_start + ftrs_per_thread);

  // -- each region --
  if ((qi < Q) && (ki < K)){

    // -- pixel location from query index --
    get_pixel_loc_2d(ref_patch,qi,stride0,H,W);
    int nh = ref_patch[0]/stride0;
    int nw = ref_patch[1]/stride0;

    // -- accumulate optical flow update --
    scalar_t acc_dFlows[8];
  #pragma unroll
    for (int _idx=0; _idx < 8; _idx++){
      acc_dFlows[_idx] = static_cast<scalar_t>(0);
    }

    // -- proposed location --
    prop_patch[0] = ref_patch[0] + inds[ibatch][ihead_sr][qi][ki][0];
    prop_patch[1] = ref_patch[1] + inds[ibatch][ihead_sr][qi][ki][1];


    weight = grad_dists[ibatch][ihead_sr][qi][ki];
    iweight[0] = grad_inds[ibatch][ihead_sr][qi][ki][0];
    iweight[1] = grad_inds[ibatch][ihead_sr][qi][ki][1];


    // -- update frames --
    update_bwd_bilin2d_patch_2d<scalar_t,DIST_TYPE>(
                     grad_frame0[ibatch][ihead_fr],grad_frame1[ibatch][ihead_fr],
                     frame0[ibatch][ihead_fr],frame1[ibatch][ihead_fr],
                     acc_dFlows,weight,ref_patch,prop_patch,
                     ps,dilation,reflect_bounds,patch_offset,
                     // ftr_start,ftr_end,ref,prop,prop_i,
                     valid_ref,valid_prop,valid,H,W);


    // -- update grad_flow from grad_dists,vid0,vid1 --
    scalar_t wi = ref_patch[1] + flow[ibatch][ihead_fl][0][nh][nw];
    scalar_t hi = ref_patch[0] + flow[ibatch][ihead_fl][1][nh][nw];
    int signW = ((wi >= 0) and (wi < W)) ? 1 : -1; // untested move from "W-1" to "W"
    int signH = ((hi >= 0) and (hi < H)) ? 1 : -1;
    bwd_flow_assign(acc_dFlows,nh,nw,signH,signW,grad_flow[ibatch][ihead_fl]);

    // -- update flows --
    atomicAdd(&(grad_flow[ibatch][ihead_fl][0][nh][nw]),signW*iweight[1]);
    atomicAdd(&(grad_flow[ibatch][ihead_fl][1][nh][nw]),signH*iweight[0]);

  }
}

void paired_search_bilin2d_backward_cuda(
    torch::Tensor grad_frame0, torch::Tensor grad_frame1,
    torch::Tensor grad_flow,
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow,
    const torch::Tensor grad_dists, const torch::Tensor grad_inds,
    const torch::Tensor inds,
    int stride0, int ps, int dilation, bool reflect_bounds,
    int patch_offset, int dist_type) {

  // -- unpack --
  int HD_frame = frame0.size(1);
  int HD_flow = grad_dists.size(1);
  int F = frame0.size(2);
  int H = frame0.size(3);
  int W = frame0.size(4);
  // int K = inds.size(3);
  // assert(pt == 1);
  int HD = max(HD_frame,HD_flow);

  // -- launch parameters --
  int B = grad_dists.size(0);
  int Q = grad_dists.size(2);
  int K = grad_dists.size(3);
  dim3 threadsPerBlock(288,2);
  dim3 blocksPerGrid(1, 1, B*HD);
  blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(K)/double(threadsPerBlock.y));

  // -- shared --
  // int psHalf = ps/2;
  // int adj = use_adj ? psHalf : 0;
  // int patch_offset = adj - psHalf;
  // int patch_offset = psHalf - adj;

  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(frame0.type(),
                               "paired_search_bilin2d_backward_kernel", ([&] {
    paired_search_bilin2d_backward_kernel<scalar_t,0><<<blocksPerGrid, threadsPerBlock>>>(
          grad_frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_flow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          flow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          grad_inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          stride0, ps, dilation, patch_offset, reflect_bounds);
    }));
  }else if (dist_type == 1){ // l2
    AT_DISPATCH_FLOATING_TYPES(frame0.type(),
                               "paired_search_bilin2d_backward_kernel", ([&] {
    paired_search_bilin2d_backward_kernel<scalar_t,1><<<blocksPerGrid, threadsPerBlock>>>(
          grad_frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_flow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          frame1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          flow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          grad_inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          stride0, ps, dilation, patch_offset, reflect_bounds);
    }));
  }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");    }


}


