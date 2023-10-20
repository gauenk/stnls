
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include "nls_int.cu"

using namespace at;


/****************************

       Forward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void non_local_search_int_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> flows,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> inds,
    int ws, int wt, int ps, int pt, int stride0, int stride1, int dilation,
    bool reflect_bounds, bool full_ws, int patch_offset,
    int nH0, int nW0, int nHW0, int st_offset,
    int q_per_thread, int ws_per_thread, int wt_per_thread){

  // -- unpack shape --
  int B = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int Q = dists.size(2);
  int HD_f = flows.size(1);

  // -- invalid constant --
  scalar_t invalid = (scalar_t)__int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }

  // -- search window params --
  int wsHalf = (ws-1)/2;
  // int wsMax = stride1*(ws-1-wsHalf);
  int wsOff_h,wsOff_w;
  int W_t = 2*wt+1;
  int t_max;

  // -- cuda index --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int q_start = blockIdx.x*q_per_thread;
  int qi,ws_i,ws_j;
  int ihead_f = ihead % HD_f;

  // -- decls --
  int ref_patch[3];
  int prop_patch[3];
  int frame_anchor[2];
  int ref_pix[3];
  int prop_pix[3];
  bool valid;
  bool valid_ref_patch,valid_prop_patch;
  bool valid_ref[4];
  bool valid_prop[4];
  int n_hi,n_wi;

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
    get_pixel_loc(ref_patch,qi,stride0,nW0,nHW0,H,W);
    n_hi = ref_patch[1] / stride0;
    n_wi = ref_patch[2] / stride0;

    // -- check bounds of pixel location --
    // check_bounds(valid_ref_patch,ref_patch,T,H,W);
    valid_ref_patch = true;

    // -- temporal search bounds --
    set_time_range(t_max, ref_patch[0], T, wt);

    // -- search across time --
    for (int _zi = 0; _zi < wt_per_thread; _zi++){
      int st_i = threadIdx.z + blockDim.z*_zi;
      if (st_i >= W_t){ continue; }


      // ---------------------------------------
      //       compute search center
      // ---------------------------------------

      // -- select time --
      prop_patch[0] = ref_patch[0] + st_i; // t_next
      prop_patch[0] = (prop_patch[0] > t_max) ? t_max - st_i : prop_patch[0];

      // -- offset with flows --
      if (st_i >= st_offset){
        auto flows_t = flows[ibatch][ihead_f][ref_patch[0]][st_i-st_offset];
        frame_anchor[0] = ref_patch[1] + flows_t[1][n_hi][n_wi];
        frame_anchor[1] = ref_patch[2] + flows_t[0][n_hi][n_wi];
        frame_anchor[0] = bounds(frame_anchor[0],H);
        frame_anchor[1] = bounds(frame_anchor[1],W);
      }else{
        frame_anchor[0] = ref_patch[1];
        frame_anchor[1] = ref_patch[2];
      }

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
          prop_patch[1] = frame_anchor[0] + stride1 * (ws_i - wsOff_h);
          prop_patch[2] = frame_anchor[1] + stride1 * (ws_j - wsOff_w);
          check_bounds(valid_prop_patch,prop_patch,T,H,W);
          valid = valid_ref_patch && valid_prop_patch;

          // -- init dist --
          dist = 0;

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
          dists[ibatch][ihead][qi][st_i][ws_i][ws_j] = dist;
          inds[ibatch][ihead][qi][st_i][ws_i][ws_j][0] = prop_patch[0] - ref_patch[0];
          inds[ibatch][ihead][qi][st_i][ws_i][ws_j][1] = prop_patch[1] - ref_patch[1];
          inds[ibatch][ihead][qi][st_i][ws_i][ws_j][2] = prop_patch[2] - ref_patch[2];
          
        }
      }
    }
  }
}

void non_local_search_int_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor flows,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, int stride1, int dilation, int pt,
    bool reflect_bounds, bool full_ws, int patch_offset, int dist_type){

   // -- derived quantities --
   int H = vid0.size(4);
   int W = vid0.size(5);
   int nH = (H-1)/stride0+1;
   int nW = (W-1)/stride0+1;
   int nHW = nH * nW;

   // -- threads --
   int nheads = dists.size(1);
   int nqueries = dists.size(2);

   int ws = dists.size(4);
   int ws_threads = std::min(ws,15);
   int ws_per_thread = ((ws-1)/ws_threads) + 1;

   int W_t = dists.size(3);
   int wt = (W_t-1)/2;
   int wt_threads = std::min(W_t,3);
   int wt_per_thread = ((W_t-1)/wt_threads) + 1;

   dim3 nthreads(ws_threads,ws_threads,wt_threads);

   // -- nblocks --
   int B = vid0.size(0);
   int HD = vid0.size(1);
   int q_per_thread = 2;
   int nquery_blocks = ((nqueries - 1) / q_per_thread) + 1;
   dim3 nblocks(nquery_blocks,B,HD);

   // -- share --
   // int psHalf = ps/2;
   // int adj = use_adj ? psHalf : 0;
   // int patch_offset = adj - psHalf;
   int st_offset = W_t - flows.size(3);
   assert(st_offset <= 1);

   // launch kernel
   if (dist_type == 0){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                                  "non_local_search_int_forward_kernel", ([&] {
       non_local_search_int_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            flows.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
            ws, wt, ps, pt, stride0, stride1, dilation, 
            reflect_bounds, full_ws, patch_offset, nH, nW, nHW,
            st_offset, q_per_thread, ws_per_thread, wt_per_thread);
          }));
   }else if(dist_type == 1){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                                  "non_local_search_int_forward_kernel", ([&] {
       non_local_search_int_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            flows.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
            ws, wt, ps, pt, stride0, stride1, dilation, 
            reflect_bounds, full_ws, patch_offset, nH, nW, nHW,
            st_offset, q_per_thread, ws_per_thread, wt_per_thread);
          }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}



/****************************

       Backward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void non_local_search_int_vid_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> inds,
    int ps, int pt, int stride0, int dilation, bool reflect_bounds,
    int patch_offset, int ftrs_per_thread) {

  // -- shape --
  int nbatch = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int nH0 = inds.size(3);
  int nW0 = inds.size(4);
  int nHW0 = nH0*nW0;
  int K =  inds.size(5);
  int Q = T*nH0*nW0;

  // -- fwd decl registers --
  int ref_patch[3];
  int prop_patch[3];
  int ref[3];
  int prop[3];
  bool valid_ref[4];
  bool valid_prop[4];
  bool valid;
  scalar_t dist,weight;
  int iftr;

  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/nbatch;
  int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;

  // -- feature chunk --
  int ftr_start = threadIdx.z * ftrs_per_thread;
  int ftr_end = min(F,ftr_start + ftrs_per_thread);

  // -- each region --
  if ((qi < Q) && (ki < K)){

    // -- pixel location from query index --
    get_pixel_loc(ref_patch,qi,stride0,nW0,nHW0,H,W);
    int ti = ref_patch[0];
    int nh = ref_patch[1]/stride0;
    int nw = ref_patch[2]/stride0;

    // -- proposed location --
    prop_patch[0] = ref_patch[0] + inds[ibatch][ihead][ti][nh][nw][ki][0];
    prop_patch[1] = ref_patch[1] + inds[ibatch][ihead][ti][nh][nw][ki][1];
    prop_patch[2] = ref_patch[2] + inds[ibatch][ihead][ti][nh][nw][ki][2];
    weight = grad_dists[ibatch][ihead][ti][nh][nw][ki];

    // -- update patch --
    update_bwd_patch_int<scalar_t,DIST_TYPE>(
                     grad_vid0[ibatch][ihead],grad_vid1[ibatch][ihead],
                     vid0[ibatch][ihead],vid1[ibatch][ihead],
                     weight,ref_patch,prop_patch,
                     ps,pt,dilation,reflect_bounds,
                     patch_offset,iftr,ftr_start,ftr_end,
                     ref,prop,valid_ref,valid_prop,valid,T,H,W);

  }
}

void non_local_search_int_vid_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset, int dist_type) {

  // -- unpack --
  int B = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int BHD = B*HD;

  // -- num --
  int nH = inds.size(3);
  int nW = inds.size(4);
  int Q = T*nH*nW;
  assert(pt == 1);
  int K = inds.size(5);

  // -- launch parameters --
  int ftr_threads = min(1,F);
  int ftrs_per_thread = (F-1)/ftr_threads+1;
  dim3 threadsPerBlock(10,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, B*HD);
  blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(K)/double(threadsPerBlock.y));

  // -- shared --
  // int psHalf = ps/2;
  // int adj = use_adj ? psHalf : 0;
  // int patch_offset = adj - psHalf;

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
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"non_local_search_int_backward_kernel", ([&] {
    non_local_search_int_vid_backward_kernel<scalar_t,0>\
      <<<blocksPerGrid, threadsPerBlock>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
          ps, pt, stride0, dilation, reflect_bounds, patch_offset,
          ftrs_per_thread);
    }));
  }else if (dist_type == 1){ // l2
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"non_local_search_int_backward_kernel", ([&] {
    non_local_search_int_vid_backward_kernel<scalar_t,1>\
      <<<blocksPerGrid, threadsPerBlock>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
          ps, pt, stride0, dilation, reflect_bounds, patch_offset,
          ftrs_per_thread);
    }));
  }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");    }

}


