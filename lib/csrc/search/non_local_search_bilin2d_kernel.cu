
// #include <torch/extension.h>
#include <cuda/std/type_traits>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include "nls_bilin2d.cu"

using namespace at;


/****************************

       Forward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void non_local_search_bilin2d_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> flows,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> inds,
    // torch::PackedTensorAccessor32<bool,7,torch::RestrictPtrTraits> reflect,
    int ws, int wt, int ps, int pt, int stride0, float _stride1, int dilation,
    bool reflect_bounds, bool full_ws, int patch_offset,
    int nH, int nW, int nHW, int st_offset, int off_Hq, int off_Wq,
    int q_per_thread, int ws_per_thread, int wt_per_thread){

  // -- unpack shape --
  int B = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int C = vid0.size(3);
  int qH = vid0.size(4);
  int qW = vid0.size(5);
  int kH = vid1.size(4);
  int kW = vid1.size(5);
  int Q = dists.size(2);
  int HD_f = flows.size(1);
  scalar_t stride1 = static_cast<scalar_t>(_stride1);

  // -- invalid constant --
  scalar_t invalid = (scalar_t)__int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }

  // -- search window params --
  scalar_t wsHalf = (ws-1)/2;
  scalar_t wsOff_h,wsOff_w;
  int W_t = 2*wt+1;
  int t_max;

  // -- offsets --
  int offs[2];
  offs[0] = off_Hq;
  offs[1] = off_Wq;

  // -- cuda index --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int ihead_f = ihead % HD_f;
  int q_start = blockIdx.x*q_per_thread;
  int qi,ws_i,ws_j;

  // -- decls --
  int ref_patch[3];
  scalar_t prop_patch[3];
  int prop_i[3];
  scalar_t frame_anchor[2];
  int ref_pix[3];
  scalar_t prop_pix[3];
  int prop_pix_i[3];
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
    get_pixel_loc<int>(ref_patch,qi,stride0,nW,nHW,qH,qW);
    n_hi = ref_patch[1] / stride0;
    n_wi = ref_patch[2] / stride0;

    // -- check bounds of pixel location --
    // check_bounds<int>(valid_ref_patch,ref_patch,T,qH,qW);
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
      int t_next = ref_patch[0] + st_i;
      t_next = (t_next > t_max) ? t_max - st_i : t_next;
      prop_patch[0] = t_next;

      // -- offset with flows --
      if (st_i >= st_offset){
        auto flows_t = flows[ibatch][ihead_f][ref_patch[0]][st_i-st_offset];
        frame_anchor[0] = ref_patch[1] + flows_t[1][n_hi][n_wi];
        frame_anchor[1] = ref_patch[2] + flows_t[0][n_hi][n_wi];
        // valid_H = (frame_anchor[0] >= 0) and (frame_anchor[0] <= (H-1));
        // valid_W = (frame_anchor[1] >= 0) and (frame_anchor[1] <= (W-1));
        // reflect[ibatch][ihead_f][ti][si][n_hi][n_wi][1] = not valid_H;
        // reflect[ibatch][ihead_f][ti][si][n_hi][n_w][0] = not valid_W;
        frame_anchor[0] = bounds(frame_anchor[0],kH);
        frame_anchor[1] = bounds(frame_anchor[1],kW);
      }else{
        frame_anchor[0] = 1.*ref_patch[1];
        frame_anchor[1] = 1.*ref_patch[2];
        // reflect[ibatch][ihead_f][ti][si][n_hi][n_wi][1] = 0;
        // reflect[ibatch][ihead_f][ti][si][n_hi][n_wi][0] = 0;
      }

      // -- search region offsets --
      set_search_offsets(wsOff_h, wsOff_w,
                         frame_anchor[0], frame_anchor[1], stride1,
                         wsHalf, ws, kH, kW, full_ws);

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
          check_bounds<scalar_t>(valid_prop_patch,prop_patch,T,kH,kW);
          valid = valid_ref_patch && valid_prop_patch;

          // -- init dist --
          dist = 0;


          //  -- compute patch difference --
          if (valid){
              compute_dist_bilin2d<scalar_t,DIST_TYPE>(dist,
                           vid0[ibatch][ihead],vid1[ibatch][ihead],
                           ref_patch, prop_patch, 
                           ref_pix, prop_pix, prop_i, valid_ref, valid_prop,
                           ps,pt,dilation,reflect_bounds,
                           patch_offset,invalid,offs,T,C,qH,qW,kH,kW);
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

void non_local_search_bilin2d_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor flows,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, float stride1, int dilation, int pt,
    bool reflect_bounds, bool full_ws, int patch_offset,
    int off_Hq, int off_Wq, int dist_type){

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
   int ws_threads = std::min(ws,13);
   int ws_per_thread = (ws/ws_threads) + 1;
   int W_t = dists.size(3);
   int wt = (W_t-1)/2;
   int wt_threads = std::min(W_t,3);
   int wt_per_thread = (W_t/wt_threads) + 1;

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

   // -- viz --
   // fprintf(stdout,"ws_h,ws_w: %d,%d\n",ws_h,ws_w);
   // fprintf(stdout,"nquery_blocks,B,HD: %d,%d,%d\n",nquery_blocks,B,HD);
   // fprintf(stdout,"full_ws,full_ws_time: %d,%d\n",full_ws,full_ws_time);
   // fprintf(stdout,"full_ws,reflect_bounds: %d,%d\n",full_ws,reflect_bounds);

   // launch kernel
   if (dist_type == 0){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                                  "non_local_search_bilin2d_forward_kernel", ([&] {
       non_local_search_bilin2d_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            ws, wt, ps, pt, stride0, stride1, dilation, 
            reflect_bounds, full_ws, patch_offset,
            nH, nW, nHW, st_offset, off_Hq, off_Wq,
            q_per_thread, ws_per_thread, wt_per_thread);
          }));
   }else if(dist_type == 1){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                                  "non_local_search_bilin2d_forward_kernel", ([&] {
       non_local_search_bilin2d_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            ws, wt, ps, pt, stride0, stride1, dilation, 
            reflect_bounds, full_ws, patch_offset,
            nH, nW, nHW, st_offset, off_Hq, off_Wq,
            q_per_thread, ws_per_thread, wt_per_thread);
          }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}


/****************************

  Backward Pass (Vid0,Vid1)

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void nls_bwd_vid_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> inds,
    int ps, int pt, int stride0, int dilation, bool reflect_bounds, int patch_offset,
    int off_Hq, int off_Wq, int ftrs_per_thread) {

  // -- shape --
  int B = grad_dists.size(0);
  int K = grad_dists.size(5);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int qH = vid0.size(4);
  int qW = vid0.size(5);
  int kH = vid1.size(4);
  int kW = vid1.size(5);
  int nH = inds.size(3);
  int nW = inds.size(4);
  int nHW = nH*nW;
  int Q = T*nHW;


  // -- fwd decl registers --
  int ref_patch[3];
  scalar_t prop_patch[3];
  int ref[3];
  scalar_t prop[3];
  int prop_i[3];
  bool valid_ref[4];
  bool valid_prop[4];

  bool valid;
  scalar_t weight;
  scalar_t iweight[3];
  int iftr;
  bool valid_prop_patch;

  // -- offsets --
  int offs[2];
  offs[0] = off_Hq;
  offs[1] = off_Wq;

  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/B;
  int ibatch = (blockIdx.z-ihead*B) % B;

  // -- feature chunk --
  int ftr_start = threadIdx.z * ftrs_per_thread;
  int ftr_end = min(F,ftr_start + ftrs_per_thread);


  // -- each region --
  if ((qi < Q) && (ki < K)){

    // -- full-resolution video query index --
    get_pixel_loc(ref_patch,qi,stride0,nW,nHW,qH,qW);
    int ti = ref_patch[0];
    int nh = ref_patch[1]/stride0;
    int nw = ref_patch[2]/stride0;

    // -- read from tensors --
    weight = grad_dists[ibatch][ihead][ti][nh][nw][ki];
    prop_patch[0] = ref_patch[0] + inds[ibatch][ihead][ti][nh][nw][ki][0];
    prop_patch[1] = ref_patch[1] + inds[ibatch][ihead][ti][nh][nw][ki][1];
    prop_patch[2] = ref_patch[2] + inds[ibatch][ihead][ti][nh][nw][ki][2];
    // check_bounds<scalar_t>(valid_prop_patch,prop_patch,T,kH,kW);
    // if (not valid_prop_patch){ return; }
    
    // -- update vid0,vid1 --
    update_bwd_patch_bilin2d<scalar_t,DIST_TYPE>(
                     grad_vid0[ibatch][ihead],grad_vid1[ibatch][ihead],
                     vid0[ibatch][ihead],vid1[ibatch][ihead],
                     weight,ref_patch,prop_patch,
                     ps,pt,dilation,reflect_bounds,patch_offset,
                     iftr,ftr_start,ftr_end,ref,prop,prop_i,
                     valid_ref,valid_prop,valid,offs,T,F,qH,qW,kH,kW);


  }
}

void non_local_search_bilin2d_vid_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset, int off_Hq, int off_Wq, int dist_type){


  // -- unpack --
  int B = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  // int H = vid0.size(4);
  // int W = vid0.size(5);
  int BHD = B*HD;

  // -- num --
  int nH = inds.size(3);
  int nW = inds.size(4);
  int nHW = nH * nW;
  int Q = T*nH*nW;
  assert(pt == 1);
  int K = inds.size(5);

  // -- share --
  // int nH = (H-1)/stride0+1;
  // int nW = (W-1)/stride0+1;
  // int nHW = nH * nW;
  // assert(pt == 1);

  // -- launch parameters --
  // int nbatch = grad_dists.size(0);
  // int nheads = grad_dists.size(1);
  // int nq = grad_dists.size(2);
  int ftr_threads = min(1,F);
  int ftrs_per_thread = (F-1)/ftr_threads+1;
  dim3 threadsPerBlock(32,16,ftr_threads);
  dim3 blocksPerGrid(1, 1, B*HD);
  blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(K)/double(threadsPerBlock.y));


  // -- view launch info --
  // fprintf(stdout,"BHD,nblocks_queries,chnls_nblocks: %d,%d,%d\n",
  //         BHD,nblocks_queries,chnls_nblocks);
  // fprintf(stdout,"query_nthreads,neigh_nthreads: %d,%d\n",
  //         query_nthreads,neigh_nthreads);
  // fprintf(stdout,"reflect_bounds: %d\n",reflect_bounds);
  // int W_t = dists.size(3);
  // int wt = (W_t-1)/2;

  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"nls_bwd_vid_kernel", ([&] {
        nls_bwd_vid_kernel<scalar_t,0>
        <<<blocksPerGrid, threadsPerBlock>>>(
            grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            ps, pt, stride0, dilation, reflect_bounds, patch_offset, 
            off_Hq, off_Wq, ftrs_per_thread);}));
  }else if (dist_type == 1){ // l2
      AT_DISPATCH_FLOATING_TYPES(vid0.type(),"nls_bwd_vid_kernel", ([&] {
      nls_bwd_vid_kernel<scalar_t,1>
        <<<blocksPerGrid, threadsPerBlock>>>(
            grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            ps, pt, stride0, dilation, reflect_bounds, patch_offset, 
            off_Hq, off_Wq, ftrs_per_thread);}));
  }else{
    throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
  }

}



/**************************************

  Backward Pass (Vid0,Vid1,Flows)

**************************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void nls_bwd_vidflows_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> grad_flows,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> flows,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> grad_inds,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> inds,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset, int st_offset,
    int off_Hq, int off_Wq, int ftrs_per_thread) {

  // -- shape --
  int B = grad_dists.size(0);
  int K = grad_dists.size(5);
  int HD = vid0.size(1);
  int HD_f = flows.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int qH = vid0.size(4);
  int qW = vid0.size(5);
  int kH = vid1.size(4);
  int kW = vid1.size(5);
  int nH = inds.size(3);
  int nW = inds.size(4);
  int nHW = nH*nW;
  int Q = T*nHW;


  // -- offsets --
  int offs[2];
  offs[0] = off_Hq;
  offs[1] = off_Wq;

  // -- fwd decl registers --
  int ref_patch[3];
  scalar_t prop_patch[3];
  int ref[3];
  scalar_t prop[3];
  int prop_i[3];
  bool valid_ref[4];
  bool valid_prop[4];
  bool valid_prop_patch;

  bool valid;
  scalar_t weight;
  scalar_t iweight[2];
  int iftr;

  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/B;
  int ihead_f = ihead % HD_f;
  int ibatch = (blockIdx.z-ihead*B) % B;

  // -- feature chunk --
  int ftr_start = threadIdx.z * ftrs_per_thread;
  int ftr_end = min(F,ftr_start + ftrs_per_thread);

  // -- each region --
  if ((qi < Q) && (ki < K)){

    // -- full-resolution video query index --
    get_pixel_loc(ref_patch,qi,stride0,nW,nHW,qH,qW);
    int ti = ref_patch[0];
    int nh = ref_patch[1]/stride0;
    int nw = ref_patch[2]/stride0;

    // -- read from tensors --
    weight = grad_dists[ibatch][ihead][ti][nh][nw][ki];
    iweight[0] = grad_inds[ibatch][ihead][ti][nh][nw][ki][1];
    iweight[1] = grad_inds[ibatch][ihead][ti][nh][nw][ki][2];
    prop_patch[0] = ref_patch[0] + inds[ibatch][ihead][ti][nh][nw][ki][0];
    prop_patch[1] = ref_patch[1] + inds[ibatch][ihead][ti][nh][nw][ki][1];
    prop_patch[2] = ref_patch[2] + inds[ibatch][ihead][ti][nh][nw][ki][2];

    // -- update sign --
    // int signH = check_bound(prop_patch[1],H) ? 1 : -1;
    // int signW = check_bound(prop_patch[2],W) ? 1 : -1;
    // prop_patch[0] = bounds(prop_patch[0],T);
    // prop_patch[1] = bounds(prop_patch[1],H);
    // prop_patch[2] = bounds(prop_patch[2],W);

    check_bounds<scalar_t>(valid_prop_patch,prop_patch,T,kH,kW);
    if (not valid_prop_patch){ return; }

    // -- search flow from difference --
    int t_max;
    set_time_range(t_max, ti, T, wt);
    int dt = static_cast<int>(prop_patch[0]) - ti;
    int dto = t_max - ti;
    int si = dt >= 0 ? (dt-st_offset) : dto - dt - st_offset;

    // -- accumulate optical flow update --
    scalar_t acc_dFlows[8];
  #pragma unroll
    for (int _idx=0; _idx < 8; _idx++){
      acc_dFlows[_idx] = static_cast<scalar_t>(0);
    }


    // -- update vid0,vid1,flows --
    update_bwd_bilin2d_vidflows<scalar_t,DIST_TYPE>(
                     grad_vid0[ibatch][ihead],grad_vid1[ibatch][ihead],
                     vid0[ibatch][ihead],vid1[ibatch][ihead],
                     acc_dFlows,weight,ref_patch,prop_patch,
                     ps,pt,dilation,stride0,reflect_bounds,patch_offset,
                     iftr,ftr_start,ftr_end,ref,prop,prop_i,
                     valid_ref,valid_prop,valid,offs,T,qH,qW,kH,kW);

    // -- update grad_flows from grad_dists,vid0,vid1 --
    if (si<0){ return; } // aka same frame
    // if (dt==0){ return; } // aka same frame.
    
    scalar_t hi = ref_patch[1] + flows[ibatch][ihead_f][ti][si][1][nh][nw];
    scalar_t wi = ref_patch[2] + flows[ibatch][ihead_f][ti][si][0][nh][nw];
    int signH = check_bound(hi,kH) ? 1 : -1;
    int signW = check_bound(wi,kW) ? 1 : -1;
    bwd_flow_assign(acc_dFlows,nh,nw,signH,signW,
                    grad_flows[ibatch][ihead_f][ref_patch[0]][si]);

    // -- update grad_flows from grad_inds --
    if (ftr_start == 0){
      atomicAdd(&(grad_flows[ibatch][ihead_f][ti][si][0][nh][nw]),signW*iweight[1]);
      atomicAdd(&(grad_flows[ibatch][ihead_f][ti][si][1][nh][nw]),signH*iweight[0]);
    }


  }
}

void non_local_search_bilin2d_vidflows_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1, torch::Tensor grad_flows,
    const torch::Tensor vid0, const torch::Tensor vid1, const torch::Tensor flows,
    const torch::Tensor grad_dists, const torch::Tensor grad_inds,
    const torch::Tensor dists, const torch::Tensor inds,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset,
    int off_Hq, int off_Wq, int dist_type){


  // -- unpack --
  int B = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  // int H = vid0.size(4);
  // int W = vid0.size(5);
  int BHD = B*HD;

  // -- num --
  int nH = inds.size(3);
  int nW = inds.size(4);
  int nHW = nH * nW;
  int Q = T*nH*nW;
  assert(pt == 1);
  int K = inds.size(5);

  // -- share --
  // int nH = (H-1)/stride0+1;
  // int nW = (W-1)/stride0+1;
  // int nHW = nH * nW;
  // assert(pt == 1);

  // -- launch parameters --
  // int nbatch = grad_dists.size(0);
  // int nheads = grad_dists.size(1);
  // int nq = grad_dists.size(2);
  int ftr_threads = min(1,F);
  int ftrs_per_thread = (F-1)/ftr_threads+1;
  dim3 threadsPerBlock(32,16,ftr_threads);
  dim3 blocksPerGrid(1, 1, B*HD);
  blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(K)/double(threadsPerBlock.y));

  // -- compute st --
  int W_t = 2*wt+1;
  int st_offset = W_t - flows.size(3);
  assert((st_offset == 1) or (st_offset == 0));

  // -- view launch info --
  // fprintf(stdout,"BHD,nblocks_queries,chnls_nblocks: %d,%d,%d\n",
  //         BHD,nblocks_queries,chnls_nblocks);
  // fprintf(stdout,"query_nthreads,neigh_nthreads: %d,%d\n",
  //         query_nthreads,neigh_nthreads);
  // int W_t = dists.size(3);
  // int wt = (W_t-1)/2;

  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"nls_bwd_vidflows_kernel", ([&] {
        nls_bwd_vidflows_kernel<scalar_t,0>
        <<<blocksPerGrid, threadsPerBlock>>>(
            grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            grad_dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            wt, ps, pt, stride0, dilation, reflect_bounds, patch_offset, 
            st_offset, off_Hq, off_Wq, ftrs_per_thread);}));
  }else if (dist_type == 1){ // l2
      AT_DISPATCH_FLOATING_TYPES(vid0.type(),"nls_bwd_vidflows_kernel", ([&] {
      nls_bwd_vidflows_kernel<scalar_t,1>
        <<<blocksPerGrid, threadsPerBlock>>>(
            grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            grad_dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            wt, ps, pt, stride0, dilation, reflect_bounds, patch_offset, 
            st_offset, off_Hq, off_Wq, ftrs_per_thread);}));
  }else{
    throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
  }

}

