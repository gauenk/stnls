
// #include <torch/extension.h>
#include <cuda/std/type_traits>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
// #include "shared_kernel.cu"
#include "nls_bilin2d.cu"

using namespace at;


/****************************

       Forward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void non_local_search_forward_bilin2d_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> inds,
    int ws_h, int ws_w, int wt, int ps, int pt,
    int stride0, float _stride1, int dilation,
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
  scalar_t stride1 = static_cast<scalar_t>(_stride1);

  // -- invalid constant --
  float invalid = __int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }

  // -- search region offsets --
  // int wsHalf_h = (ws_h-1)/2;
  // int wsHalf_w = (ws_w-1)/2;
  // int wsOff_h,wsOff_w;

  // -- search region offsets --
  scalar_t wsHalf_h = trunc((ws_h)/2);
  scalar_t wsHalf_w = trunc((ws_w)/2);
  // scalar_t wsHalf_h = (ws_h-1)/2;
  // scalar_t wsHalf_w = (ws_w-1)/2;
  scalar_t wsOff_h,wsOff_w;

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
  // bool acc_flow = fflow.size(2) == 1;
  bool acc_flow = true;//fflow.size(1) != ST;
  int deltaT;

  // decls
  int ref_patch[3];
  scalar_t prop_patch[3];
  int prop_i[3];
  scalar_t frame_anchor[3];
  int ref_pix[3];
  scalar_t prop_pix[3];
  int prop_pix_i[3];
  bool valid;
  bool valid_ref_patch,valid_prop_patch;
  bool valid_ref[4];
  bool valid_prop[4];
  scalar_t acc_frame[3];


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
    get_pixel_loc<int>(ref_patch,qindex,qindex_tmp,stride0,nW0,nHW0,H,W);

    // -- check bounds of pixel location --
    check_bounds<int>(valid_ref_patch,ref_patch,T,H,W);

    // -- temporal search bounds --
    set_time_range(t_max,t_shift,ref_patch[0],T,wt);

    // -- init search params --
    frame_anchor[0] = __int2float_rn(ref_patch[0]);
    frame_anchor[1] = __int2float_rn(ref_patch[1]);
    frame_anchor[2] = __int2float_rn(ref_patch[2]);
    acc_frame[0] = frame_anchor[0];
    acc_frame[1] = frame_anchor[1];
    acc_frame[2] = frame_anchor[2];
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
      // increment_frame<scalar_t>(acc_frame[0],prev_ti,t_inc,
      //                           swap_dir,dir,ref_patch[0],t_max);
      increment_frame<scalar_t>(frame_anchor[0],prev_ti,t_inc,
                                swap_dir,dir,ref_patch[0],t_max);

      // -- possibly reset (frame_anchor <- reference_patch) --
      // reset_centers<scalar_t>(frame_anchor,ref_patch,swap_dir || not(acc_flow));
      reset_centers<scalar_t>(acc_frame,ref_patch,swap_dir);
      // reset_centers<scalar_t>(acc_frame,ref_patch,swap_dir || not(acc_flow));
      // reset_centers<scalar_t>(frame_anchor,ref_patch,true);
      // frame_anchor[0] = swap_dir ? __float2int_rn(ref_patch[0])-1 : frame_anchor[0];
      // frame_anchor[1] = __int2float_rn(ref_patch[1]);
      // frame_anchor[2] = __int2float_rn(ref_patch[2]);

      // -- compute offset with optical flow --
      // deltaT = abs(ref_patch[0] - __float2int_rn(acc_frame[0]));
      // deltaT = abs(ref_patch[0] - __float2int_rn(frame_anchor[0]));
      // int acc_index = acc_flow ? 0 : deltaT;
      deltaT = 0;//acc_flow ? deltaT : 1;
      update_centers_v0<scalar_t>(frame_anchor[1],frame_anchor[2],dir,H,W,
                                  fflow[ibatch][prev_ti][deltaT],
                                  bflow[ibatch][prev_ti][deltaT]);
      // update_centers<scalar_t,scalar_t>(frame_anchor[1],frame_anchor[2],
      //                                   ref_patch[0],dir,deltaT,H,W,
      //                                   fflow[ibatch][acc_index],
      //                                   bflow[ibatch][acc_index]);
      // update_centers<scalar_t,scalar_t>(acc_frame[1],acc_frame[2],
      //                                   ref_patch[0],dir,deltaT,H,W,
      //                                   fflow[ibatch][acc_index],
      //                                   bflow[ibatch][acc_index]);
      // auto flow = dir > 0 ? fflow[ibatch][acc_index] : bflow[ibatch][acc_index];
      // if (dir!=0){
      //   update_centers_flow(acc_frame[1],acc_frame[2],H,W,flow[deltaT-1]);
      // }

      // frame_anchor[0] = acc_frame[0];
      // frame_anchor[1] = bounds(acc_frame[1],H);
      // frame_anchor[2] = bounds(acc_frame[2],W);
      frame_anchor[1] = bounds(frame_anchor[1],H);
      frame_anchor[2] = bounds(frame_anchor[2],W);
      // frame_anchor[1] = floorf(frame_anchor[1]*1000)/(float)1000;
      // frame_anchor[2] = floorf(frame_anchor[2]*1000)/(float)1000;


      // -- search region offsets --
      // wsOff_h = 0;
      // wsOff_w = 0;
      set_search_offsets<scalar_t>(wsOff_h, wsOff_w,
                                   frame_anchor[1], frame_anchor[2],
                                   stride1, wsHalf_h, wsHalf_w,
                                   ws_h, ws_w, H, W, true);

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
          prop_patch[0] = frame_anchor[0];
          prop_patch[1] = frame_anchor[1] + stride1 * (ws_i - wsOff_h);
          prop_patch[2] = frame_anchor[2] + stride1 * (ws_j - wsOff_w);
          // set_search_patch<scalar_t>(prop_patch,frame_anchor,stride1,
          //                            ws_i,ws_j,wsOff_h,wsOff_w,search_abs);
          check_bounds<scalar_t>(valid_prop_patch,prop_patch,T,H,W);
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

void non_local_search_forward_bilin2d_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int wt, int ps, int k, int dist_type,
    int stride0, float stride1, int dilation, int pt, int q_shift,
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
   int ws_h_threads = std::min(ws_h,25);
   int ws_w_threads = std::min(ws_w,25);
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
   // fprintf(stdout,"full_ws,full_ws_time: %d,%d\n",full_ws,full_ws_time);

   // launch kernel
   if (dist_type == 0){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                                  "non_local_search_forward_bilin2d_kernel", ([&] {
       non_local_search_forward_bilin2d_kernel<scalar_t,0><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            fflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            bflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            ws_h, ws_w, wt, ps, pt, stride0, stride1, dilation, 
            q_shift, nH0, nW0, nHW0, reflect_bounds, full_ws, full_ws_time,
            search_abs, patch_offset, off_H0, off_W0, off_H1, off_W1,
            q_per_thread, ws_h_per_thread, ws_w_per_thread);
          }));
   }else if(dist_type == 1){
       AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                                  "non_local_search_forward_bilin2d_kernel", ([&] {
       non_local_search_forward_bilin2d_kernel<scalar_t,1><<<nblocks, nthreads>>>(
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            fflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            bflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
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

       Backward Pass

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void nls_bwd_dists_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
    // torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_fflow,
    // torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_bflow,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    // const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> fflow,
    // const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> bflow,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
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
  scalar_t prop_patch[3];
  int ref[3];
  scalar_t prop[3];
  int prop_i[3];
  bool valid_ref[4];
  bool valid_prop[4];
  int qindex,qindex_tmp;

  bool valid;
  scalar_t weight,pix0,pix1,pix;
  scalar_t iweight[3];
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
    get_pixel_loc(ref_patch,qindex,qindex_tmp,stride0,nW0,nHW0,H,W);

    // -- read from tensors --
    weight = grad_dists[ibatch][ihead][i0][i1];
  #pragma unroll
    for (int _idx=0; _idx < 3; _idx++){
      prop_patch[_idx] = inds[ibatch][ihead][i0][i1][_idx];
    }

    // -- update vid0,vid1 --
    update_bwd_patch_bilin2d<scalar_t,DIST_TYPE>(
                     grad_vid0[ibatch][ihead],grad_vid1[ibatch][ihead],
                     vid0[ibatch][ihead],vid1[ibatch][ihead],
                     weight,ref_patch,prop_patch,
                     ps,pt,dilation,reflect_bounds,
                     center_offsets,patch_offset,
                     iftr,ftr_start,ftr_end,
                     ref,prop,prop_i,
                     valid_ref,valid_prop,valid,
                     T,H,W,pix0,pix1,pix,i1);

  }
}


// template <typename scalar_t, int DIST_TYPE>
// __global__ void nls_bwd_flow_kernel(
//     torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_fflow,
//     torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_bflow,
//     const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> fflow,
//     const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> bflow,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_inds,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
//     int q_shift, int stride0, int nH0, int nW0, int nHW0) {

//   // -- shape --
//   int nbatch = grad_dists.size(0);
//   int Q = grad_dists.size(2);
//   int K =  grad_dists.size(3);
//   int HD = vid0.size(1);
//   int T = vid0.size(2);
//   int F = vid0.size(3);
//   int H = vid0.size(4);
//   int W = vid0.size(5);

//   // -- fwd decl registers --
//   int ref_patch[3];
//   scalar_t prop_patch[3];
//   int ref[3];
//   scalar_t prop[3];
//   int prop_i[3];
//   bool valid_ref[4];
//   bool valid_prop[4];
//   int qindex,qindex_tmp;
//   scalar_t iweight[3];


//   // -- pixel location from query index --
//   qindex = i0 + q_shift;
//   get_pixel_loc(ref_patch,qindex,qindex_tmp,stride0,nW0,nHW0,H,W);

//   // -- update fflow,bflow --
//   update_bwd_flow<scalar_t>(
//              grad_fflow[ibatch][0],
//              grad_bflow[ibatch][0],
//              fflow[ibatch][0],bflow[ibatch][0],
//              inds[ibatch][ihead][i0],
//              grad_inds[ibatch][ihead][i0],
//              ref_patch,prop_patch,iweight,ref_patch,prop_i,
//              // ps,pt,dilation,reflect_bounds,
//              // center_offsets,patch_offset,
//              // iftr,ftr_start,ftr_end,
//              // valid_ref,valid_prop,valid,
//              T,H,W);

// }


void non_local_search_backward_bilin2d_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor grad_fflow, torch::Tensor grad_bflow,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    const torch::Tensor grad_dists, const torch::Tensor grad_inds,
    const torch::Tensor inds, int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int wt, int dilation, bool reflect_bounds, bool use_adj,
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
  // int ftr_threads = min(1,F);
  // dim3 threadsPerBlock(32,8,ftr_threads);
  dim3 threadsPerBlock(4,4,ftr_threads);
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
  int st = 2*wt+1;
  bool ACC_FLOW = fflow.size(1) != st; // (b,st,t,2,h,w) because Transposed.
  // fprintf(stdout,"%d,%d\n",fflow.size(2),st);

  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"nls_bwd_dists_kernel", ([&] {
        nls_bwd_dists_kernel<scalar_t,0>
        <<<blocksPerGrid, threadsPerBlock>>>(
            grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            // grad_fflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            // grad_bflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            // count0.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
            // count1.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
            q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
            ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);}));
  }else if (dist_type == 1){ // l2
      AT_DISPATCH_FLOATING_TYPES(vid0.type(),"nls_bwd_dists_kernel", ([&] {
      nls_bwd_dists_kernel<scalar_t,1>
        <<<blocksPerGrid, threadsPerBlock>>>(
            grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            // grad_fflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            // grad_bflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            // count0.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
            // count1.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
            q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
            ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);}));
  }else{
    throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
  }

  // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  //
  //     backward optical flow
  //
  // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  // // -- fill the accumulated flows --
  // auto options = torch::TensorOptions().dtype(torch::kFloat32)
  //   .layout(torch::kStrided)
  //   .device(torch::kCUDA, fflow.device().index());
  // torch::Tensor afflow = torch::zeros({B,T,T-1,2,H,W});
  // torch::Tensor bfflow = torch::zeros({B,T,T-1,2,H,W});
  // threadsPerBlock.x = 256;
  // threadsPerBlock.y = 2;
  // threadsPerBlock.z = T-1;
  // blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  // blocksPerGrid.y = B;
  // blocksPerGrid.z = 0;
  // AT_DISPATCH_FLOATING_TYPES(fflow.type(),"nls_fill_acc_flows_kernel", ([&] {
  //   nls_fill_acc_flows_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
  //         afflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
  //         bfflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
  //         grad_fflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
  //         grad_fflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
  //         q_shift, stride0);}));


  // int locs_per_thread = 1;
  // int _nblocks = (nRun-1)/(_nthreads*locs_per_thread)+1;
  // dim3 nblocks(_nblocks,B);
  // int Q = inds.size(1);
  // threadsPerBlock.x = 256;
  // threadsPerBlock.y = 2;
  // threadsPerBlock.z = T-1;
  // blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  // blocksPerGrid.y = B;
  // blocksPerGrid.z = 0;


  // AT_DISPATCH_FLOATING_TYPES(fflow.type(),"nls_bwd_flow_kernel", ([&] {
  //   nls_bwd_flow_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
  //         grad_fflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
  //         grad_bflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
  //         fflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
  //         bflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
  //         grad_inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
  //         inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
  //         q_shift, stride0);}));

}


