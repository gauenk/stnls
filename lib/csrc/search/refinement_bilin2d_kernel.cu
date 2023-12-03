
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include "nls_bilin2d.cu"
using namespace at;


/****************************

     Forward (Bilinear-2d)

****************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void refinement_bilin2d_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> flows,
    torch::PackedTensorAccessor32<scalar_t,8,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<scalar_t,9,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<int,8,torch::RestrictPtrTraits> kselect,
    torch::PackedTensorAccessor32<bool,7,torch::RestrictPtrTraits> reflect,
    int wr, int ws, int ps, int pt, int stride0, float _stride1, int dilation,
    bool restrict_radius, bool reflect_bounds, bool full_ws, int patch_offset,
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
  int Ks = flows.size(5);
  scalar_t stride1 = static_cast<scalar_t>(_stride1);
  int ti,nh,nw;

  // -- invalid constant --
  scalar_t invalid = (scalar_t)__int_as_float(0x7f800000);
  if(DIST_TYPE == 0){ // prod
    invalid = -invalid;
  }

  // -- search region offsets --
  int psHalf = (ps)/2;
  scalar_t wrHalf = (wr-1)/2;
  scalar_t wrOff_h = wrHalf;
  scalar_t wrOff_w = wrHalf;

  // -- cuda index --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int ihead_f = ihead % HD_f;
  int q_start = blockIdx.x*q_per_thread;
  int qi,ki,wh,ww;

  // -- fwd decls --
  scalar_t prop_center[3];
  scalar_t prop_patch[3];
  scalar_t prop_pix[3];
  int prop_i[3];
  int ref_patch[3];
  int ref_pix[3];
  bool valid;
  bool valid_prop[4];
  bool valid_ref[4];
  scalar_t dist;


  for (int _qix = 0; _qix < q_per_thread; _qix++){

    //---------------------------
    //       Anchor Pixel
    //---------------------------

    // -- block start --
    qi = q_start + _qix;
    if (qi >= Q){ continue; }

    // -- pixel location from query index --
    get_pixel_loc(ref_patch,qi,stride0,nW,nHW,H,W);
    ti = ref_patch[0];
    nh = ref_patch[1]/stride0;
    nw = ref_patch[2]/stride0;

    // -- check bounds of pixel location --
    check_bounds(valid_ref[3],ref_patch,T,H,W);

    // ---------------------------------------
    //     for each neighbor in k_search
    // ---------------------------------------
    for(int _ki = 0; _ki < k_per_thread; _ki++){
      ki = threadIdx.x + blockDim.x*_ki;
      if (ki >= Ks){ continue; }

      // -- unpack base -- 
      prop_patch[0] = ref_patch[0] + floor(flows[ibatch][ihead_f][ti][nh][nw][ki][0]+0.5);
      prop_center[0] = ref_patch[1] + flows[ibatch][ihead_f][ti][nh][nw][ki][1];
      prop_center[1] = ref_patch[2] + flows[ibatch][ihead_f][ti][nh][nw][ki][2];
      prop_patch[0] = bounds(prop_patch[0],T);

      // -- possibly illegal flows --
      valid = abs(flows[ibatch][ihead_f][ti][nh][nw][ki][1]) < 1e8;
      valid = valid and abs(flows[ibatch][ihead_f][ti][nh][nw][ki][2]) < 1e8;
      if (not(valid)){ continue; }

      // -- bounding --
      reflect[ibatch][ihead_f][ti][nh][nw][ki][0] = not check_bound(prop_center[0],H);
      reflect[ibatch][ihead_f][ti][nh][nw][ki][1] = not check_bound(prop_center[1],W);
      prop_center[0] = bounds(prop_center[0],H);
      prop_center[1] = bounds(prop_center[1],W);

      // -- search region offsets --
      set_search_offsets(wrOff_h, wrOff_w,
                         prop_center[0], prop_center[1], stride1,
                         wrHalf, wr, H, W, full_ws);

      // -- [unused] set search bounds for [optionally] expanded region --
      // set_search_minmax(wrMax_h, wrMin_h, wrOff_h, wr_h, stride1, full_ws);
      // set_search_minmax(wrMax_w, wrMin_w, wrOff_w, wr_w, stride1, full_ws);

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
          valid = valid_prop[3];//valid_ref[3] && valid_prop[3];


          //  -- compute patch difference --
          if (valid){
            compute_dist_bilin2d<scalar_t,DIST_TYPE>(dist,
                         vid0[ibatch][ihead],vid1[ibatch][ihead],
                         ref_patch, prop_patch, ref_pix, prop_pix,
                         prop_i, valid_ref, valid_prop,
                         ps,pt,dilation,reflect_bounds,
                         patch_offset,invalid,T,F,H,W);
          }

          // -- assignent --
          if (!valid){ dist = invalid; }
          dists[ibatch][ihead][ti][nh][nw][ki][wh][ww] = dist;
          inds[ibatch][ihead][ti][nh][nw][ki][wh][ww][0] = prop_patch[0]-ref_patch[0];
          inds[ibatch][ihead][ti][nh][nw][ki][wh][ww][1] = prop_patch[1]-ref_patch[1];
          inds[ibatch][ihead][ti][nh][nw][ki][wh][ww][2] = prop_patch[2]-ref_patch[2];
          kselect[ibatch][ihead][ti][nh][nw][ki][wh][ww] = ki;


        } //  ww
      } // wh
    } // ki
  } // qi
} // fxn

void refinement_bilin2d_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1, const torch::Tensor flows,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor kselect, torch::Tensor reflect,
    int ws, int ps,  int stride0, float stride1, int dilation, int pt,
    bool restrict_radius, bool reflect_bounds, bool full_ws,
    int patch_offset, int dist_type){

   // -- num threads --
   int B = dists.size(0);
   int HD = dists.size(1);
   int T = dists.size(2);
   int nH = dists.size(3);
   int nW = dists.size(4);
   int Ks = dists.size(5);
   int wr = dists.size(6);

   int Q = T*nH*nW;
   int Ks_threads = std::min(Ks,11);
   int k_per_thread = ((Ks-1)/Ks_threads)+1;
   int wr_threads = std::min(wr,5);
   int wr_per_thread = ((wr-1)/wr_threads) + 1;
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
     AT_DISPATCH_FLOATING_TYPES(vid0.type(),"refinement_bilin2d_forward_kernel", ([&] {
          refinement_bilin2d_forward_kernel<scalar_t,0><<<nblocks, nthreads>>>(
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,8,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<scalar_t,9,torch::RestrictPtrTraits>(),
          kselect.packed_accessor32<int,8,torch::RestrictPtrTraits>(),
          reflect.packed_accessor32<bool,7,torch::RestrictPtrTraits>(),
          wr, ws, ps, pt, stride0, stride1, dilation,
          restrict_radius, reflect_bounds, full_ws, patch_offset,
          q_per_thread, k_per_thread, wr_per_thread);
        }));
   }else if (dist_type == 1){
     AT_DISPATCH_FLOATING_TYPES(vid0.type(),"refinement_bilin2d_forward_kernel", ([&] {
          refinement_bilin2d_forward_kernel<scalar_t,1><<<nblocks, nthreads>>>(
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,8,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<scalar_t,9,torch::RestrictPtrTraits>(),
          kselect.packed_accessor32<int,8,torch::RestrictPtrTraits>(),
          reflect.packed_accessor32<bool,7,torch::RestrictPtrTraits>(),
          wr, ws, ps, pt, stride0, stride1, dilation,
          restrict_radius, reflect_bounds, full_ws, patch_offset,
          q_per_thread, k_per_thread, wr_per_thread);
        }));
   }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
   }
}



/****************************

       Backward Indices 

****************************/

template <typename scalar_t>
__global__ void refinement_flows_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_flows,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_inds,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> flows,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
    int QK, int num_per_thread){

  // -- unpack shape --
  int Ksearch = flows.size(3);
  int Kagg = inds.size(3);

  // -- decl helpers --
  bool eq;
  int index;

  // -- get indices --
  int _index = num_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;

  for (int _ix = 0; _ix < num_per_thread; _ix++){

    // -- select (qi,ki) --
    index = _index + _ix;
    if (index >= QK){ break; }
    int qi = index / Kagg;
    int ki = index - qi*Kagg;

    for (int ks=0; ks < Ksearch; ks++){

      // -- find matching index --
      eq = true;
#pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        eq = eq and (fabs(inds[ibatch][ihead][qi][ki][_idx] -    \
                          flows[ibatch][ihead][qi][ks][_idx]) < 1e-10);
      }

      // -- assign --
      if (eq){
#pragma unroll
        for (int _idx=0; _idx < 3; _idx++){
          grad_flows[ibatch][ihead][qi][ks][_idx] = \
            grad_inds[ibatch][ihead][qi][ki][_idx]; // should be unique
        }
        continue; // pick next (qi,ki)
      }

    }

  }

} // fxn

void refinement_flows_backward_cuda(
    torch::Tensor grad_flows, const torch::Tensor grad_inds,
    const torch::Tensor flows, const torch::Tensor inds){

   // -- shape --
   int nbatch = inds.size(0);
   int nheads = inds.size(1);
   int nqueries = inds.size(2);
   int kagg = inds.size(3);
   int QK = nqueries * kagg;

   // -- num threads --
   int _nthreads = 256;
   dim3 nthreads(_nthreads);

   // -- num blocks --
   int num_per_thread = 1;
   int nRun = nqueries*kagg;
   int _nblocks = (nRun-1)/(_nthreads*num_per_thread)+1;
   dim3 nblocks(_nblocks,nbatch,nheads);

   // -- launch kernel --
   AT_DISPATCH_FLOATING_TYPES(inds.type(),"ref_bwd_inds_kernel", ([&] {
   refinement_flows_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
          grad_flows.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          flows.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          QK,num_per_thread);
       }));

}


/**************************************

  Backward Pass (Vid0,Vid1,Flows)

**************************************/

template <typename scalar_t, int DIST_TYPE>
__global__ void refinement_vidflows_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> grad_flows,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    // const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> flows,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> grad_inds,
    // const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> kselect,
    const torch::PackedTensorAccessor32<bool,7,torch::RestrictPtrTraits> reflect,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset, int ftrs_per_thread) {

  // -- shape --
  int B = grad_dists.size(0);
  int K = grad_dists.size(5);
  int HD = vid0.size(1);
  int HD_f = grad_flows.size(1);
  // int HD_f = flows.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
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
    get_pixel_loc(ref_patch,qi,stride0,nW,nHW,H,W);
    int ti = ref_patch[0];
    int nh = ref_patch[1]/stride0;
    int nw = ref_patch[2]/stride0;

    // -- read from tensors --
    weight = grad_dists[ibatch][ihead][ti][nh][nw][ki];
    iweight[0] = grad_inds[ibatch][ihead][ti][nh][nw][ki][1];
    iweight[1] = grad_inds[ibatch][ihead][ti][nh][nw][ki][2];
  #pragma unroll
    for (int _idx=0; _idx < 3; _idx++){
      prop_patch[_idx] = ref_patch[_idx]+inds[ibatch][ihead][ti][nh][nw][ki][_idx];
    }

    // if (!check_bound(prop_patch[0],T)){ // skip invalid times
    //   assert(1==0);
    //   return;
    // }

    // -- get source kj from shuffled ki--
    int kj = kselect[ibatch][ihead][ti][nh][nw][ki];

    // -- update grad_flows from grad_inds --
    int signH = reflect[ibatch][ihead_f][ti][nh][nw][kj][0] ? -1 : 1;
    int signW = reflect[ibatch][ihead_f][ti][nh][nw][kj][1] ? -1 : 1;
    if (ftr_start == 0){
      atomicAdd(&(grad_flows[ibatch][ihead_f][ti][nh][nw][kj][1]),signH*iweight[0]);
      atomicAdd(&(grad_flows[ibatch][ihead_f][ti][nh][nw][kj][2]),signW*iweight[1]);
    }

    // -- accumulate optical flow update --
    scalar_t acc_dFlows[8];
  #pragma unroll
    for (int _idx=0; _idx < 8; _idx++){
      acc_dFlows[_idx] = static_cast<scalar_t>(0);
    }

    // -- optionally skip if invalid --
    check_bounds<scalar_t>(valid_prop_patch,prop_patch,T,H,W);
    // assert(valid_prop_patch==true); // only when wr == 1
    if (not valid_prop_patch){ return; }

    // -- update vid0,vid1,flows --
    update_bwd_bilin2d_vidflows<scalar_t,DIST_TYPE>(
                     grad_vid0[ibatch][ihead],grad_vid1[ibatch][ihead],
                     vid0[ibatch][ihead],vid1[ibatch][ihead],
                     acc_dFlows,weight,ref_patch,prop_patch,
                     ps,pt,dilation,stride0,reflect_bounds,patch_offset,
                     iftr,ftr_start,ftr_end,ref,prop,prop_i,
                     valid_ref,valid_prop,valid,T,H,W);

    // -- update grad_flows from grad_dists --
    bwd_flow_assign_v2(acc_dFlows,signH,signW,
                       grad_flows[ibatch][ihead_f][ti][nh][nw][kj]);

    // -- update grad_flows from grad_inds --
    // if (ftr_start == 0){
    //   atomicAdd(&(grad_flows[ibatch][ihead_f][ti][nh][nw][kj][2]),sW*iweight[1]);
    //   atomicAdd(&(grad_flows[ibatch][ihead_f][ti][nh][nw][kj][1]),sH*iweight[0]);
    // }


  }
}

void refinement_bilin2d_vidflows_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1, torch::Tensor grad_flows,
    const torch::Tensor vid0, const torch::Tensor vid1, //const torch::Tensor flows,
    const torch::Tensor grad_dists, const torch::Tensor grad_inds,
    const torch::Tensor inds, const torch::Tensor kselect, const torch::Tensor reflect,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset, int dist_type){


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
  // int W_t = dists.size(3);
  // int wt = (W_t-1)/2;

  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),"refinement_vidflows_kernel", ([&] {
        refinement_vidflows_backward_kernel<scalar_t,0>
        <<<blocksPerGrid, threadsPerBlock>>>(
            grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            // flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            grad_dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            // dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            kselect.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
            reflect.packed_accessor32<bool,7,torch::RestrictPtrTraits>(),
            wt, ps, pt, stride0, dilation, reflect_bounds, patch_offset, 
            ftrs_per_thread);}));
  }else if (dist_type == 1){ // l2
      AT_DISPATCH_FLOATING_TYPES(vid0.type(),"refinement_vidflows_kernel", ([&] {
      refinement_vidflows_backward_kernel<scalar_t,1>
        <<<blocksPerGrid, threadsPerBlock>>>(
            grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            // flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            grad_dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            grad_inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            // dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
            kselect.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
            reflect.packed_accessor32<bool,7,torch::RestrictPtrTraits>(),
            wt, ps, pt, stride0, dilation, reflect_bounds, patch_offset, 
            ftrs_per_thread);}));
  }else{
    throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");
  }

}

