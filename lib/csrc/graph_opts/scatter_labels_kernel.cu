/*

  Stack non-local patches into a video

*/

#include <cuda/std/type_traits>
#include <cstdio>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <math.h>
#include <ATen/ATen.h>
#include "../shared_kernel.cu"

using namespace at;

// #include "scatter_int.cu"


/**************************************

          Get Scatter Labels

**************************************/

__device__ __forceinline__
void get_unique_index(int& li, bool& oob,
                      int nl_hi,int nl_wi,int hi,int wi,
                      int wsOff_h,int wsOff_w,int time_offset,
                      int stride0, int stride1,int ws,int wsHalf,bool full_ws){
  
  // -- init --
  int ws_i = -1;
  int ws_j = -1;
  
  // -- check spatial coordinates --
  int num_h = nl_hi - hi;//stride1;
  // num_h = (nl_hi >= hi) ? num_h : -num_h;
  int num_w = nl_wi - wi;//stride1;
  // num_w = (nl_wi >= wi) ? num_w : -num_w;
  
  // -- check oob --
  int wsNum = (ws-1)/stride0+1;
  bool oob_i = abs(num_h) > wsHalf;
  bool oob_j = abs(num_w) > wsHalf;
  oob = (oob_i or oob_j) and full_ws;
  bool and_oob = oob_i and oob_j and full_ws;
  bool xor_oob = (oob_i or oob_j) and not(oob_i and oob_j) and full_ws;

  // -- oob names --
  if (oob_i and oob_j){

    // // -- di,dj --
    int adj_h = wsHalf - wsOff_h;
    int adj_w = wsHalf - wsOff_w;

    // int di = wsHalf - abs(wsHalf - wsOff_h);
    // int dj = wsHalf - abs(wsHalf - wsOff_w);

    // // -- small square --
    // int mi = di + wsHalf*dj;
    // ws_i = mi % ws;
    // ws_j = mi / ws + (ws-1);

    // -- only adj --
    ws_i = (abs(adj_h)-1)/stride0;
    ws_j = (abs(adj_w)-1)/stride0;

  }else if (oob_i and not(oob_j)){
    ws_j = abs(num_h) - (wsHalf+1);
    ws_i = num_w+wsHalf;
  }else if (oob_j and not(oob_i)){
    // ws_j = abs(num_w) - (wsHalf+1) + (wsHalf);
    ws_j = abs(num_w) - (wsHalf+1);
    ws_i = num_h+wsHalf;
  }

  // -- standard names --
  if (not(oob)){
    ws_i = num_h + wsHalf;
    ws_j = num_w + wsHalf;
  }

  // -- standard names --
  if (not(and_oob)){
    ws_i = ws_i/stride0;
    ws_j = ws_j/stride0;
  }

  // -- get unique index --
  if (not(oob_i or oob_j)){
      li = (ws_i) + (ws_j)*wsNum + time_offset;
  // }else if (xor_oob and oob_i){
  }else if (not(oob_j) and oob_i){
    li = (ws_i) + (ws_j)*wsNum + time_offset + wsNum*wsNum;
  // }else if (xor_oob and oob_j){
  }else if (not(oob_i) and oob_j){
    li = (ws_i) + (ws_j)*wsNum + (wsNum/2)*wsNum + time_offset + wsNum*wsNum;
  }else if (and_oob){
      li = (ws_i) + (ws_j)*(wsNum/2);
      li = li + time_offset + wsNum*wsNum + 2*(wsNum/2)*wsNum;
  }else{
    assert(1==0);
    // li = -1; // skip me!
  }

  // // -- get unique index --
  // li = (ws_i) + (ws_j)*ws + time_offset;
  // li = oob ? li + ws*ws : li;

}

__global__ void scatter_labels_kernel(
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> flows,
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> flows_k,
    torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> names,
    int ws, int wt, int stride0, int stride1, bool full_ws, int st_offset){

    // -- unpack --
    int B = flows.size(0);
    int HD = flows.size(1);
    int T = flows.size(2);
    int nH = flows.size(5);
    int nW = flows.size(6);
    int K = flows_k.size(5);
    int S = names.size(2);
    int H = names.size(4);
    int W = names.size(5);

    // -- derived --
    int nHW = nH*nW;
    int Q = T*nHW;

    // -- indexing variables --
    int ref_patch[3];
    int nl_patch[3];
    bool valid_patch;
  
    // -- search window params --
    int wsHalf0 = (ws-1)/2;
    int wsHalf = (ws)/2;
    int wsOff_h = wsHalf0;
    int wsOff_w = wsHalf0;
    int wsOff_h_nl = wsHalf;
    int wsOff_w_nl = wsHalf;

    // -- location to fill --
    int qi = blockIdx.x*blockDim.x+threadIdx.x;
    int ki = blockIdx.y*blockDim.y+threadIdx.y;
    int ihead = blockIdx.z/B;
    int ibatch = (blockIdx.z-ihead*B) % B;
  
    // -- each region --
    if ((qi < Q) && (ki < K)){
  
      //----------------------------------
      //   Reference & Non-Local Pixel
      //----------------------------------
  
      // -- reference index --
      get_pixel_loc(ref_patch,qi,stride0,nW,nHW,H,W);
      int ti = ref_patch[0];
      // int h_ref = ref_patch[1];
      // int w_ref = ref_patch[2];
      int hi = ref_patch[1]/stride0;
      int wi = ref_patch[2]/stride0;

      // bool test = (hi == 0) and (wi == 0) and (ti == 0);
      // test = test or ((hi == 1) and (wi == 0) and (ti == 0));
      // test = test or ((hi == 0) and (wi == 1) and (ti == 0));
      // test = test or ((hi == 1) and (wi == 1) and (ti == 0));
      // test = test or ((hi == 2) and (wi == 2) and (ti == 0));
      // bool test = (hi < 5) and (wi < 5) and (ti == 0);
      // if (not(test)){ return; }

      // -- non-local index --
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl_patch[_idx] = ref_patch[_idx] + flows_k[ibatch][ihead][ti][hi][wi][ki][_idx];
      }
      check_bounds(valid_patch,nl_patch,T,H,W);
      // valid_patch = valid_patch or full_ws;
      if (not(valid_patch)){ return; }

      // nl_patch[0] = bounds(nl_patch[0],T);
      // nl_patch[1] = bounds(nl_patch[1],H);
      // nl_patch[2] = bounds(nl_patch[2],W);

      // -- search flow from difference --
      int t_max;
      set_time_range(t_max, ti, T, wt);
      int dt = static_cast<int>(nl_patch[0]) - ti;
      int dto = t_max - ti;
      int si = (dt > 0) ? (dt-st_offset) : (dto - dt - st_offset);
      int ws_ti = (wt > 0)  ? (ref_patch[0]+nl_patch[0]) % T : 0;

      // -- offset reference --
      // if (si >= 0){
      //   ref_patch[1] += flows[ibatch][ihead][ti][si][1][hi][wi];
      //   ref_patch[2] += flows[ibatch][ihead][ti][si][0][hi][wi];
      //   ref_patch[1] = bounds(ref_patch[1],H);
      //   ref_patch[2] = bounds(ref_patch[2],W);
      // }

      // -- search region offsets --
      set_search_offsets(wsOff_h, wsOff_w,
                         ref_patch[1], ref_patch[2],
                         stride1, wsHalf0, ws, H, W, full_ws);

      // -- how different from my reference? --
      int li;
      bool oob;
      int time_offset = ws_ti*(ws*ws+2*(wsHalf)*ws+wsHalf*wsHalf);
      get_unique_index(li, oob, nl_patch[1], nl_patch[2],
                       ref_patch[1], ref_patch[2], 
                       wsOff_h, wsOff_w, time_offset,
                       stride0, stride1, ws, wsHalf, full_ws);

      // -- assign to sparse matrix --
      // if (not(oob)){ return; }

      // -- check --
      // assert(ws_i >= 0);
      // assert(ws_j >= 0);
      // assert(ws_i <= (ws-1));
      // assert(ws_j <= (ws-1));
      // if (oob){
      //   assert(ws_i <= wsHalf);
      //   assert(ws_j <= wsHalf);
      // }
      // assert(nl_patch[1] == (ref_patch[1] + stride1*(ws_i_orig-wsOff_h)));
      // assert(nl_patch[2] == (ref_patch[2] + stride1*(ws_j_orig-wsOff_w)));
      assert(li >= 0); // allow invalid at "-1"
      assert(li <= (S-1));

      // -- assigns --
      // if (li < 0){ continue; }
      names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][0] = qi;
      names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][1] = ki;
      // atomicAdd(&(names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][0]),1);
      // atomicAdd(&(names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][1]),ki);


    }
}


__global__ void scatter_labels_norm_kernel(
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> names,
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> labels){

    // -- unpack --
    int B = names.size(0);
    int HD = names.size(1);
    int S = names.size(2);
    int T = names.size(3);
    int H = names.size(4);
    int W = names.size(5);

    // -- derived --
    int HW = H*W;
    int Q = T*HW;

    // -- indexing variables --
    int nl_patch[3];
  
    // -- location to fill --
    int nl_i = blockIdx.x*blockDim.x+threadIdx.x;
    int ihead = blockIdx.y/B;
    int ibatch = (blockIdx.y-ihead*B);
    if (nl_i > (Q-1)){ return; } // skip invalid

    // -- reference index --
    get_pixel_loc(nl_patch,nl_i,1,W,HW,H,W);

    // -- assign valid names to their (qi,ki) home --
    int ci = 0;
    for (int si=0; si < S; si++){

      // -- read names --
      int qi = names[ibatch][ihead][si][nl_patch[0]][nl_patch[1]][nl_patch[2]][0];
      int ki = names[ibatch][ihead][si][nl_patch[0]][nl_patch[1]][nl_patch[2]][1];

      // -- skip if not filled --
      if ((qi == -1) or (ki == -1)){ continue; }
      labels[ibatch][ihead][qi][ki] = ci;
      ci += 1;

    }
}

void scatter_labels_cuda(
    const torch::Tensor flows, const torch::Tensor flows_k,
    torch::Tensor labels, torch::Tensor names,
    int ws, int wt, int stride0, float stride1, bool full_ws){

  // -- sizes --
  int B = flows.size(0);
  int HD = flows.size(1);
  int T = flows.size(2);
  // int W_t' = flows.size(3);
  int nH = flows.size(5);
  int nW = flows.size(6);
  int K = flows_k.size(5);
  int Q = T*nH*nW;
  // fprintf(stdout,"Q,K: %d,%d\n",Q,K);

  // -- compute st --
  int W_t = 2*wt+1;
  int st_offset = W_t - flows.size(3);
  assert((st_offset == 1) or (st_offset == 0));

  // -- launch parameters --
  dim3 threadsPerBlock(156,4);
  dim3 blocksPerGrid(1, 1, HD*B);
  blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(K)/double(threadsPerBlock.y));

  // -- launch kernel --
  scatter_labels_kernel<<<blocksPerGrid, threadsPerBlock>>>(
           flows.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
           flows_k.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
           names.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
           ws, wt, stride0, stride1, full_ws, st_offset);

  // -- launch parameters --
  int H = names.size(4);
  int W = names.size(5);
  Q = T*H*W; // all pixels, not query coordinates
  dim3 threadsPerBlock_norm(960);
  dim3 blocksPerGrid_norm(1, B*HD);
  blocksPerGrid_norm.x = ceil(double(Q)/double(threadsPerBlock.x));

  // -- launch kernel --
  scatter_labels_norm_kernel<<<blocksPerGrid_norm, threadsPerBlock_norm>>>(
     names.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
     labels.packed_accessor32<int,4,torch::RestrictPtrTraits>());

}

