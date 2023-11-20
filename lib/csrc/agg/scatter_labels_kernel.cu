/*

  Stack non-local patches into a video

*/

#include "scatter_int.cu"


/**************************************

          Get Scatter Labels

**************************************/

template<typename itype=int>
__device__ __forceinline__
void handle_oob(int& ws_i, int& ws_j, bool& oob, int* nl_patch,
                int* ref_patch, int stride1, int ws, int wsHalf,
                int wsOff_h, int wsOff_w, bool full_ws){

  // -- check spatial coordinates --
  int ws_i_tmp;
  int ws_j_tmp;
  ws_i_tmp = (nl_patch[1] - ref_patch[1])/stride1 + wsHalf;
  ws_j_tmp = (nl_patch[2] - ref_patch[2])/stride1 + wsHalf;
  // ws_i_tmp = ws_i - wsOff_h + wsHalf;
  // ws_j_tmp = ws_j - wsOff_w + wsHalf;
  assert(ws_i_tmp >= -wsHalf);
  assert(ws_j_tmp >= -wsHalf);
  assert(ws_i_tmp < ws+wsHalf);
  assert(ws_j_tmp < ws+wsHalf);

  // -- check offset --
  int delta_h = wsHalf - wsOff_h;
  int delta_w = wsHalf - wsOff_w;

  // int delta_i,delta_j;
  int delta_i = delta_h > 0 ? (ws-1-ws_i) : ((delta_h < 0) ? ws_i : ws);
  bool oob_i = (delta_h != 0) and (abs(delta_h) >= delta_i);
  int delta_j = delta_w > 0 ? (ws-1-ws_j) : ((delta_w < 0) ? ws_j : ws);
  bool oob_j = (delta_w != 0) and (abs(delta_w) >= delta_j);

  // -- new idea --
  int dH = abs(nl_patch[1] - ref_patch[1]);
  int dW = abs(nl_patch[2] - ref_patch[2]);

  // -- check oob --
  oob = (oob_i or oob_j) and full_ws;
  ws_i = oob ? ws_i_tmp % ws : ws_i;
  ws_j = oob ? ws_j_tmp % ws : ws_j;

  // ws_i = oob_i ? (delta_i) : ws_i_tmp;
  // ws_i = oob_j ? (ws_i % wsHalf) : ws_i;

  // ws_j = oob_j ? (delta_j) : ws_j_tmp;
  // ws_j = oob_i ? (ws_j % wsHalf) : ws_j;

  // ws_i = oob_i ? (wsOff_h) : (oob_j ? ws_i_tmp % wsHalf : ws_i_tmp);
  // ws_j = oob_j ? (wsOff_w) : (oob_i ? ws_j_tmp % wsHalf : ws_j_tmp);
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

    // -- derived --
    int nHW = nH*nW;
    int Q = T*nHW;
    int H = nH*stride0;
    int W = nW*stride0;

    // -- indexing variables --
    int ref_patch[3];
    int nl_patch[3];
    bool valid_patch;
  
    // -- search window params --
    int wsHalf = (ws-1)/2;
    int wsOff_h = wsHalf;
    int wsOff_w = wsHalf;
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
      int h_ref = ref_patch[1];
      int w_ref = ref_patch[2];
      int hi = ref_patch[1]/stride0;
      int wi = ref_patch[2]/stride0;

      // bool test = (hi == 0) and (wi == 0) and (ti == 0);
      // test = test or ((hi == 1) and (wi == 0) and (ti == 0));
      // test = test or ((hi == 0) and (wi == 1) and (ti == 0));
      // test = test or ((hi == 1) and (wi == 1) and (ti == 0));
      // test = test or ((hi == 2) and (wi == 2) and (ti == 0));
      bool test = (hi < 5) and (wi < 5) and (ti == 0);
      if (not(test)){ return; }

      // -- non-local index --
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl_patch[_idx] = ref_patch[_idx] + flows_k[ibatch][ihead][ti][hi][wi][ki][_idx];
      }
      check_bounds(valid_patch,nl_patch,T,H,W);
      valid_patch = valid_patch or full_ws;

      // nl_patch[0] = bounds(nl_patch[0],T);
      // nl_patch[1] = bounds(nl_patch[1],H);
      // nl_patch[2] = bounds(nl_patch[2],W);

      // -- search flow from difference --
      int t_max;
      set_time_range(t_max, ti, T, wt);
      int dt = static_cast<int>(nl_patch[0]) - ti;
      int dto = t_max - ti;
      int si = (dt > 0) ? (dt-st_offset) : (dto - dt - st_offset);

      // -- offset reference --
      // if (si >= 0){
      //   ref_patch[1] += flows[ibatch][ihead][ti][si][1][hi][wi];
      //   ref_patch[2] += flows[ibatch][ihead][ti][si][0][hi][wi];
      //   ref_patch[1] = bounds(ref_patch[1],H);
      //   ref_patch[2] = bounds(ref_patch[2],W);
      // }

      // -- shift (ws_i,ws_j) to handle out of bounds --
      int ws_i,ws_j;

      // -- search region offsets --
      set_search_offsets(wsOff_h, wsOff_w,
                         ref_patch[1], ref_patch[2],
                         stride1, wsHalf, ws, H, W, full_ws);
      // set_search_offsets(wsOff_h_nl, wsOff_w_nl,
      //                    nl_patch[1], nl_patch[2],
      //                    stride1, wsHalf, ws, H, W, full_ws);

      ws_i = (nl_patch[1] - ref_patch[1])/stride1 + wsOff_h;
      ws_j = (nl_patch[2] - ref_patch[2])/stride1 + wsOff_w;
      int ws_i_orig = ws_i;
      int ws_j_orig = ws_j;

      // -- how different from my reference? --
      bool oob;
      int _ws_i,_ws_j;
      handle_oob(ws_i, ws_j, oob, nl_patch, ref_patch,
                 stride1, ws, wsHalf, wsOff_h, wsOff_w, full_ws);

      // -- compute the assigned index --
      // int li_num = oob ? wsHalf : ws;
      int li = (ws_i) + (ws_j)*ws;// + (si+st_offset)*ws_peak;
      int li_off = (ws_i_orig < ws_j_orig) ? ws-1 : 0;
      li = oob ? li + ws*ws + li_off : li;

      // -- assign to sparse matrix --
      if (not(oob)){ return; }
      if (not(valid_patch)){ return; }

      // -- check --
      assert(ws_i >= 0);
      assert(ws_j >= 0);
      assert(ws_i <= (ws-1));
      assert(ws_j <= (ws-1));
      // if (oob){
      //   assert(ws_i <= wsHalf);
      //   assert(ws_j <= wsHalf);
      // }
      assert(nl_patch[1] == (ref_patch[1] + stride1*(ws_i_orig-wsOff_h)));
      assert(nl_patch[2] == (ref_patch[2] + stride1*(ws_j_orig-wsOff_w)));
      assert(li >= 0);
      assert(li <= (S-1));

      // -- assigns --
      // names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][0] = qi;
      // names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][1] = ki;
      atomicAdd(&(names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][0]),1);
      atomicAdd(&(names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][1]),ws_i);
      // names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][1] = ki;


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

    // -- assign valid names to their (qi,ki) home --
    int ci = 0;
    for (int si=0; si < S; si++){

      // -- reference index --
      get_pixel_loc(nl_patch,nl_i,1,W,HW,H,W);

      // -- read names --
      int qi = names[ibatch][ihead][si][nl_patch[0]][nl_patch[1]][nl_patch[2]][0];
      int ki = names[ibatch][ihead][si][nl_patch[0]][nl_patch[1]][nl_patch[2]][1];

      // -- skip if not filled --
      if ((qi == -1) or (ki == -1)){ continue; }
      // atomicAdd(&(labels[ibatch][ihead][qi][ki]),1);
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
  fprintf(stdout,"Q,K: %d,%d\n",Q,K);

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
  // scatter_labels_norm_kernel<<<blocksPerGrid_norm, threadsPerBlock_norm>>>(
  //    names.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
  //    labels.packed_accessor32<int,4,torch::RestrictPtrTraits>());

}

