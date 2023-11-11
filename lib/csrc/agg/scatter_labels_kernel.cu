/*

  Stack non-local patches into a video

*/

#include "scatter_int.cu"


/**************************************

          Get Scatter Labels

**************************************/

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

    // -- derived --
    int nHW = nH*nW;
    int Q = T*nHW;
    int H = nH*stride0;
    int W = nW*stride0;

    // -- indexing variables --
    int ref_patch[3];
    int nl_patch[3];
  
    // -- search window params --
    int wsOff_h,wsOff_w;
    int wsHalf = (ws-1)/2;

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
      int hi = ref_patch[1]/stride0;
      int wi = ref_patch[2]/stride0;

      // -- non-local index --
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl_patch[_idx] = ref_patch[_idx] + flows_k[ibatch][ihead][ti][hi][wi][ki][_idx];
      }
      nl_patch[0] = bounds(nl_patch[0],T);
      nl_patch[1] = bounds(nl_patch[1],H);
      nl_patch[2] = bounds(nl_patch[2],W);

      // -- search flow from difference --
      int t_max;
      set_time_range(t_max, ti, T, wt);
      int dt = static_cast<int>(nl_patch[0]) - ti;
      int dto = t_max - ti;
      int si = dt > 0 ? (dt-st_offset) : dto - dt - st_offset;

      // -- offset reference --
      if (si >= 0){
        ref_patch[1] += flows[ibatch][ihead][ti][si][1][hi][wi];
        ref_patch[2] += flows[ibatch][ihead][ti][si][0][hi][wi];
        ref_patch[1] = bounds(ref_patch[1],H);
        ref_patch[2] = bounds(ref_patch[2],W);
      }

      // -- search region offsets --
      set_search_offsets(wsOff_h, wsOff_w,
                         ref_patch[1], ref_patch[2],
                         stride1, wsHalf, ws, H, W, full_ws);

      // -- compute proposed location --
      int ws_i = (nl_patch[1] - ref_patch[1])/stride1 + wsOff_h;
      int ws_j = (nl_patch[2] - ref_patch[2])/stride1 + wsOff_w;

      // -- check it --
      // assert(nl_patch[1] == (ref_patch[1] + stride1*(ws_i+wsOff_h)));
      // assert(nl_patch[2] == (ref_patch[2] + stride1*(ws_j+wsOff_h)));

      // -- full rasterized location --
      int li = ws_i + ws_j*ws + si*ws*ws;

      // -- assign to sparse matrix --
      names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][0] = qi;
      names[ibatch][ihead][li][nl_patch[0]][nl_patch[1]][nl_patch[2]][1] = ki;
    }
}


__global__ void scatter_labels_norm_kernel(
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> flows,
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> flows_k,
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> names,
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> labels,
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
    int nl_patch[3];
  
    // -- search window params --
    int wsOff_h,wsOff_w;
    int wsHalf = (ws-1)/2;

    // -- location to fill --
    int nl_i = blockIdx.x*blockDim.x+threadIdx.x;
    int ihead = blockIdx.y/B;
    int ibatch = (blockIdx.y-ihead*B) % B;
  
    // -- each region --
    int ci = 0;
    for (int si=0; si < S; si++){

      // -- reference index --
      get_pixel_loc(nl_patch,nl_i,stride0,nW,nHW,H,W);
      int ti = nl_patch[0];
      int hi = nl_patch[1]/stride0;
      int wi = nl_patch[2]/stride0;

      // -- read names --
      int qi = names[ibatch][ihead][si][nl_patch[0]][nl_patch[1]][nl_patch[2]][0];
      int ki = names[ibatch][ihead][si][nl_patch[0]][nl_patch[1]][nl_patch[2]][1];

      // -- skip if not filled --
      if (qi == -1){ continue; }
      
      // -- fill name --
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
  dim3 threadsPerBlock_norm(960);
  dim3 blocksPerGrid_norm(1, HD*B);
  blocksPerGrid_norm.x = ceil(double(Q)/double(threadsPerBlock.x));

  // -- launch kernel --
  scatter_labels_norm_kernel<<<blocksPerGrid_norm, threadsPerBlock_norm>>>(
     flows.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
     flows_k.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
     names.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
     labels.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
     ws, wt, stride0, stride1, full_ws, st_offset);

}

