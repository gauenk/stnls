
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/****************************

       Forward Pass

****************************/


__inline__ __device__ int bounds(int val, int lim ){
  int vval = val;
  if (val < 0){
    vval = -val;
  }else if (val >= lim){
    vval = 2*(lim-1) - val;
  }
  return vval;
}

// __inline__ __device__ double warpSum(double tot){
//   unsigned mask = 0xffff;
//   for(int stride=16;stride%lt;0;stride/=2){
//     tot += __shfl_down_sync(mask,tot,stride);
//     mask /= 2;
//   }
//   return tot;
// }

template <typename scalar_t>
__global__ void dnls_xsearch_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid0,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid1,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> queryInds,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fflow,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> nlDists,
    torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> nlInds,
    int ps, int pt, int ws, int wt, int chnls, int stride, int dilation, 
    bool use_search_abs, bool use_bounds, bool use_adj,
    int oh0, int ow0, int oh1, int ow1,
    torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> bufs,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> tranges,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> n_tranges,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> min_tranges,
    int ws_iters, int bpb){

  // shapes
  float nan = __int_as_float(0xffe00000);
  float inf = __int_as_float(0x7f800000);
   int nframes,color,h,w,height,width;
  nframes = vid0.size(0);
  color = vid0.size(1);
  h = vid0.size(2);
  w = vid0.size(3);
  height = h;
  width = w;

  // offsets
  int psHalf = ps/2;
  int wsHalf = (ws-1)/2;
  int numQueries = queryInds.size(0);
  // int adj = use_adj ? psHalf : 0;
  int adj = psHalf;

  // column index
  int blkDimX = blockDim.x; // num threads in x-block
  int blkDimY = blockDim.y; // num threads in y-block
  int cu_tidX = threadIdx.x;
  int cu_tidY = threadIdx.y;
  int block_start = blockIdx.x*bpb;
  int bidx,ws_i,ws_j,dtd;

  // decls
  int ti,hi,wi;
  int n_ti,n_hi,n_wi;
  int vH,vW,vT,nH,nW,nT;
  bool valid,vvalid,nvalid;
  bool valid_ti,valid_hi,valid_wi,valid_anchor;
  bool valid_n_ti,valid_n_hi,valid_n_wi,valid_n;
  bool eq_ti,eq_hi,eq_wi,eq_dim;

  float cw0,ch0,ct0,cw_f,ch_f;
  int l_cw0,l_ch0,l_ct0;
  int cw_i,ch_i,ch,cw,ct;
  float v_pix,n_pix;
  double _dist,dist;

  for (int _bidx = 0; _bidx < bpb; _bidx++){

    //---------------------------
    //   extract anchor pixel
    //---------------------------

    // -- block start --
    bidx = block_start + _bidx;
    if (bidx >= numQueries){ continue; }

    // -- unpack pixel locs --
    ti = queryInds[bidx][0];
    hi = queryInds[bidx][1];
    wi = queryInds[bidx][2];

    // -- valid (anchor pixel) --
    valid_ti = (ti < nframes) && (ti >= 0);
    valid_hi = (hi < height) && (hi >= 0);
    valid_wi = (wi < width) && (wi >= 0);
    valid_anchor = valid_ti && valid_hi && valid_wi;

    // ---------------------------------------
    //     xsearching loop for (ti,top,left)
    // ---------------------------------------

    // -- we loop over xsearch space if needed --
    for (int _xi = 0; _xi < ws_iters; _xi++){

      int ws_i = cu_tidX + blkDimX*_xi;
      if (ws_i >= ws){ continue; }

      for (int _yi = 0; _yi < ws_iters; _yi++){
        ws_j = cu_tidY + blkDimY*_yi;
        if (ws_j >= ws){ continue; }

        for( int wt_k = 0; wt_k < n_tranges[ti]; wt_k++){
          int n_ti = tranges[ti][wt_k];
          int dt = n_ti - min_tranges[ti];

          // ------------------------
          //      init direction
          // ------------------------
          int direction = max(-1,min(1,n_ti - ti));
          if (direction != 0){

            // -- get offset at index --
            int dtd = int(dt-direction);
            cw0 = 1.*bufs[bidx][0][dtd][ws_i][ws_j];
            ch0 = 1.*bufs[bidx][1][dtd][ws_i][ws_j];
            ct0 = 1.*bufs[bidx][2][dtd][ws_i][ws_j];

            // -- legalize access --
            l_cw0 = int(max(0,min(w-1,int(cw0))));
            l_ch0 = int(max(0,min(h-1,int(ch0))));
            l_ct0 = int(max(0,min(nframes-1,int(ct0))));

            // -- access flows --
            if (direction > 0 ){
              cw_f = cw0 + fflow[l_ct0][0][l_ch0][l_cw0];
              ch_f = ch0 + fflow[l_ct0][1][l_ch0][l_cw0];
            }else{
              cw_f = cw0 + bflow[l_ct0][0][l_ch0][l_cw0];
              ch_f = ch0 + bflow[l_ct0][1][l_ch0][l_cw0];
            }
            cw_i = int(cw_f+0.5);
            ch_i = int(ch_f+0.5);

            // -- rounding --
            cw = max(0,min(width-1,cw_i));
            ch = max(0,min(height-1,ch_i));
            ct = n_ti;

          }else{
            cw = wi;
            ch = hi;
            ct = ti;
          }

          // ----------------
          //     update
          // ----------------
          bufs[bidx][0][dt][ws_i][ws_j] = cw;
          bufs[bidx][1][dt][ws_i][ws_j] = ch;
          bufs[bidx][2][dt][ws_i][ws_j] = ct;

          // --------------------
          //      init dists
          // --------------------
          dist = 0;

          // -----------------
          //    spatial dir
          // -----------------
          if (use_search_abs){
            n_hi = stride * (ws_i - 0);
            n_wi = stride * (ws_j - 0);
          }else{
            n_hi = ch + stride * (ws_i - wsHalf);
            n_wi = cw + stride * (ws_j - wsHalf);
          }
          // n_hi = stride * ws_i;
          // n_wi = stride * ws_j;

          // ---------------------------
          //      valid (xsearch "n")
          // ---------------------------

          valid_n_ti = (n_ti < nframes) && (n_ti >= 0);
          valid_n_hi = (n_hi < height) && (n_hi >= 0);
          valid_n_wi = (n_wi < width) && (n_wi >= 0);
          valid_n = valid_n_ti && valid_n_hi && valid_n_wi;
          valid = valid_n && valid_anchor;
          // valid = valid_anchor;

          // ---------------------------------
          //
          //  compute delta over patch vol.
          //
          // ---------------------------------
          for (int pk = 0; pk < pt; pk++){
            for (int pi = 0; pi < ps; pi++){
              for (int pj = 0; pj < ps; pj++){
                
                // -- inside entire image --
                vH = (hi-oh0) + dilation*(pi - psHalf + adj);
                vW = (wi-ow0) + dilation*(pj - psHalf + adj);
                vH = use_bounds ? bounds(vH,height) : vH;
                vW = use_bounds ? bounds(vW,width) : vW;
                vT = bounds(ti + pk,nframes);

                nH = (n_hi-oh1) + dilation*(pi - psHalf + adj);
                nW = (n_wi-ow1) + dilation*(pj - psHalf + adj);
                nH = use_bounds ? bounds(nH,height) : nH;
                nW = use_bounds ? bounds(nW,height) : nW;
                nT = bounds(n_ti + pk,nframes);

                // -- valid checks [for testing w/ zero pads] --
                vvalid = (vH < height) && (vH >= 0);
                vvalid = vvalid && (vW < width) && (vW >= 0);
                vvalid = vvalid && (vT < nframes) && (vT >= 0);

                nvalid = (nH < height) && (nH >= 0);
                nvalid = nvalid && (nW < width) && (nW >= 0);
                nvalid = nvalid && (nT < nframes) && (nT >= 0);

                // -- all channels --
                for (int ci = 0; ci < chnls; ci++){

                  // -- get data --
                  if (vvalid){
                    v_pix = vid0[vT][ci][vH][vW];
                  }else{
                    v_pix = 0.;
                  }

                  if (nvalid){
                    n_pix = vid1[nT][ci][nH][nW];
                  }else{
                    n_pix = 0.;
                  }

                  // -- compute dist --
                  if (valid){
                    _dist = v_pix * n_pix;
                    dist += _dist;
                  }
                }
              }
            }
          }

          // -- dists --
          if (!valid){ dist = nan; }
          nlDists[bidx][wt_k][ws_i][ws_j] = dist;

          // -- inds --
          nlInds[bidx][wt_k][ws_i][ws_j][0] = n_ti;
          nlInds[bidx][wt_k][ws_i][ws_j][1] = n_hi;
          nlInds[bidx][wt_k][ws_i][ws_j][2] = n_wi;

          // -- final check [put self@index 0] --
          // eq_ti = n_ti == ti;
          // eq_hi = n_hi == hi;
          // eq_wi = n_wi == wi;
          // eq_dim = eq_ti && eq_hi && eq_wi;
          // dist = nlDists[bidx][wt_k][ws_i][ws_j];
          // if (eq_dim){
          //   nlDists[bidx][wt_k][ws_i][ws_j] = inf;
          // }

        }
      }
    }
  }
}

void dnls_cuda_xsearch_forward(
    torch::Tensor vid0, torch::Tensor vid1, torch::Tensor queryInds,
    torch::Tensor fflow, torch::Tensor bflow, torch::Tensor nlDists, torch::Tensor nlInds,
    int ps, int pt, int ws, int wt, int chnls, int stride, int dilation,
    bool use_search_abs, bool use_bounds, bool use_adj,
    int oh0, int ow0, int oh1, int ow1,
    torch::Tensor bufs, torch::Tensor tranges, torch::Tensor n_tranges,
    torch::Tensor min_tranges){

    // # -- launch params --
    // w_threads = min(ws,32)
    // nthreads = (w_threads,w_threads)
    // ws_iters = (ws-1)//w_threads + 1
    // nblocks = (nq-1)//batches_per_block+1
    // fprintf(stdout,"use_search_abs, bool use_bounds, bool use_adj: %d,%d,%d\n",use_search_abs, use_bounds, use_adj);
    // fprintf(stdout,"oh0, ow0, oh1, ow1: %d,%d,%d,%d\n",oh0, ow0, oh1, ow1);
    // fprintf(stdout,"stride, dilation: %d,%d\n",stride, dilation);

    // launch params 
    int bpb = 32;
    int numQueries = queryInds.size(0);
    int w_threads = std::min(ws,32);
    int ws_iters = ((ws-1)/w_threads) + 1;
    dim3 nthreads(w_threads,w_threads);
    int nblocks = ((numQueries - 1) / bpb) + 1;
    // fprintf(stdout,"nblocks,w_threads: %d,%d\n",nblocks,w_threads);
     
    // launch kernel
    AT_DISPATCH_FLOATING_TYPES(vid0.type(), "dnls_xsearch_forward_kernel", ([&] {
       dnls_xsearch_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
         vid0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
         vid1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
         queryInds.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
         fflow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
         bflow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
         nlDists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
         nlInds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
         ps, pt, ws, wt, chnls, stride, dilation, 
         use_search_abs, use_bounds, use_adj,
         oh0, ow0, oh1, ow1,
         bufs.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
         tranges.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
         n_tranges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
         min_tranges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
         ws_iters, bpb);
       }));
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t>
__global__ void dnls_xsearch_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid0_grad,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid1_grad,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid0,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid1,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> qinds,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> nlDists,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> nlInds,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int oh0, int ow0, int oh1, int ow1,
    int ps, int pt, float lam, bool use_bounds,
    int bpb, int npt, int cpt) {

  // shape
  int colors = vid0.size(1);
  int nframes = vid0.size(0);
  int height = vid0.size(2);
  int width = vid0.size(3);
  int ti,hi,wi,tk,hk,wk;
  int tk_a,hk_a,wk_a;
  float weight;

  // limits
  int i0_max = nlInds.size(0);
  int i1_max = nlInds.size(1);

  // get indices
  int i0_start = blockIdx.x * bpb;
  int i1_start = threadIdx.x * npt;
  int c0_start = threadIdx.y * cpt;

  // get block limits
  int i0_end = min(i0_start + bpb,i0_max);
  int i1_end = min(i1_start + npt,i1_max);
  int c0_end = min(c0_start + cpt,colors);
  int c0 = 0;
  int c0_dist = c0_end - c0_start;
  int c0_offset = 0;

  // misc
  // int oh0,ow0,oh1,ow1;
  int psHalf = ps/2;
  int adj = psHalf;
  int dilation = 1;
  int tj,hj,wj;
  bool valid_hj,valid_wj,valid_j;
  bool valid_hk,valid_wk,valid_k;
  bool valid;
  float pix0,pix1;

  // iterate over regions
  for (int i0=i0_start; i0 < i0_end; i0++){

    tk_a = qinds[i0][0];
    hk_a = qinds[i0][1];
    wk_a = qinds[i0][2];
    c0_offset = __float2int_rd(c0_dist * rand_nums[i0][0][0]);

    // k neighbors
    for (int i1=i1_start; i1 < i1_end; i1++){
      c0_offset = (c0_offset + 1) % c0_dist;
      ti = nlInds[i0][i1][0];
      hi = nlInds[i0][i1][1];
      wi = nlInds[i0][i1][2];
      weight = nlDists[i0][i1];

      for (int pk = 0; pk < pt; pk++){
        for (int pi = 0; pi < ps; pi++){
          for (int pj = 0; pj < ps; pj++){
            
            // -- anchor patch --
            if (use_bounds){
              hk = bounds((hk_a-oh0) + dilation*(pi - psHalf + adj),height);
              wk = bounds((wk_a-ow0) + dilation*(pj - psHalf + adj),width);
            }else{
              hk = (hk_a-oh0) + dilation*(pi - psHalf + adj);
              wk = (wk_a-ow0) + dilation*(pj - psHalf + adj);
            }
            tk = bounds(tk_a+pk,nframes);

            // current image
            if (use_bounds){
              hj = bounds((hi-oh1) + dilation*(pi - psHalf + adj),height);
              wj = bounds((wi-ow1) + dilation*(pj - psHalf + adj),width);
            }else{
              hj = (hi-oh1) + dilation*(pi - psHalf + adj);
              wj = (wi-ow1) + dilation*(pj - psHalf + adj);
            }
            tj = bounds(ti+pk,nframes);

            valid_hj = (hj >= 0) && (hj < height);
            valid_wj = (wj >= 0) && (wj < width);
            valid_j = valid_hj && valid_wj;

            valid_hk = (hk >= 0) && (hk < height);
            valid_wk = (wk >= 0) && (wk < width);
            valid_k = valid_hk && valid_wk;

            valid = valid_j && valid_k;
            for (int _c0 = c0_start; _c0 < c0_end; _c0++){
              c0 = (_c0 + c0_offset) % c0_dist + c0_start;
              if (valid){
                pix0 = vid0[tk][c0][hk][wk] * weight;
                pix1 = vid1[tj][c0][hj][wj] * weight;
                vid1_grad[tj][c0][hj][wj] += pix0;
                vid0_grad[tk][c0][hk][wk] += pix1;
              }
            }

          }
        }
      }
    }
  }
}

void dnls_cuda_xsearch_backward(
    torch::Tensor vid0_grad, torch::Tensor vid1_grad,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor qinds, torch::Tensor nlDists, torch::Tensor nlInds,
    int oh0, int ow0, int oh1, int ow1,
    int ps, int pt, float lam, bool use_bounds, bool exact) {

  // unpack
  int num0 = nlInds.size(0);
  int num1 = nlInds.size(1);
  int colors = vid0.size(1);

  // launch params
  int max_nblocks = 8;
  int bpb = 16;//max(4,(num0-1)/max_nblocks+1);
  int nblocks = (num0-1)/bpb+1;
  if (exact){
    bpb = num0;
    nblocks = 1;
  }

  // launch threads
  int tdim0 = min(16*16,num1);
  tdim0 = exact ? 1 : tdim0;
  int tdim1 = exact ? colors : 1;
  dim3 nthreads(tdim0,tdim1);
  int npt = (num1-1)/tdim0+1;
  int cpt = (colors-1)/tdim1+1;

  // fprintf(stdout,"num0,num1: %d,%d\n",num0,num1);
  // fprintf(stdout,"nblocks,tdim0,tdim1: %d,%d,%d\n",
  //         nblocks,tdim0,tdim1);
  // fprintf(stdout,"bpb,npt,cpt: %d,%d,%d\n",
  //         bpb,npt,cpt);


  // -- allocate random memory --
  auto cu_index = vid0_grad.device().index();
  auto options = torch::TensorOptions().device(torch::kCUDA, cu_index).dtype(torch::kFloat32);
  torch::Tensor rand_nums = torch::rand({num0,1,1},options);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid0.type(), "dnls_xsearch_backward_kernel", ([&] {
    dnls_xsearch_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid0_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        vid1_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        vid0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        vid1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        qinds.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        nlDists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        nlInds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        oh0,ow0,oh1,ow1,ps,pt,lam,use_bounds,bpb,npt,cpt);
  }));

}