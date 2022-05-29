
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/****************************

       Forward Pass

****************************/

__inline__ __device__ int bounds(int val, int lim ){
  if (val < 0){
    val = -val - 1;
  }else if (val >= lim){
    val = 2*lim - val - 1;
  }
  return val;
}
// #define ROUND_2_INT(f) ((int)(f >= 0.0 ? (f + 0.5) : (f - 0.5)))
#define ROUND_2_INT(f) ((int)(f + 0.49999))

// __inline__ __device__ double warpSum(double tot){
//   unsigned mask = 0xffff;
//   for(int stride=16;stride%lt;0;stride/=2){
//     tot += __shfl_down_sync(mask,tot,stride);
//     mask /= 2;
//   }
//   return tot;
// }

template <typename scalar_t>
__global__ void dnls_search_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> queryInds,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fflow,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> nlDists,
    torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> nlInds,
    int ps, int pt, int ws, int wt, int chnls, int dilation, int stride,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bufs,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> tranges,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> n_tranges,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> min_tranges,
    int ws_iters, int bpb){

  // shapes
  int nframes,color,h,w,height,width;
  nframes = vid.size(0);
  color = vid.size(1);
  h = vid.size(2);
  w = vid.size(3);
  height = h;
  width = w;

  // offsets
  int psHalf = (ps-1)/2;
  int wsHalf = (ws-1)/2;
  int numQueries = queryInds.size(0);

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
  float dist,v_pix,n_pix;

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
    //     searching loop for (ti,top,left)
    // ---------------------------------------

    // -- we loop over search space if needed --
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
            cw0 = bufs[bidx][0][dtd][ws_i][ws_j];
            ch0 = bufs[bidx][1][dtd][ws_i][ws_j];
            ct0 = bufs[bidx][2][dtd][ws_i][ws_j];

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
            cw_i = ROUND_2_INT(cw_f);
            ch_i = ROUND_2_INT(ch_f);

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
          n_hi = ch + stride * (ws_i - wsHalf);
          n_wi = cw + stride * (ws_j - wsHalf);

          // ---------------------------
          //      valid (search "n")
          // ---------------------------
          valid_n_ti = (n_ti < nframes) && (n_ti >= 0);
          valid_n_hi = (n_hi < height) && (n_hi >= 0);
          valid_n_wi = (n_wi < width) && (n_wi >= 0);
          valid_n = valid_n_ti && valid_n_hi && valid_n_wi;
          valid = valid_n && valid_anchor;
          if (!valid){ dist = 100000; }

          // ---------------------------------
          //
          //  compute delta over patch vol.
          //
          // ---------------------------------
          for (int pk = 0; pk < pt; pk++){
            for (int pi = 0; pi < ps; pi++){
              for (int pj = 0; pj < ps; pj++){
                
                // -- inside entire image --
                vH = bounds(hi + dilation*(pi - psHalf),height);
                vW = bounds(wi + dilation*(pj - psHalf),width);
                vT = ti + pk;

                nH = bounds(n_hi + dilation*(pi - psHalf),height);
                nW = bounds(n_wi + dilation*(pj - psHalf),width);
                nT = n_ti + pk;

                // -- valid checks [for testing w/ zero pads] --
                vvalid = (vH < height and vH >= 0);
                vvalid = vvalid and (vW < width and vW >= 0);
                vvalid = vvalid and (vT < nframes and vT >= 0);

                nvalid = (nH < height and nH >= 0);
                nvalid = nvalid and (nW < width and nW >= 0);
                nvalid = nvalid and (nT < nframes and nT >= 0);

                // -- all channels --
                for (int ci = 0; ci < chnls; ci++){

                    // -- get data --
                  if (vvalid){
                    v_pix = vid[vT][ci][vH][vW];
                  }else{
                    v_pix = 0.;
                  }
                  if (nvalid){
                    n_pix = vid[nT][ci][nH][nW];
                  }else{
                    n_pix = 0;
                  }

                  // -- compute dist --
                  if (valid){
                    dist += std::pow((v_pix - n_pix),2);
                  }
                }
              }
            }
          }

          // -- dists --
          nlDists[bidx][wt_k][ws_i][ws_j] = dist;

          // -- inds --
          nlInds[bidx][wt_k][ws_i][ws_j][0] = n_ti;
          nlInds[bidx][wt_k][ws_i][ws_j][1] = n_hi;
          nlInds[bidx][wt_k][ws_i][ws_j][2] = n_wi;

          // -- final check [put self@index 0] --
          eq_ti = n_ti == ti;
          eq_hi = n_hi == hi;
          eq_wi = n_wi == wi;
          eq_dim = eq_ti && eq_hi && eq_wi;
          dist = nlDists[bidx][wt_k][ws_i][ws_j];
          if (eq_dim){
            nlDists[bidx][wt_k][ws_i][ws_j] = -100;
          }

        }
      }
    }
  }
}

void dnls_cuda_search_forward(
    torch::Tensor vid, torch::Tensor queryInds,
    torch::Tensor fflow, torch::Tensor bflow, torch::Tensor nlDists, torch::Tensor nlInds,
    int ps, int pt, int ws, int wt, int chnls, int dilation, int stride,
    torch::Tensor bufs, torch::Tensor tranges, torch::Tensor n_tranges,
    torch::Tensor min_tranges){

    // # -- launch params --
    // w_threads = min(ws,32)
    // nthreads = (w_threads,w_threads)
    // ws_iters = (ws-1)//w_threads + 1
    // nblocks = (nq-1)//batches_per_block+1

   // launch params 
   int bpb = 10;
   int numQueries = queryInds.size(0);
   int w_threads = min(ws,32);
   int ws_iters = ((ws-1)/w_threads) + 1;
   fprintf(stdout,"w_threads: %d\n",w_threads);
   fprintf(stdout,"ws_iters: %d\n",ws_iters);
   dim3 nthreads(w_threads,w_threads);
   int nblocks = ((numQueries - 1) / bpb) + 1;
    
   // launch kernel
   AT_DISPATCH_FLOATING_TYPES(vid.type(), "dnls_search_forward_kernel", ([&] {
      dnls_search_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        queryInds.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        fflow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        bflow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nlDists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nlInds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        ps, pt, ws, wt, chnls, dilation, stride,
        bufs.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
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
__global__ void dnls_search_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> nlDists,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> nlInds,
    int ps, int pt, float lam, int bpb) {

  // shape
  int nq = nlDists.size(0);
  int k =  nlDists.size(1);
  int color = vid.size(1);
  int ti,hi,wi;
  float weight;

  // get indices
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;
  int index = tidx + bidx * blockDim.x;

  if (bidx < nq){
    // iterate
    for (int ki = 0; ki < k; ki++){
      for (int pk = 0; pk < ps; pk++){
        for (int pi = 0; pi < ps; pi++){
          for (int pj = 0; pj < ps; pj++){
            ti = nlInds[bidx][ki][0];
            hi = nlInds[bidx][ki][1];
            wi = nlInds[bidx][ki][2];
            for (int ci = 0; ci < color; ci++){
              vid[ti][ci][hi][wi] += nlDists[bidx][ki];
            }
          }
        }
      }
    }
  }

}

void dnls_cuda_search_backward(
    torch::Tensor vid, torch::Tensor nlDists, torch::Tensor nlInds,
    int ps, int pt, float lam) {

  // launch params
  int numQueries = nlInds.size(0);
  int k = nlDists.size(1);
  assert(pt == 1);

  int bpb = 10;
  int nthreads = 512;
  int num_per_block = nthreads * bpb;
  int nblocks = ((numQueries - 1) / num_per_block) + 1;

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "dnls_search_backward_kernel", ([&] {
    dnls_search_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nlDists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        nlInds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        ps,pt,lam,bpb);
  }));

}
