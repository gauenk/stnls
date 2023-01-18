
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
__global__ void search_prod_pf_with_index_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid0,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid1,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> fflow,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor64<int,6,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> self_dists,
    int qstart, int stride0, int n_h0, int n_w0,
    int ps, int pt, int ws_h, int ws_w, int wt, int chnls, int stride1, int dilation, 
    bool search_abs, bool use_bounds, bool use_adj, bool full_ws,
    bool anchor_self, bool use_self,
    int h0_off, int w0_off, int h1_off, int w1_off,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> tranges,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> n_tranges,
    int ws_h_iters, int ws_w_iters, int wt_iters, int bpt){

  // shapes
  float nan = __int_as_float(0xffe00000);
  float inf = __int_as_float(0x7f800000);
  int bsize,nframes,color,h,w,height,width;
  bsize = vid0.size(0);
  nframes = vid0.size(1);
  color = vid0.size(2);
  h = vid0.size(3);
  w = vid0.size(4);
  height = h;
  width = w;
  int numQueries = dists.size(1);

  // offsets
  int psHalf = ps/2;
  int wsHalf_h = (ws_h)/2;
  int wsHalf_w = (ws_w)/2;
  int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  int wsMax_w = stride1*(ws_w-1-wsHalf_w);
  int adj = use_adj ? psHalf : 0;
  int n_hw0 = n_h0 * n_w0;

  // column index
  int blkDimX = blockDim.x; // num threads in x-block
  int blkDimY = blockDim.y; // num threads in y-block
  // int bindex = blockDim.z;
  int cu_tidX = threadIdx.x;
  int cu_tidY = threadIdx.y;
  int bindex = blockIdx.x;
  int block_start = blockIdx.y*bpt;
  int bidx,ws_i,ws_j,wt_k;
  int qindex,i_mod;

  // decls
  int ti,hi,wi;
  int n_ti,n_hi,n_wi;
  int vH,vW,vT,nH,nW,nT;
  bool valid,vvalid,nvalid;
  bool vvalid_t,vvalid_h,vvalid_w;
  bool nvalid_t,nvalid_h,nvalid_w;
  bool valid_ti,valid_hi,valid_wi,valid_anchor;
  bool valid_n_ti,valid_n_hi,valid_n_wi,valid_n;
  int wsOff_h,wsOff_w;
  bool eq_dim;
  // bool eq_ti,eq_hi,eq_wi,eq_dim;

  float cw0,ch0,ct0,cw_f,ch_f;
  int l_cw0,l_ch0,l_ct0;
  int cw_i,ch_i,ch,cw,ct;
  float v_pix,n_pix;
  double _dist,dist;

  for (int _bidx = 0; _bidx < bpt; _bidx++){

    //---------------------------
    //   extract anchor pixel
    //---------------------------

    // -- block start --
    bidx = block_start + _bidx;
    if (bidx >= numQueries){ continue; }

    // -- unpack pixel locs --
    qindex = bidx + qstart;
    i_mod = qindex % n_hw0;
    ti = qindex / n_hw0;
    wi = ((i_mod % n_w0) * stride0) % width ;
    hi = ((i_mod / n_w0) * stride0) % height;

    // -- valid (anchor pixel) --
    valid_ti = (ti < nframes) && (ti >= 0);
    valid_hi = (hi < height) && (hi >= 0);
    valid_wi = (wi < width) && (wi >= 0);
    valid_anchor = valid_ti && valid_hi && valid_wi;

    // -- search offset --
    if(full_ws){
      wsOff_h = (hi-max(hi-stride1*wsHalf_h,0))/stride1;
      wsOff_w = (wi-max(wi-stride1*wsHalf_w,0))/stride1;
      if ((hi+wsMax_h) >= height){
        wsOff_h+=(hi+wsMax_h-min(hi+stride1*wsMax_h,height-1)-1)/stride1 + 1;
      }
      if ((wi+wsMax_w) >= width){
        wsOff_w+=(wi+wsMax_w-min(wi+stride1*wsMax_w,width-1)-1)/stride1 + 1;
      }
    }else{
      wsOff_h = wsHalf_h;
      wsOff_w = wsHalf_w;
    }

    // ---------------------------------------
    //     xsearching loop for (ti,top,left)
    // ---------------------------------------
    for( int _wt_k = 0; _wt_k < wt_iters; _wt_k++){
      wt_k = threadIdx.z + blockDim.z*_wt_k;
      if (wt_k > n_tranges[ti]){ continue; }
      int n_ti = tranges[ti][wt_k];

      // ------------------------
      //      init direction
      // ------------------------

      // -- init direction --
      int direction = max(-1,min(1,n_ti - ti));

      // -- compute optical flow --
      if (direction != 0){

        // -- access flows --
        if (direction > 0 ){
          cw_f = cw0 + fflow[bindex][wt_k][ti][0][hi][wi];
          ch_f = ch0 + fflow[bindex][wt_k][ti][1][hi][wi];
        }else{
          cw_f = cw0 + bflow[bindex][wt_k][ti][0][hi][wi];
          ch_f = ch0 + bflow[bindex][wt_k][ti][1][hi][wi];
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

      // -- we loop over xsearch space if needed --
      for (int _xi = 0; _xi < ws_h_iters; _xi++){
  
        int ws_i = cu_tidX + blkDimX*_xi;
        if (ws_i >= ws_h){ continue; }
  
        for (int _yi = 0; _yi < ws_w_iters; _yi++){
  
          ws_j = cu_tidY + blkDimY*_yi;
          if (ws_j >= ws_w){ continue; }


          // --------------------
          //      init dists
          // --------------------
          dist = 0;

          // -----------------
          //    spatial dir
          // -----------------
          if (search_abs){
            n_hi = stride1 * ws_i;
            n_wi = stride1 * ws_j;
          }else{
            n_hi = ch + stride1 * (ws_i - wsOff_h);
            n_wi = cw + stride1 * (ws_j - wsOff_w);
          }

          // ---------------------------
          //      valid (xsearch "n")
          // ---------------------------

          valid_n_ti = (n_ti < nframes) && (n_ti >= 0);
          valid_n_hi = (n_hi < height) && (n_hi >= 0);
          valid_n_wi = (n_wi < width) && (n_wi >= 0);
          valid_n = valid_n_ti && valid_n_hi && valid_n_wi;
          valid = valid_n && valid_anchor;

          // ---------------------------------
          //
          //  compute delta over patch vol.
          //
          // ---------------------------------
          for (int pk = 0; pk < pt; pk++){
            // -- anchor time --
            vT = bounds(ti + pk,nframes);
            vvalid_t = (vT < nframes) && (vT >= 0);

            // -- proposed time --
            nT = bounds(n_ti + pk,nframes);
            nvalid_t = (nT < nframes) && (nT >= 0);

            for (int pi = 0; pi < ps; pi++){
              // -- anchor height --
              vH = (hi-h0_off) + dilation*(pi - psHalf + adj);
              vH = use_bounds ? bounds(vH,height) : vH;
              vvalid_h = (vH < height) && (vH >= 0);

              // -- propose height --
              nH = (n_hi-h1_off) + dilation*(pi - psHalf + adj);
              nH = use_bounds ? bounds(nH,height) : nH;
              nvalid_h = (nH < height) && (nH >= 0);

              for (int pj = 0; pj < ps; pj++){

                // -- anchor width --
                vW = (wi-w0_off) + dilation*(pj - psHalf + adj);
                vW = use_bounds ? bounds(vW,width) : vW;
                vvalid_w = (vW < width) && (vW >= 0);

                // -- propose width --
                nW = (n_wi-w1_off) + dilation*(pj - psHalf + adj);
                nW = use_bounds ? bounds(nW,width) : nW;
                nvalid_w = (nW < width) && (nW >= 0);

                // -- check valid --
                vvalid = vvalid_t && vvalid_h && vvalid_w;
                nvalid = nvalid_t && nvalid_h && nvalid_w;

                // -- all channels --
                for (int ci = 0; ci < chnls; ci++){

                  // -- get data --
                  v_pix = vvalid ? vid0[bindex][vT][ci][vH][vW] : 0.;
                  n_pix = nvalid ? vid1[bindex][nT][ci][nH][nW] : 0.;

                  // -- compute dist --
                  _dist = v_pix * n_pix;
                  dist += _dist;
                }
              }
            }
          }

          // -- dists --
          if (!valid){ dist = nan; }
          dists[bindex][bidx][wt_k][ws_i][ws_j] = dist;

          // -- inds --
          inds[bindex][bidx][wt_k][ws_i][ws_j][0] = n_ti;
          inds[bindex][bidx][wt_k][ws_i][ws_j][1] = n_hi;
          inds[bindex][bidx][wt_k][ws_i][ws_j][2] = n_wi;

          // -- final check [put self@index 0] --
          if (anchor_self){
            eq_dim = n_ti == ti;
            eq_dim = eq_dim && (n_hi == hi);
            eq_dim = eq_dim && (n_wi == wi);
            // eq_dim = eq_ti && eq_hi && eq_wi;
            if (eq_dim && use_self){
              self_dists[bindex][bidx] = dist; // update self
              dists[bindex][bidx][wt_k][ws_i][ws_j] = inf;
            }else if (eq_dim){
              dists[bindex][bidx][wt_k][ws_i][ws_j] = inf;
            }
          }

        }
      }
    }
  }
}

void search_prod_pf_with_index_forward_cuda(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor fflow, torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor self_dists,
    int qstart, int stride0, int n_h0, int n_w0,
    int ps, int pt, int ws_h_noneed, int ws_w_noneed,
    int wt, int chnls, int stride1, int dilation,
    bool use_search_abs, bool use_bounds, bool use_adj,
    bool full_ws, bool anchor_self, bool use_self,
    int h0_off, int w0_off, int h1_off, int w1_off,
    torch::Tensor tranges, torch::Tensor n_tranges){

    // # -- launch params --
    // w_threads = min(ws,32)
    // nthreads = (w_threads,w_threads)
    // ws_iters = (ws-1)//w_threads + 1
    // nblocks = (nq-1)//batches_per_block+1
    // fprintf(stdout,"use_search_abs, bool use_bounds, bool use_adj: %d,%d,%d\n",use_search_abs, use_bounds, use_adj);
    // fprintf(stdout,"h0_off, w0_off, h1_off, w1_off: %d,%d,%d,%d\n",h0_off, w0_off, h1_off, w1_off);
    // fprintf(stdout,"stride, dilation: %d,%d\n",stride, dilation);

    // -- threads --
    int bsize = dists.size(0);
    int numQueries = dists.size(1);
    int st = dists.size(2);
    int ws_h = dists.size(3);
    int ws_w = dists.size(4);
    int ws_h_threads = std::min(ws_h,15);
    int ws_w_threads = std::min(ws_w,15);
    int st_threads = std::min(st,5);
    int ws_h_iters = ((ws_h-1)/ws_h_threads) + 1;
    int ws_w_iters = ((ws_w-1)/ws_w_threads) + 1;
    int st_iters = ((st-1)/st_threads) + 1;
    dim3 nthreads(ws_h_threads,ws_w_threads,st_threads);

    // -- blocks --
    int bpt = 4;
    int nblocks_queries = ((numQueries - 1) / bpt) + 1;
    dim3 nblocks(bsize,nblocks_queries);
    // fprintf(stdout,"nblocks_queries,ws_h_threads,ws_w_threads,ws_h_iters,ws_w_iters,ws_h,ws_w: %d,%d,%d,%d,%d,%d,%d\n",nblocks_queries,ws_h_threads,ws_w_threads,ws_h_iters,ws_w_iters,ws_h,ws_w);
     
    // launch kernel
    AT_DISPATCH_FLOATING_TYPES(vid0.type(), "dnls_xsearch_forward_kernel", ([&] {
       search_prod_pf_with_index_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
         vid0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         vid1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         fflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
         bflow.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
         dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         inds.packed_accessor64<int,6,torch::RestrictPtrTraits>(),
         self_dists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
         qstart, stride0, n_h0, n_w0,
         ps, pt, ws_h, ws_w, wt, chnls, stride1, dilation, 
         use_search_abs, use_bounds, use_adj, full_ws,
         anchor_self, use_self, h0_off, w0_off, h1_off, w1_off,
         tranges.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
         n_tranges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
         ws_h_iters, ws_w_iters, st_iters, bpt);
       }));
}

