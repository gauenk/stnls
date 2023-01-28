
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

template <typename scalar_t>
__global__ void search_prod_pf_with_index_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> self_dists,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w, int wt, int chnls,
    int dilation, int stride1, bool use_adj, bool reflect_bounds,
    bool search_abs, bool full_ws, bool anchor_self, bool use_self,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> tranges,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> n_tranges,
    int ws_h_iters, int ws_w_iters, int st_per_thread, int bpt){

  // constants
  float nan = __int_as_float(0xffe00000);
  float inf = __int_as_float(0x7f800000);

  // shapes
  int bsize = vid0.size(0);
  int nframes = vid0.size(1);
  int color = vid0.size(2);
  int height = vid0.size(3);
  int width = vid0.size(4);
  int nqueries = dists.size(1);
  int st = dists.size(2);

  // offsets
  int psHalf = (ps)/2;
  int wsHalf_h = (ws_h)/2;
  int wsHalf_w = (ws_w)/2;
  int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  int wsMax_w = stride1*(ws_w-1-wsHalf_w);
  int adj = use_adj ? psHalf : 0;
  int n_hw0 = n_h0 * n_w0;

  // cuda index
  int bindex = blockIdx.x;
  int st_start = blockIdx.y*st_per_thread;
  int block_start = blockIdx.z*bpt;
  int blkDimX = blockDim.x; // num threads in x-block
  int blkDimY = blockDim.y; // num threads in y-block
  int cu_tidX = threadIdx.x;
  int cu_tidY = threadIdx.y;

  // decls
  int ch,cw;
  int hn,wn,tj;
  int ti,hi,wi;
  int qindex,i_mod;
  int n_ti,n_hi,n_wi;
  int bidx,ws_i,ws_j,wt_k;
  int wsOff_h,wsOff_w;
  int vH,vW,vT,nH,nW,nT;
  bool eq_dim;
  bool valid,vvalid,nvalid;
  bool valid_anchor,valid_n;
  bool vvalid_t,vvalid_h,vvalid_w;
  bool nvalid_t,nvalid_h,nvalid_w;

  scalar_t dist,v_pix,n_pix;

  for (int _bidx = 0; _bidx < bpt; _bidx++){

    //---------------------------
    //   extract anchor pixel
    //---------------------------

    // -- block start --
    bidx = block_start + _bidx;
    if (bidx >= nqueries){ continue; }

    // -- unpack pixel locs --
    qindex = bidx + qstart;
    i_mod = qindex % n_hw0;
    ti = qindex / n_hw0;
    wi = ((i_mod % n_w0) * stride0) % width ;
    hi = ((i_mod / n_w0) * stride0) % height;
    wn = (i_mod % n_w0);
    hn = (i_mod / n_w0) % n_h0;

    // -- valid (anchor pixel) --
    valid_anchor = (ti < nframes) && (ti >= 0);
    valid_anchor = valid_anchor && (hi < height) && (hi >= 0);
    valid_anchor = valid_anchor && (wi < width) && (wi >= 0);

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

    // -- reset flow search --
    for( int _st = 0; _st < st_per_thread; _st++){
      wt_k = _st + st_start;
      if (wt_k >= st){ continue; }
      int n_ti = tranges[ti][wt_k];

      // ------------------------
      //      init direction
      // ------------------------

      // -- compute direction --
      int direction = max(-1,min(1,n_ti - ti));

      // -- access flows --
      if (direction > 0 ){
        tj = n_ti - ti - 1;
        cw = fflow[bindex][tj][ti][0][hn][wn];
        ch = fflow[bindex][tj][ti][1][hn][wn];
      }else if (direction < 0){
        tj = ti - n_ti - 1;
        cw = bflow[bindex][tj][ti][0][hn][wn];
        ch = bflow[bindex][tj][ti][1][hn][wn];
      }else{
        cw = wi;
        ch = hi;
      }
      
      // ---------------------------------------
      //     searching loop for (ti,top,left)
      // ---------------------------------------
  
      // -- we loop over search space if needed --
      for (int _xi = 0; _xi < ws_h_iters; _xi++){
  
        ws_i = cu_tidX + blkDimX*_xi;
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
          //      valid (search "n")
          // ---------------------------
          valid_n = (n_ti < nframes) && (n_ti >= 0);
          valid_n = valid_n && (n_hi < height) && (n_hi >= 0);
          valid_n = valid_n && (n_wi < width) && (n_wi >= 0);
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
              vH = reflect_bounds ? bounds(vH,height) : vH;
              vvalid_h = (vH < height) && (vH >= 0);
              
              // -- propose height --
              nH = (n_hi-h1_off) + dilation*(pi - psHalf + adj);
              nH = reflect_bounds ? bounds(nH,height) : nH;
              nvalid_h = (nH < height) && (nH >= 0);


              for (int pj = 0; pj < ps; pj++){
                
                // -- anchor width --
                vW = (wi-w0_off) + dilation*(pj - psHalf + adj);
                vW = reflect_bounds ? bounds(vW,width) : vW;
                vvalid_w = (vW < width) && (vW >= 0);

                // -- propose width --
                nW = (n_wi-w1_off) + dilation*(pj - psHalf + adj);
                nW = reflect_bounds ? bounds(nW,width) : nW;
                nvalid_w = (nW < width) && (nW >= 0);

                // -- check valid --
                vvalid = vvalid_t && vvalid_h && vvalid_w;
                nvalid = nvalid_t && nvalid_h && nvalid_w;

                // -- all channels --
                for (int ci = 0; ci < chnls; ci++){

                  // -- get data --
                  v_pix = vvalid ? vid0[bindex][vT][ci][vH][vW] : (scalar_t)0.;
                  n_pix = nvalid ? vid1[bindex][nT][ci][nH][nW] : (scalar_t)0.;

                  // -- compute dist --
                  dist += v_pix * n_pix;
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
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor self_dists,
    int qstart, int stride0, int n_h0, int n_w0,
    int ps, int pt, int ws_h_noneed, int ws_w_noneed,
    int wt, int chnls, int stride1, int dilation,
    bool use_search_abs, bool reflect_bounds, bool use_adj,
    bool full_ws, bool anchor_self, bool use_self,
    int h0_off, int w0_off, int h1_off, int w1_off,
    const torch::Tensor tranges, const torch::Tensor n_tranges){

    // -- threads --
    int bsize = dists.size(0);
    int numQueries = dists.size(1);
    int st = dists.size(2);
    int ws_h = dists.size(3);
    int ws_w = dists.size(4);
    int ws_h_threads = std::min(ws_h,27);
    int ws_w_threads = std::min(ws_w,27);
    int ws_h_iters = ((ws_h-1)/ws_h_threads) + 1;
    int ws_w_iters = ((ws_w-1)/ws_w_threads) + 1;
    dim3 nthreads(ws_h_threads,ws_w_threads);

    // -- blocks --
    int bpt = 2;
    int st_iters = 1;
    int st_blocks = ((st-1)/st_iters) + 1;
    int nquery_blocks = ((numQueries - 1) / bpt) + 1;
    dim3 nblocks(bsize,st_blocks,nquery_blocks);
    // fprintf(stdout,"nquery_blocks,ws_h_threads,ws_w_threads,ws_h_iters,"
    // 	    "ws_w_iters,ws_h,ws_w: %d,%d,%d,%d,%d,%d,%d\n",
    // 	    nquery_blocks,ws_h_threads,ws_w_threads,ws_h_iters,ws_w_iters,ws_h,ws_w);
    // fprintf(stdout,"st_iters,st_blocks: %d,%d\n",st_iters,st_blocks);
     
    // launch kernel
    AT_DISPATCH_FLOATING_TYPES(vid0.type(), "prod_pf_with_index_kernel", ([&] {
       search_prod_pf_with_index_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
         vid0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         vid1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         fflow.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
         bflow.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
         dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         inds.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
         self_dists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
         qstart, stride0, n_h0, n_w0,
	 h0_off, w0_off, h1_off, w1_off,
         ps, pt, ws_h, ws_w, wt, chnls, dilation, stride1, 
         use_adj, reflect_bounds, use_search_abs, full_ws,
         anchor_self, use_self, 
         tranges.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
         n_tranges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
         ws_h_iters, ws_w_iters, st_iters, bpt);
       }));
}
