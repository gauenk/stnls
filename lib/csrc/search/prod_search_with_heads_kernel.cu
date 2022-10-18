
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
using namespace at;

/****************************

       Inline Functions

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

inline __host__ __device__
int unravel_index(int& ti, int& hi, int& wi, const int qindex,
                  const int h, const int w, const int hw){
  // index to pixel location
  int i_mod = qindex % hw;
  ti = qindex / hw;
  wi = (i_mod % w);
  hi = (i_mod / w) % h;
}

/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void prod_search_with_heads_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int dilation, int stride1,
    bool use_adj, bool reflect_bounds, bool search_abs,
    bool full_ws, bool anchor_self,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> tranges,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> n_tranges,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> min_tranges,
    int ws_h_iters, int ws_w_iters, int bpt){

  // shapes
  int bsize,nheads,nframes,color,h,w,height,width;
  bsize = vid0.size(0);
  nheads = vid0.size(1);
  nframes = vid0.size(2);
  color = vid0.size(3);
  h = vid0.size(4);
  w = vid0.size(5);
  height = h;
  width = w;
  int n_hw0 = n_h0 * n_w0;
  int nqueries = dists.size(2);

  // constants
  float nan = __int_as_float(0xffe00000);
  float inf = __int_as_float(0x7f800000);

  // offsets
  int psHalf = (ps)/2;
  int wsHalf_h = (ws_h)/2;
  int wsHalf_w = (ws_w)/2;
  int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  int wsMax_w = stride1*(ws_w-1-wsHalf_w);
  int adj = use_adj ? psHalf : 0;

  // cuda index
  int bindex = blockIdx.x;
  int head = blockIdx.y;
  int blkDimX = blockDim.y; // num threads in x-block
  int blkDimY = blockDim.z; // num threads in y-block
  int cu_tidX = threadIdx.x;
  int cu_tidY = threadIdx.y;
  int block_start = blockIdx.z*bpt;
  int bidx,ws_i,ws_j;

  // accumulate time offsets
  bool dir_fwd = true; // forward
  bool swap_dir = false;
  int prev_h,prev_w,prev_t;
  int min_t;

  // decls
  int ti,hi,wi;
  int n_ti,n_hi,n_wi;
  int vH,vW,vT,nH,nW,nT;
  bool valid,vvalid,nvalid;
  bool valid_ti,valid_hi,valid_wi,valid_anchor;
  bool valid_n_ti,valid_n_hi,valid_n_wi,valid_n;
  bool vvalid_t,vvalid_h,vvalid_w;
  bool nvalid_t,nvalid_h,nvalid_w;
  bool eq_ti,eq_hi,eq_wi,eq_dim;
  int wsOff_h,wsOff_w;

  int l_cw0,l_ch0,l_ct0;
  int cw_i,ch_i,ch,cw,ct;
  // float cw0,ch0,ct0,cw_f,ch_f;
  // float dist,v_pix,n_pix;
  scalar_t cw0,ch0,ct0,cw_f,ch_f;
  scalar_t dist,v_pix,n_pix;


  for (int _bidx = 0; _bidx < bpt; _bidx++){

    //---------------------------
    //   extract anchor pixel
    //---------------------------

    // -- block start --
    bidx = block_start + _bidx;
    if (bidx >= nqueries){ continue; }

    // -- unpack pixel locs --
    int qindex = bidx + qstart;
    int i_mod = qindex % n_hw0;
    ti = qindex / n_hw0;
    wi = ((i_mod % n_w0) * stride0) % width ;
    hi = ((i_mod / n_w0) * stride0) % height;

    // -- min of search range --
    min_t = min_tranges[ti];

    // unravel_index(ti, hi, wi, qindex, height, width, hw);

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
    //     searching loop for (ti,top,left)
    // ---------------------------------------

    // -- we loop over search space if needed --
    for (int _xi = 0; _xi < ws_h_iters; _xi++){

      int ws_i = cu_tidX + blkDimX*_xi;
      if (ws_i >= ws_h){ continue; }

      for (int _yi = 0; _yi < ws_w_iters; _yi++){
        ws_j = cu_tidY + blkDimY*_yi;
        if (ws_j >= ws_w){ continue; }

        // -- reset flow search --
        dir_fwd = true;
        swap_dir = false;
        for( int wt_k = 0; wt_k < n_tranges[ti]; wt_k++){
          int n_ti = tranges[ti][wt_k];
          int dt = n_ti - min_t;

          // ------------------------
          //      init direction
          // ------------------------

          // -- compute direction --
          int direction = max(-1,min(1,n_ti - ti));

          // -- reset to anchor when swapping directions --
          swap_dir = (dir_fwd == true) && (direction == -1);
          dir_fwd = swap_dir ? false : dir_fwd;
          prev_w = swap_dir ? wi : prev_w;
          prev_h = swap_dir ? hi : prev_h;
          prev_t = swap_dir ? ti : prev_t;

          // -- optical flow --
          if (direction != 0){

            // -- get offset at index --
            // int dtd = int(dt-direction);
            cw0 = 1.*prev_w;
            ch0 = 1.*prev_h;
            ct0 = 1.*prev_t;

            // -- legalize access --
            l_cw0 = int(max(0,min(width-1,int(cw0))));
            l_ch0 = int(max(0,min(height-1,int(ch0))));
            l_ct0 = int(max(0,min(nframes-1,int(ct0))));

            // -- access flows --
            if (direction > 0 ){
              cw_f = cw0 + fflow[bindex][l_ct0][0][l_ch0][l_cw0];
              ch_f = ch0 + fflow[bindex][l_ct0][1][l_ch0][l_cw0];
            }else{
              cw_f = cw0 + bflow[bindex][l_ct0][0][l_ch0][l_cw0];
              ch_f = ch0 + bflow[bindex][l_ct0][1][l_ch0][l_cw0];
            }
            cw_i = int(cw_f + 0.5);
            ch_i = int(ch_f + 0.5);

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
          prev_w = cw;
          prev_h = ch;
          prev_t = ct;

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
                  v_pix = vvalid ? vid0[bindex][head][vT][ci][vH][vW] : (scalar_t)0.;
                  n_pix = nvalid ? vid1[bindex][head][nT][ci][nH][nW] : (scalar_t)0.;

                  // -- compute dist --
                  dist += v_pix * n_pix;
                  // if (valid){
                  // }
                }
              }
            }
          }

          // -- dists --
          if (!valid){ dist = nan; }
          dists[bindex][head][bidx][wt_k][ws_i][ws_j] = dist;

          // -- inds --
          inds[bindex][head][bidx][wt_k][ws_i][ws_j][0] = n_ti;
          inds[bindex][head][bidx][wt_k][ws_i][ws_j][1] = n_hi;
          inds[bindex][head][bidx][wt_k][ws_i][ws_j][2] = n_wi;

          // -- final check [put self@index 0] --
          if (anchor_self){
            eq_ti = n_ti == ti;
            eq_hi = n_hi == hi;
            eq_wi = n_wi == wi;
            eq_dim = eq_ti && eq_hi && eq_wi;
            if (eq_dim){
              dists[bindex][head][bidx][wt_k][ws_i][ws_j] = inf;
            }
          }
          
        }
      }
    }
  }
}

void prod_search_with_heads_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int dilation, int stride1,
    bool use_adj, bool reflect_bounds, bool search_abs,
    bool full_ws, bool anchor_self,
    const torch::Tensor tranges,
    const torch::Tensor n_tranges, const torch::Tensor min_tranges){

    // # -- launch params --
    // w_threads = min(ws,32)
    // nthreads = (w_threads,w_threads)
    // ws_iters = (ws-1)//w_threads + 1
    // nblocks = (nq-1)//batches_per_block+1

   // fprintf(stdout,"qstart, nqueries: %d,%d\n",qstart,nqueries);
   // launch params
   // our many (too many?) registers limit the number of threads
   int nheads = dists.size(1);
   int nqueries = dists.size(2);
   int ws_h_threads = std::min(ws_h,29);
   int ws_w_threads = std::min(ws_w,29);
   int ws_h_iters = ((ws_h-1)/ws_h_threads) + 1;
   int ws_w_iters = ((ws_w-1)/ws_w_threads) + 1;
   dim3 nthreads(ws_h_threads,ws_w_threads);

   int bsize = vid0.size(0);
   int rem_blocks = (65535-1)/nheads+1;
   int bpt = 2;
   int nquery_blocks = ((nqueries - 1) / bpt) + 1;
   nquery_blocks = min(nquery_blocks,rem_blocks);
   bpt = ((nqueries - 1) / nquery_blocks) + 1;
   dim3 nblocks(bsize,nheads,nquery_blocks);

   // fprintf(stdout,"bsize,nheads,nquery_blocks: %d,%d,%d\n",
   //         bsize,nheads,nquery_blocks);
   // fprintf(stdout,"bpt,nquery_blocks,w_threads: %d,%d,%d,%d\n",
   //         bpt,nquery_blocks,ws_h_threads,ws_w_threads);
   // fprintf(stdout,"reflect_bounds,search_abs,full_ws,anchor_self: %d,%d,%d,%d\n",
   //         reflect_bounds,search_abs,full_ws,anchor_self);
   // fprintf(stdout,"ws_h_iters,ws_w_iters: %d,%d\n",ws_h_iters,ws_w_iters);
    
   // launch kernel
   AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                              "prod_seach_with_heads_forward_kernel", ([&] {
      prod_search_with_heads_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        fflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        bflow.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
        qstart, stride0, n_h0, n_w0,
        h0_off, w0_off, h1_off, w1_off,
        ps, pt, ws_h, ws_w, wt, chnls, dilation, stride1,
        use_adj, reflect_bounds, search_abs, full_ws, anchor_self,
        tranges.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        n_tranges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
        min_tranges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
        ws_h_iters, ws_w_iters, bpt);
      }));
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t>
__global__ void prod_search_with_heads_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, bool use_adj, bool reflect_bounds,
    int bpt, int npt, int cpt) {

  // -- shape --
  int bs = grad_dists.size(0);
  int nq = grad_dists.size(2);
  int k =  grad_dists.size(3);
  // int bs = vid0.size(0);
  int nframes = vid0.size(2);
  int colors = vid0.size(3);
  int height = vid0.size(4);
  int width = vid0.size(5);
  int n_hw0 = n_h0 * n_w0;

  // -- fwd decl registers --
  int ti,hi,wi;
  int tj,hj,wj;
  int tk,hk,wk;
  int tk_a,hk_a,wk_a;
  bool valid_hj,valid_wj;
  bool valid_hk,valid_wk;
  bool valid,valid_j,valid_k;
  // float 
  scalar_t weight,pix,pix0,pix1;

  // -- declare constants --
  int psHalf = ps/2;
  int adj = use_adj ? psHalf : 0;

  // -- limits --
  int i0_max = inds.size(2); // nq
  int i1_max = inds.size(3); // k

  // -- get indices --
  int bindex = blockIdx.x;
  int i0_start = bpt * (threadIdx.x + blockDim.x * blockIdx.y);
  int i1_start = threadIdx.y * npt;
  int c0_start = threadIdx.z * cpt;
  int head = blockIdx.z;

  // -- get block limits --
  int i0_end = min(i0_start + bpt,i0_max);
  int i1_end = min(i1_start + npt,i1_max);
  int c0_end = min(c0_start + cpt,colors);

  // -- color offset --
  int c0 = 0;
  int c0_dist = c0_end - c0_start;
  int c0_offset = 0;
  // if (threadIdx.x == 0){
  //   printf("c0_dist,c0_start,c0_end: %d,%d,%d\n",c0_dist,c0_start,c0_end);
  // }

  // -- each region --
  for (int i0=i0_start; i0 < i0_end; i0++){

    int qindex = i0 + qstart;
    int i_mod = qindex % n_hw0;
    tk_a = qindex / n_hw0;
    wk_a = ((i_mod % n_w0) * stride0) % width ;
    hk_a = ((i_mod / n_w0) * stride0) % height;
    c0_offset = __float2int_rd(c0_dist * rand_nums[i0][0][0]);
    // printf("c0_offset: %d\n",c0_offset);

    // k neighbors
    for (int i1=i1_start; i1 < i1_end; i1++){
      ti = inds[bindex][head][i0][i1][0];
      hi = inds[bindex][head][i0][i1][1];
      wi = inds[bindex][head][i0][i1][2];
      weight = grad_dists[bindex][head][i0][i1];

      for (int pk = 0; pk < pt; pk++){
        for (int pi = 0; pi < ps; pi++){
          for (int pj = 0; pj < ps; pj++){
            

            // -- anchor patch --
            hk = (hk_a-h0_off) + dilation*(pi - psHalf + adj);
            hk = reflect_bounds ? bounds(hk,height) : hk;
            wk = (wk_a-w0_off) + dilation*(pj - psHalf + adj);
            wk = reflect_bounds ? bounds(wk,width) : wk;
            tk = reflect_bounds ? bounds(tk_a+pk,nframes) : tk_a+pk;

            // -- proposed location --
            hj = (hi-h1_off) + dilation*(pi - psHalf + adj);
            hj = reflect_bounds ? bounds(hj,height) : hj;
            wj = (wi-w1_off) + dilation*(pj - psHalf + adj);
            wj = reflect_bounds ? bounds(wj,width) : wj;
            tj = reflect_bounds ? bounds(ti+pk,nframes) : ti+pk;

            // -- assess if valid --
            valid_hj = (hj >= 0) && (hj < height);
            valid_wj = (wj >= 0) && (wj < width);
            valid_j = valid_hj && valid_wj;

            valid_hk = (hk >= 0) && (hk < height);
            valid_wk = (wk >= 0) && (wk < width);
            valid_k = valid_hk && valid_wk;

            // __syncthreads();
            valid = valid_j && valid_k;
            for (int _c0 = c0_start; _c0 < c0_end; _c0++){
              c0 = (_c0 + c0_offset) % c0_dist + c0_start;
              if(valid){
                pix0 = weight*vid0[bindex][head][tk][c0][hk][wk];
                pix1 = weight*vid1[bindex][head][tj][c0][hj][wj];
                grad_vid1[bindex][head][tj][c0][hj][wj] += pix0;
                grad_vid0[bindex][head][tk][c0][hk][wk] += pix1;
              }
            }
          }
        }
      }
    }
  }
}

void prod_search_with_heads_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int qstart, int nheads, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation,
    bool use_adj, bool reflect_bounds, bool use_rand,
    bool exact) {

  // -- unpack --
  int bsize = vid0.size(0);
  int nframes = vid0.size(2);
  int colors = vid0.size(3);
  int height = vid0.size(4);
  int width = vid0.size(5);
  int nqueries = inds.size(2);
  int k = inds.size(3);
  assert(pt == 1);

  // -- compute number of neighbor threads --
  int npt = 4;
  int neigh_nthreads = (k-1) / npt + 1;
  if (neigh_nthreads > 32){
    neigh_nthreads = 32;
    npt = (k-1)/neigh_nthreads + 1;
  }
  if (exact){
    neigh_nthreads = 1;
    npt = k;
  }

  // -- compute number of color threads --
  int cpt = exact ? 1 : colors;
  int color_nthreads = (colors - 1)/cpt + 1;

  // -- compute number of blocks --
  //    [think: parallelization over "nqueries"]
  int bpt = 2;
  int query_nthreads = 32;
  int total_per_block = bpt * query_nthreads;
  int nquery_blocks = ((nqueries - 1) / total_per_block) + 1;
  if (exact){
    bpt = nqueries;
    query_nthreads = 1;
    nquery_blocks = 1;
  }
  dim3 nblocks(bsize,nquery_blocks,nheads);

  // -- launch params --
  dim3 nthreads(query_nthreads, neigh_nthreads, color_nthreads);

  // -- info --
  // fprintf(stdout,
  //         "query_nthreads, neigh_nthreads, color_nthreads: %d,%d,%d\n",
  //         query_nthreads, neigh_nthreads, color_nthreads);
  // fprintf(stdout,"nblocks: %d\n",nblocks);
  // fprintf(stdout,"bpt,npt,cpt: %d,%d,%d\n",bpt,npt,cpt);
  // fprintf(stdout,"h0_off,w0_off,h1_off,w1_off: %d,%d,%d,%d\n",
  //         h0_off,w0_off,h1_off,w1_off);
  // fprintf(stdout,"use_adj,use_reflect: %d,%d\n",
  //         use_adj,reflect_bounds);
  // fprintf(stdout,"ps,pt,dil: %d,%d,%d\n",ps,pt,dilation);

  // -- allocate random values --
  auto cu_index = grad_vid0.device().index();
  auto options = torch::TensorOptions().device(torch::kCUDA,
                                               cu_index).dtype(torch::kFloat32);
  torch::Tensor rand_nums;
  if (use_rand){
    rand_nums = torch::rand({nqueries,1,1},options);
  }else{
    rand_nums = torch::zeros({nqueries,1,1},options);
  }

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                             "prod_seach_with_heads_backward_kernel", ([&] {
    prod_search_with_heads_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        qstart, stride0, n_h0, n_w0, h0_off, w0_off, h1_off, w1_off,
        ps, pt, dilation, use_adj, reflect_bounds, bpt, npt, cpt);
  }));

}
