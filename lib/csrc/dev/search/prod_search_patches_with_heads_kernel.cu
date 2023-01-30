
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
__global__ void prod_search_patches_with_heads_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches1,
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> access_inds,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> inds,
    int chnls, int dilation, int anchor_self, int ws_h_iters, int ws_w_iters, int bpt){

  // shapes
  int nbatch,nqueries,nheads,nftr;
  nbatch = patches0.size(0);
  nqueries = patches0.size(1);
  nheads = patches0.size(2);
  nftr = patches0.size(3);

  // constants
  float nan = __int_as_float(0xffe00000);
  float inf = __int_as_float(0x7f800000);

  // cuda index
  int ibatch = blockIdx.x;
  int iquery = blockIdx.y;
  int ihead = blockIdx.z;

  // accumulate time offsets
  bool dir_fwd = true; // forward
  bool swap_dir = false;
  int prev_h,prev_w;
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
  int cw_i,ch_i,ch,cw;//,ct;
  // float cw0,ch0,ct0,cw_f,ch_f;
  // float dist,v_pix,n_pix;
  scalar_t cw0,ch0;//,ct0;//,cw_f,ch_f;
  scalar_t dist,v_pix,n_pix;


  for (int _bidx = 0; _bidx < bpt; _bidx++){

    //---------------------------
    //   extract anchor pixel
    //---------------------------

    // -- block start --
    bidx = block_start + _bidx;
    if (bidx >= nqueries){ continue; }


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

        // ----------------
        //     update
        // ----------------
        prev_w = cw;
        prev_h = ch;
  
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
                v_pix = vvalid ? patches0[bindex][head][vT][ci][vH][vW] : (scalar_t)0.;
                n_pix = nvalid ? patches1[bindex][head][nT][ci][nH][nW] : (scalar_t)0.;
  
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

void prod_search_patches_with_heads_forward_cuda(
    const torch::Tensor patches0, const torch::Tensor patches1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    const torch::Tensor access_inds,
    int chnls, int dilation, bool anchor_self){

    // # -- launch params --
    int nheads = dists.size(1);
    int nqueries = dists.size(2);
    int ws_h_threads = std::min(ws_h,29);
    int ws_w_threads = std::min(ws_w,29);
    int ws_h_iters = ((ws_h-1)/ws_h_threads) + 1;
    int ws_w_iters = ((ws_w-1)/ws_w_threads) + 1;
    dim3 nthreads(ws_h_threads,ws_w_threads);
    
    int nbatch = patches0.size(0);
    int rem_blocks = (65535-1)/nheads+1;
    int bpt = 2;
    int nquery_blocks = ((nqueries - 1) / bpt) + 1;
    nquery_blocks = min(nquery_blocks,rem_blocks);
    bpt = ((nqueries - 1) / nquery_blocks) + 1;
    dim3 nblocks(nbatch,nheads,nquery_blocks);

   // -- launch kernel --
   AT_DISPATCH_FLOATING_TYPES(patches0.type(),
                              "prod_seach_with_heads_forward_kernel", ([&] {
      prod_search_patches_with_heads_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        patches0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        patches1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        access_inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
        chnls, dilation, anchor_self, ws_h_iters, ws_w_iters, bpt);
      }));
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t>
__global__ void prod_search_patches_with_heads_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_patches0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_patches1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, bool use_adj, bool reflect_bounds,
    int bpt, int npt, int cpt) {

  // -- shape --
  int nbatch = grad_dists.size(0);
  int nhead = grad_dists.size(2);
  int nqueries = grad_dists.size(2);
  int k =  grad_dists.size(3);
  // int bs = patches0.size(0);
  int nframes = patches0.size(2);
  int colors = patches0.size(3);
  int height = patches0.size(4);
  int width = patches0.size(5);

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
                pix0 = weight*patches0[bindex][head][tk][c0][hk][wk];
                pix1 = weight*patches1[bindex][head][tj][c0][hj][wj];
                grad_patches1[bindex][head][tj][c0][hj][wj] += pix0;
                grad_patches0[bindex][head][tk][c0][hk][wk] += pix1;
              }
            }
          }
        }
      }
    }
  }
}

void prod_search_patches_with_heads_backward_cuda(
    torch::Tensor grad_patches0, torch::Tensor grad_patches1,
    torch::Tensor patches0, torch::Tensor patches1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int qstart, int nheads, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation,
    bool use_adj, bool reflect_bounds, bool use_rand,
    bool exact) {

  // -- unpack --
  int nbatch = patches0.size(0);
  int nframes = patches0.size(2);
  int colors = patches0.size(3);
  int height = patches0.size(4);
  int width = patches0.size(5);
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
  dim3 nblocks(nbatch,nquery_blocks,nheads);

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
  auto cu_index = grad_patches0.device().index();
  auto options = torch::TensorOptions().device(torch::kCUDA,
                                               cu_index).dtype(torch::kFloat32);
  torch::Tensor rand_nums;
  if (use_rand){
    rand_nums = torch::rand({nqueries,1,1},options);
  }else{
    rand_nums = torch::zeros({nqueries,1,1},options);
  }

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(patches0.type(),
                             "prod_seach_with_heads_backward_kernel", ([&] {
    prod_search_patches_with_heads_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        grad_patches0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        grad_patches1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        patches0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        patches1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        qstart, stride0, n_h0, n_w0, h0_off, w0_off, h1_off, w1_off,
        ps, pt, dilation, use_adj, reflect_bounds, bpt, npt, cpt);
  }));

}
