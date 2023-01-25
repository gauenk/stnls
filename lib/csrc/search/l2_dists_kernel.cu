
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
__global__ void l2_dists_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid0,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid1,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, int chnls,
    bool use_adj, bool reflect_bounds, bool anchor_self,
    int qpt, int npt){

  // constants
  float nan = __int_as_float(0xffe00000);
  float inf = __int_as_float(0x7f800000);

  // -- image shapes --
  int nframes,color,h,w,height,width,n_hw0;
  nframes = vid0.size(1);
  color = vid0.size(2);
  h = vid0.size(3);
  w = vid0.size(4);
  height = h;
  width = w;
  n_hw0 = n_h0 * n_w0;

  // -- offsets --
  int psHalf = (ps)/2;
  int adj = use_adj ? psHalf : 0;

  // -- access boundary --
  int i0_max = inds.size(1);
  int i1_max = inds.size(2);

  // -- get indices --
  int i0_start = qpt * blockIdx.x;
  int i1_start = npt * threadIdx.x;
  int c0_start = 0;
  int ibatch = blockIdx.y;

  // -- get block limits --
  int i0_end = min(i0_start + qpt,i0_max);
  int i1_end = min(i1_start + npt,i1_max);
  int c0_end = chnls;
  int c0_dist = c0_end - c0_start;
  
  // --decls --
  int qi,c0;
  int c0_offset;
  int i0,i1;
  int a_ti,a_hi,a_wi;
  int ti,hi,wi;
  int n_tj,n_hj,n_wj;
  int tj,hj,wj;
  // int aH,aW,aT,nH,nW,nT;
  // bool valid,vvalid,nvalid;
  bool valid_i,valid_hi,valid_wi;
  bool valid_j,valid_hj,valid_wj;
  // bool eq_ti,eq_hi,eq_wi,eq_dim;
  float _dist,dist,pix_i,pix_j;

  for (int i0 = i0_start; i0 < i0_end; i0++){

    // -- index block --
    qi = qstart + i0;

    // -- anchor [unpack] --
    int i_mod = qi % n_hw0;
    a_ti = qi / n_hw0;
    a_wi = ((i_mod % n_w0) * stride0) % width;
    a_hi = ((i_mod / n_w0) * stride0) % height;
    // c0_offset = __float2int_rd(c0_dist * rand_nums[i0][0][0]);
    c0_offset = 0;

    for (int i1 = i1_start; i1 < i1_end; i1++){

      // -- neighbor [unpack] --
      int n_tj = inds[ibatch][i0][i1][0];
      int n_hj = inds[ibatch][i0][i1][1];
      int n_wj = inds[ibatch][i0][i1][2];

      dist = 0;
      for (int pk = 0; pk < pt; pk++){
        for (int pi = 0; pi < ps; pi++){
          for (int pj = 0; pj < ps; pj++){

            // -- anchor patch --
            hi = (a_hi-h0_off) + dilation*(pi - psHalf + adj);
            hi = reflect_bounds ? bounds(hi,height) : hi;
            wi = (a_wi-w0_off) + dilation*(pj - psHalf + adj);
            wi = reflect_bounds ? bounds(wi,width) : wi;
            ti = reflect_bounds ? bounds(a_ti+pk,nframes) : a_ti+pk;

            // -- proposed location --
            hj = (n_hj-h1_off) + dilation*(pi - psHalf + adj);
            hj = reflect_bounds ? bounds(hj,height) : hj;
            wj = (n_wj-w1_off) + dilation*(pj - psHalf + adj);
            wj = reflect_bounds ? bounds(wj,width) : wj;
            tj = reflect_bounds ? bounds(n_tj+pk,nframes) : n_tj+pk;

            // -- assess if valid --
            valid_hi = (hi >= 0) && (hi < height);
            valid_wi = (wi >= 0) && (wi < width);
            valid_i = valid_hi && valid_wi;

            valid_hj = (hj >= 0) && (hj < height);
            valid_wj = (wj >= 0) && (wj < width);
            valid_j = valid_hj && valid_wj;

            // __syncthreads();
            for (int _c0 = c0_start; _c0 < c0_end; _c0++){
              c0 = (_c0 + c0_offset) % c0_dist + c0_start;
              pix_i =  valid_i ? vid0[ibatch][ti][c0][hi][wi] : 0.;
              pix_j =  valid_j ? vid1[ibatch][tj][c0][hj][wj] : 0.;
              _dist = (pix_i - pix_j);
              dist += _dist * _dist;
            }
          }
        }
      }
      dists[ibatch][i0][i1] = dist;
    }
  }
}


void l2_dists_forward_cuda(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists,torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, int chnls,
    bool use_adj, bool reflect_bounds, bool anchor_self){

    // -- shapes --
    int nbatch = inds.size(0);
    int nqueries = inds.size(1);
    int nneighs = inds.size(2);

    // -- threads --
    int tpt = 2;
    int nthreads = (nneighs-1)/tpt + 1;
    nthreads = min(nthreads,1024);
    tpt = ((nneighs - 1)/nthreads) + 1;
 
    // -- blocks --
    int qpt = 2;
    int nblocks_q = ((nqueries - 1) / qpt) + 1;
    nblocks_q = min(nblocks_q,65535);
    dim3 nblocks(nblocks_q,nbatch);
    qpt = ((nqueries - 1) / nblocks_q) + 1;

    // launch kernel
    AT_DISPATCH_FLOATING_TYPES(vid0.type(), "l2_dists_forward_kernel", ([&] {
       l2_dists_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
         vid0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         vid1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         dists.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
         inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
         qstart, stride0, n_h0, n_w0, h0_off, w0_off, h1_off, w1_off,
         ps, pt, dilation, chnls, use_adj, reflect_bounds, anchor_self,
         qpt, tpt);
       }));
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t>
__global__ void l2_dists_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_vid1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_dists,
    const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, bool use_adj, bool reflect_bounds,
    int bpt, int npt, int cpt) {

  // -- shape --
  int nq = grad_dists.size(1);
  int k =  grad_dists.size(2);
  int nframes = vid0.size(1);
  int colors = vid0.size(2);
  int height = vid0.size(3);
  int width = vid0.size(4);
  int n_hw0 = n_h0 * n_w0;

  // -- fwd decl registers --
  int ti,hi,wi;
  int tj,hj,wj;
  int tk,hk,wk;
  int tk_a,hk_a,wk_a;
  bool valid_hj,valid_wj;
  bool valid_hk,valid_wk;
  bool valid,valid_j,valid_k;
  float weight,pix,pix0,pix1;

  // -- declare constants --
  int psHalf = ps/2;
  int adj = use_adj ? psHalf : 0;

  // -- limits --
  int i0_max = inds.size(1);
  int i1_max = inds.size(2);

  // -- get indices --
  int i0_start = bpt * (threadIdx.x + blockDim.x * blockIdx.x);
  int i1_start = threadIdx.y * npt;
  int c0_start = threadIdx.z * cpt;
  int ibatch = blockIdx.y;

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
      ti = inds[ibatch][i0][i1][0];
      hi = inds[ibatch][i0][i1][1];
      wi = inds[ibatch][i0][i1][2];
      weight = grad_dists[ibatch][i0][i1];

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
            for (int _c0 = c0_start; _c0 < c0_end; _c0++){
              c0 = (_c0 + c0_offset) % c0_dist + c0_start;
              pix0 =  valid_k ? vid0[ibatch][tk][c0][hk][wk] : 0.;
              pix1 =  valid_j ? vid1[ibatch][tj][c0][hj][wj] : 0.;
              pix = 2 * weight * (pix0 - pix1);

              if (valid_j){
                grad_vid1[ibatch][tj][c0][hj][wj] -= pix;
              }
              if (valid_k){
                grad_vid0[ibatch][tk][c0][hk][wk] += pix;
              }

            }
          }
        }
      }
    }
  }
}

void l2_dists_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, int chnls,
    bool use_adj, bool reflect_bounds,
    bool use_rand, bool exact) {

  // -- unpack --
  int nbatch = vid0.size(0);
  int nframes = vid0.size(1);
  int colors = vid0.size(2);
  int height = vid0.size(3);
  int width = vid0.size(4);
  int nqueries = inds.size(1);
  int k = grad_dists.size(2);
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
  int nblocks_q = ((nqueries - 1) / total_per_block) + 1;
  if (exact){
    bpt = nqueries;
    query_nthreads = 1;
    nblocks_q = 1;
  }
  dim3 nblocks(nblocks_q,nbatch);

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
  AT_DISPATCH_FLOATING_TYPES(vid0.type(), "dnls_search_backward_kernel", ([&] {
    l2_dists_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        grad_vid0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_vid1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        vid0.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        vid1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_dists.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        qstart, stride0, n_h0, n_w0, h0_off, w0_off, h1_off, w1_off,
        ps,pt,dilation,use_adj,reflect_bounds,
        bpt,npt,cpt);
  }));

}


