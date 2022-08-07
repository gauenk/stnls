
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// #include <ATen/NativeFunctions.h>
// #include <ATen/Context.h>
// #include <ATen/Dispatch.h>
// #include <ATen/cuda/CUDAApplyUtils.cuh>

#define FULL_MASK 0xffffffff


/****************************

       Helper Funcs

****************************/

__inline__ __device__ int bounds(int val, int lb, int ub ){
  int vval = val;
  if (val < lb){
    vval = 2*lb - val;
  }else if (val >= ub){
    vval = 2*(ub-1) - val;
  }
  return vval;
}

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}

/************************************

       Forward Pass

************************************/

template <typename scalar_t>
__global__ void dnls_gather_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> wvid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    int ws, int wt, int dilation, int qpt) {

  // shared mem
  static __shared__ float shared[32*3*2];

  // unpack cuda threads
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;

  // warp info for reduction
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  int numWarps = ((blockDim.x-1) / warpSize)+1;

  // shape
  int nq =    patches.size(0);
  int k =     patches.size(1);
  int pt =    patches.size(2);
  int colors = patches.size(3);
  int ps =    patches.size(4);

  // vid shape
  int nframes = vid.size(0);
  int height = vid.size(2);
  int width = vid.size(3);
  int psHalf = (ps-1)/2;

  // shape helpers
  int hw = height * width;
  int sq_hp = height;
  int sq_wp = width;
  int sq_hwp = sq_hp * sq_wp;

  // spatial radius
  int S = ws + 2*(dilation*(ps/2));
  int S2 = S * S;
  int L = S * S * (2 * wt + 1);
  bool valid_thread = threadIdx.x < L;

  // location
  int ti,hi,wi;
  float weight;
  bool valid;

  // get thread's center
  int q0 = bidx;
  int q_start = 0; // must be linked to inds[min]
  int index = q0 + q_start;

  // compute p0
  const int64_t t0 = (index / sq_hwp);
  const int64_t i_mod = index % sq_hwp;
  const int64_t w0 = (i_mod % width);
  const int64_t h0 = ((i_mod / width) % height);

  // get offset coords
  int t_offset = tidx / S2;
  const int64_t thr_mod = tidx % S2;
  int w_offset = thr_mod % S;
  int h_offset = (thr_mod / S) % S;

  // get top-left
  int shift_t = min(0,(int)(t0 - wt)) + max(0,(int)(t0 + wt - nframes + pt));
  int t_start = max((int)(t0 - wt - shift_t),0);
  // int t_start = t0 - wt;
  int h_start = h0 - (ws/2 + dilation*(ps/2));
  int w_start = w0 - (ws/2 + dilation*(ps/2));

  // compute p1
  int t1 = t_start + t_offset;
  int h1 = h_start + h_offset;
  int w1 = w_start + w_offset;

  // check valid
  bool valid_q1 = (t1 >= 0) && (t1 < nframes);
  valid_q1 = valid_q1 && (h1 >= 0) && (h1 < height);
  valid_q1 = valid_q1 && (w1 >= 0) && (w1 < width);
  valid_q1 = valid_q1 && valid_thread;

  // find p1 in Q
  int q1_raw = t1 * hw + h1 * width + w1;
  int q1 = valid_q1 ? q1_raw : 0;
  float pix_i;
  float pix[3];
  float wpix[3];

  // init to zero
  for (int ci = 0; ci < colors; ci++){
    pix[ci]  = 0;
    wpix[ci] = 0;
  }
  
  // accumulate over K neighbors
  for (int ki = 0; ki < k; ki++){
    weight = dists[q1][ki];
    int t1_i = inds[q1][ki][0];
    int h1_i = inds[q1][ki][1];
    int w1_i = inds[q1][ki][2];

    for (int pk = 0; pk < pt; pk++){
      for (int pi = 0; pi < ps; pi++){
        for (int pj = 0; pj < ps; pj++){
  
          // prop ind
          ti = t1_i + pk;
          hi = h1_i + dilation*(pi - psHalf);
          wi = w1_i + dilation*(pj - psHalf);
  
          // valid
          valid = (ti == t0);
          valid = valid && (hi == h0);
          valid = valid && (wi == w0);
          valid = valid && valid_q1;
  
          // fill
          for (int ci = 0; ci < colors; ci++){
            pix_i = patches[q1][ki][pk][ci][pi][pj];
            pix[ci] += valid ? (weight * pix_i) : 0.;
            wpix[ci] += valid ? weight : 0.;
          }
        }
      }
    }
  }
  
  // accumulate over threads in warp
  for (int ci = 0; ci < colors; ci++){
    pix[ci] = warpReduceSum(pix[ci]);
    wpix[ci] = warpReduceSum(wpix[ci]);
  }

  
  // Write reduced value to shared memory
  if (lane==0){ 
    for (int ci = 0; ci < colors; ci++){
      shared[2*wid*colors+2*ci]=pix[ci];
      shared[2*wid*colors+2*ci+1]=wpix[ci];
    }
  }

  __syncthreads(); // Wait for all partial reductions

  //read from shared memory only if that warp existed
  bool valid_warp_thread = threadIdx.x < numWarps;
  for (int ci = 0; ci < colors; ci++){
    pix[ci]  = valid_warp_thread ? shared[2*lane*colors+2*ci] : 0;
    wpix[ci] = valid_warp_thread ? shared[2*lane*colors+2*ci+1] : 0;
  }

  if (wid==0){ 

    // Final reduce within first warp
    for (int ci = 0; ci < colors; ci++){
      pix[ci] = warpReduceSum(pix[ci]);
      wpix[ci] = warpReduceSum(wpix[ci]);
    }

    // Assign first lane to image value
    if (lane==0){
      for (int ci = 0; ci < colors; ci++){
        vid[t0][ci][h0][w0] = pix[ci];
        wvid[t0][ci][h0][w0] = wpix[ci];
      }
    }
  }
}


void dnls_cuda_gather_forward(
    torch::Tensor vid,torch::Tensor wvid,torch::Tensor patches,
    torch::Tensor dists,torch::Tensor inds,
    int ws, int wt, int dilation) {

  // launch params
  int numQueries = inds.size(0);
  int k = dists.size(1);
  int pt = patches.size(2);
  int color = patches.size(3);
  int ps = patches.size(4);
  assert(pt == 1);
  assert(color <= 3);

  // spatial radius
  int S = ws + 2*(dilation*(ps/2));
  int L = S * S * (2 * wt + 1);
  int kwWarpSize = 32;
  int nWarps = (L-1)/kwWarpSize+1;
  // fprintf(stdout,"S,L,ws,wt: %d,%d,%d,%d\n",S,L,ws,wt);

  // launching params
  int qpt = 1;
  int nthreads = kwWarpSize*nWarps;
  int queries_per_block = nthreads * qpt;
  int nblocks = numQueries;
  // fprintf(stdout,"nthreads,nblocks: %d,%d\n",nthreads,nblocks);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_gather_forward_kernel", ([&] {
    dnls_gather_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        wvid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        ws,wt,dilation,qpt);
      }));
}


/***********************************************************

     Forward Pass (dist threads across search space)

***********************************************************/


template <typename scalar_t>
__global__ void dnls_gather_forward_kernel_dist(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> wvid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    int ws, int wt, int dilation, int qpt) {

  // shared mem
  static __shared__ float shared[32*3*2];

  // unpack cuda threads
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;

  // warp info for reduction
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  int numWarps = ((blockDim.x-1) / warpSize)+1;

  // shape
  int nq =    patches.size(0);
  int k =     patches.size(1);
  int pt =    patches.size(2);
  int colors = patches.size(3);
  int ps =    patches.size(4);

  // vid shape
  int nframes = vid.size(0);
  int height = vid.size(2);
  int width = vid.size(3);
  int psHalf = (ps-1)/2;

  // shape helpers
  int hw = height * width;
  int sq_hp = height;
  int sq_wp = width;
  int sq_hwp = sq_hp * sq_wp;

  // spatial radius
  int S = ws + 2*(dilation*(ps/2));
  int S2 = S * S;
  int L = S * S * (2 * wt + 1);
  bool valid_thread = threadIdx.x < L;

  // location
  int ti,hi,wi;
  float weight;
  bool valid;

  // get thread's center
  int q0 = bidx;
  int q_start = 0; // must be linked to inds[min]
  int index = q0 + q_start;

  // compute p0
  const int64_t t0 = (index / sq_hwp);
  const int64_t i_mod = index % sq_hwp;
  const int64_t w0 = (i_mod % width);
  const int64_t h0 = ((i_mod / width) % height);

  // get offset coords
  int t_offset = tidx / S2;
  const int64_t thr_mod = tidx % S2;
  int w_offset = thr_mod % S;
  int h_offset = (thr_mod / S) % S;

  // get top-left
  int shift_t = min(0,(int)(t0 - wt)) + max(0,(int)(t0 + wt - nframes + pt));
  int t_start = max((int)(t0 - wt - shift_t),0);
  // int t_start = t0 - wt;
  int h_start = h0 - (ws/2 + dilation*(ps/2));
  int w_start = w0 - (ws/2 + dilation*(ps/2));

  // compute p1
  int t1 = t_start + t_offset;
  int h1 = h_start + h_offset;
  int w1 = w_start + w_offset;

  // check valid
  bool valid_q1 = (t1 >= 0) && (t1 < nframes);
  valid_q1 = valid_q1 && (h1 >= 0) && (h1 < height);
  valid_q1 = valid_q1 && (w1 >= 0) && (w1 < width);
  valid_q1 = valid_q1 && valid_thread;

  // find p1 in Q
  int q1_raw = t1 * hw + h1 * width + w1;
  int q1 = valid_q1 ? q1_raw : 0;
  float pix_i;
  float pix[3];
  float wpix[3];

  // init to zero
  for (int ci = 0; ci < colors; ci++){
    pix[ci]  = 0;
    wpix[ci] = 0;
  }
  
  // accumulate over K neighbors
  for (int ki = 0; ki < k; ki++){
    weight = dists[q1][ki];
    int t1_i = inds[q1][ki][0];
    int h1_i = inds[q1][ki][1];
    int w1_i = inds[q1][ki][2];

    for (int pk = 0; pk < pt; pk++){
      for (int pi = 0; pi < ps; pi++){
        for (int pj = 0; pj < ps; pj++){
  
          // prop ind
          ti = t1_i + pk;
          hi = h1_i + dilation*(pi - psHalf);
          wi = w1_i + dilation*(pj - psHalf);
  
          // valid
          valid = (ti == t0);
          valid = valid && (hi == h0);
          valid = valid && (wi == w0);
          valid = valid && valid_q1;
  
          // fill
          for (int ci = 0; ci < colors; ci++){
            pix_i = patches[q1][ki][pk][ci][pi][pj];
            pix[ci] += valid ? (weight * pix_i) : 0.;
            wpix[ci] += valid ? weight : 0.;
          }
        }
      }
    }
  }
  
  // accumulate over threads in warp
  for (int ci = 0; ci < colors; ci++){
    pix[ci] = warpReduceSum(pix[ci]);
    wpix[ci] = warpReduceSum(wpix[ci]);
  }

  
  // Write reduced value to shared memory
  if (lane==0){ 
    for (int ci = 0; ci < colors; ci++){
      shared[2*wid*colors+2*ci]=pix[ci];
      shared[2*wid*colors+2*ci+1]=wpix[ci];
    }
  }

  __syncthreads(); // Wait for all partial reductions

  //read from shared memory only if that warp existed
  bool valid_warp_thread = threadIdx.x < numWarps;
  for (int ci = 0; ci < colors; ci++){
    pix[ci]  = valid_warp_thread ? shared[2*lane*colors+2*ci] : 0;
    wpix[ci] = valid_warp_thread ? shared[2*lane*colors+2*ci+1] : 0;
  }

  if (wid==0){ 

    // Final reduce within first warp
    for (int ci = 0; ci < colors; ci++){
      pix[ci] = warpReduceSum(pix[ci]);
      wpix[ci] = warpReduceSum(wpix[ci]);
    }

    // Assign first lane to image value
    if (lane==0){
      for (int ci = 0; ci < colors; ci++){
        vid[t0][ci][h0][w0] = pix[ci];
        wvid[t0][ci][h0][w0] = wpix[ci];
      }
    }
  }
}


void dnls_cuda_gather_forward_dist(
    torch::Tensor vid,torch::Tensor wvid,torch::Tensor patches,
    torch::Tensor dists,torch::Tensor inds,
    int ws, int wt, int dilation) {

  // launch params
  int numQueries = inds.size(0);
  int k = dists.size(1);
  int pt = patches.size(2);
  int color = patches.size(3);
  int ps = patches.size(4);
  assert(pt == 1);
  assert(color <= 3);

  // spatial radius
  int S = ws + 2*(dilation*(ps/2));
  int L = S * S * (2 * wt + 1);
  int kwWarpSize = 32;
  int nWarps = (L-1)/kwWarpSize+1;
  // fprintf(stdout,"S,L,ws,wt: %d,%d,%d,%d\n",S,L,ws,wt);

  // launching params
  int qpt = 1;
  int nthreads = kwWarpSize*nWarps;
  int queries_per_block = nthreads * qpt;
  int nblocks = numQueries;
  // fprintf(stdout,"nthreads,nblocks: %d,%d\n",nthreads,nblocks);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_gather_forward_kernel", ([&] {
    dnls_gather_forward_kernel_dist<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        wvid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        ws,wt,dilation,qpt);
      }));
}


/************************************

   Forward Pass (with race cond)

************************************/

template <typename scalar_t>
__global__ void dnls_gather_forward_kernel_race(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> wvid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int dilation, int qpt, int cpt) {

  // shape
  int nq =    patches.size(0);
  int k =     patches.size(1);
  int pt =    patches.size(2);
  int colors = patches.size(3);
  int ps =    patches.size(4);

  // vid shape
  int nframes = vid.size(0);
  int height = vid.size(2);
  int width = vid.size(3);
  int psHalf = (ps-1)/2;

  // location
  int qi,ti,hi,wi;
  float weight;

  // only valid 
  bool valid;
  float pix;

  // -- endpoints --
  int c0_start = threadIdx.y * cpt;
  int c0_end = min(c0_start + cpt,colors);

  // -- color offset --
  int ci = 0;
  int c0_dist = c0_end - c0_start;
  int c0_offset = 0;

  // get indices
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;
  int q_start = qpt*(tidx + bidx * blockDim.x);

  for (int _qi = 0; _qi < qpt; _qi++){
    qi = q_start + _qi;
    c0_offset = __float2int_rd(c0_dist * rand_nums[qi][0][0]);

    if (qi < nq){
      // iterate
      for (int ki = 0; ki < k; ki++){
        weight = dists[qi][ki];
        for (int pk = 0; pk < pt; pk++){
          for (int pi = 0; pi < ps; pi++){
            for (int pj = 0; pj < ps; pj++){

              // prop ind
              ti = inds[qi][ki][0] + pk;
              hi = inds[qi][ki][1] + dilation*(pi - psHalf);
              wi = inds[qi][ki][2] + dilation*(pj - psHalf);

              // valid
              valid = (ti >= 0) && (ti < nframes);
              valid = valid && (hi >= 0) && (hi < height);
              valid = valid && (wi >= 0) && (wi < width);

              // fill
              for (int _c0 = c0_start; _c0 < c0_end; _c0++){
                ci = (_c0 + c0_offset) % c0_dist + c0_start;
                pix = patches[qi][ki][pk][ci][pi][pj];
                if (valid){
                  vid[ti][ci][hi][wi] += weight * pix;
                  wvid[ti][ci][hi][wi] += weight;
                }
              }
            }
          }
        }
      }
    }
  }
}

void dnls_cuda_gather_forward_race(
    torch::Tensor vid,torch::Tensor wvid,torch::Tensor patches,
    torch::Tensor dists,torch::Tensor inds,
    int dilation, bool use_rand, bool exact) {

  // launch params
  int nqueries = inds.size(0);
  int k = dists.size(1);
  int pt = patches.size(2);
  int color = patches.size(3);
  int ps = patches.size(4);
  assert(pt == 1);

  int cpt = exact ? 1 : color;
  int nthreads_color = (color - 1)/cpt + 1;
  int qpt = 2;
  int nthreads_blocks = 1024;
  int queries_per_block = nthreads_blocks * qpt;
  int nblocks = ((nqueries - 1) / queries_per_block) + 1;
  if (exact){
    nthreads_blocks = 1;
    nblocks = 1;
    qpt = nqueries;
  }
  dim3 nthreads(nthreads_blocks,nthreads_color);

  // -- allocate random values --
  auto cu_index = vid.device().index();
  auto options = torch::TensorOptions().device(
                        torch::kCUDA,cu_index).dtype(torch::kFloat32);
  torch::Tensor rand_nums;
  if (use_rand){
    rand_nums = torch::rand({nqueries,1,1},options);
  }else{
    rand_nums = torch::zeros({nqueries,1,1},options);
  }
  fprintf(stdout,"nthreads_blocks,nthreads_color,nblocks,use_rand: %d,%d,%d,%d\n",
          nthreads_blocks,nthreads_color,nblocks,use_rand);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_gather_forward_kernel_race", ([&] {
    dnls_gather_forward_kernel_race<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        wvid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dilation,qpt,cpt);
      }));
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t>
__global__ void dnls_gather_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    int dilation, int qpt, int kpt) {

    // -- shapes --
    int nframes = grad_vid.size(0);
    int colors = grad_vid.size(1);
    int height = grad_vid.size(2);
    int width = grad_vid.size(3);
    int nq = patches.size(0);
    int k = patches.size(1);
    int pt = patches.size(2);
    int ps = patches.size(4);
    int psHalf = (int)ps/2;

    // -- cuda threads --
    int pi = threadIdx.y;
    int pj = threadIdx.z;

    // -- batching --
    int query_start = blockIdx.x*qpt;
    int k_start = threadIdx.x*kpt;

    // inits
    int qi,ki,ti,hi,wi;
    int vi_h,vi_w,vi_t;
    bool valid_hw,valid_t,valid;
    scalar_t pix;

    // -- range --
    for(int _qi = 0; _qi < qpt; _qi++){

      // -- query index --
      qi = query_start + _qi;
      if (qi >= nq){ continue; }

      for(int _ki = 0; _ki < kpt; _ki++){

        // -- k index --
        ki = k_start + _ki;
        if (ki >= k){ continue; }

        // -- fill --
        ti = inds[qi][ki][0];
        hi = inds[qi][ki][1];
        wi = inds[qi][ki][2];

        // -- fill across cuda threads --
        vi_h = hi+dilation*(pi - psHalf);
        vi_w = wi+dilation*(pj - psHalf);
        // vi_h = bounds(hi+dilation*(pi - psHalf),height);
        // vi_w = bounds(wi+dilation*(pj - psHalf),width);

        // -- spatially valid --
        valid_hw = (vi_h >= 0) && (vi_h < height);
        valid_hw = valid_hw && (vi_w >= 0) && (vi_w < width);

        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- check valid --
          vi_t = bounds(ti + pk,0,nframes);
          valid_t = (vi_t >= 0) && (vi_t < nframes);
          valid = valid_hw && valid_t;

          // -- colors --
          for(int ci = 0; ci < colors; ci++){
            if (valid){
              pix = grad_vid[vi_t][ci][vi_h][vi_w];
            }else{
              pix = 0.;
            }
            patches[qi][ki][pk][ci][pi][pj] = pix;
          }
        }
      }
    }
}

void dnls_cuda_gather_backward(
  torch::Tensor grad_vid,torch::Tensor patches,torch::Tensor inds,
  int dilation) {

  // -- kernel blocks --
  int nqueries = inds.size(0);
  int k = inds.size(1);
  int qpt = 10;
  int nblocks = (nqueries-1)/qpt+1;

  // -- kernel threads --
  int ps = patches.size(5);
  int MAX_THREADS = 1024;
  int dim = ps*ps;
  int kpb = MAX_THREADS/dim; // num of "k" managed per block
  int kpt = ((k - 1)/kpb) + 1; // num of "k" per thread
  dim3 nthreads(kpb,ps,ps);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_gather_backward_kernel", ([&] {
    dnls_gather_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        grad_vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        dilation,qpt,kpt);
  }));

}
