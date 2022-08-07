
// #include <torch/extension.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/****************************

       Helper Funcs

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

__inline__ int cpu_bounds(int val, int lim ){
  int vval = val;
  if (val < 0){
    vval = -val;
  }else if (val >= lim){
    vval = 2*(lim-1) - val;
  }
  return vval;
}

/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void dnls_unfoldk_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    int dilation, int adj, bool use_bounds, int qpt, int kpt){

    // -- shapes --
    int nframes = vid.size(0);
    int colors = vid.size(1);
    int height = vid.size(2);
    int width = vid.size(3);
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
        if (use_bounds){
          vi_h = bounds(hi+dilation*(pi - psHalf + adj),height);
          vi_w = bounds(wi+dilation*(pj - psHalf + adj),width);
        }else{
          vi_h = hi+dilation*(pi - psHalf + adj);
          vi_w = wi+dilation*(pj - psHalf + adj);
        }

        // -- spatially valid --
        valid_hw = (vi_h >= 0) && (vi_h < height);
        valid_hw = valid_hw && (vi_w >= 0) && (vi_w < width);

        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- check valid --
          vi_t = bounds(ti + pk,nframes);
          valid_t = (vi_t >= 0) && (vi_t < nframes);
          valid = valid_hw && valid_t;

          // -- colors --
          for(int ci = 0; ci < colors; ci++){
            if (valid){
              pix = vid[vi_t][ci][vi_h][vi_w];
            }else{
              pix = 0.;
            }
            patches[qi][ki][pk][ci][pi][pj] = pix;
          }
        }
      }
    }
}

void dnls_cuda_unfoldk_forward(
    torch::Tensor vid, torch::Tensor patches, torch::Tensor inds,
    int dilation, int adj, bool use_bounds) {

  // -- kernel blocks --
  int numQueries = inds.size(0);
  int k = inds.size(1);
  int qpt = 10;
  int nblocks = (numQueries-1)/qpt+1;

  // -- kernel threads --
  int ps = patches.size(5);
  int MAX_THREADS = 1024;
  int dim = ps*ps;
  int kpb = MAX_THREADS/dim; // num of "k" managed per block
  int kpt = ((k - 1)/kpb) + 1; // num of "k" per thread
  dim3 nthreads(kpb,ps,ps);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "dnls_unfoldk_forward_kernel", ([&] {
    dnls_unfoldk_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        dilation, adj, use_bounds, qpt, kpt);
    }));
}


/****************************

   Backward Pass (Simple)

****************************/


template <typename scalar_t>
__global__ void dnls_unfoldk_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_patches,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int dilation, int adj, bool use_bounds, int qpt, int cpt){

  // shape
  int nq =    grad_patches.size(0);
  int k =     grad_patches.size(1);
  int pt =    grad_patches.size(2);
  int colors = grad_patches.size(3);
  int ps =    grad_patches.size(4);
  int nframes = vid.size(0);
  int qi,ti,hi,wi;
  float pix;
  int height = vid.size(2);
  int width = vid.size(3);
  int psHalf = ps/2;
  bool valid_h,valid_w,valid;
  int pi_offset,pj_offset;
  int _hi,_wi;
  int pi,pj;
    
  // color indices
  int c0_start = threadIdx.y*cpt;
  int c0_end = min(c0_start + cpt,colors);
  int c0 = 0;
  int c0_dist = c0_end - c0_start;
  int c0_offset = 0;
  // int c0_offset = (threadIdx.x + blockIdx.x) % c0_dist;

  // block indices
  int thread_x = threadIdx.x;
  int block_x = blockIdx.x;
  int q_start = qpt*( thread_x + block_x * blockDim.x);
  
  for (int _qi = 0; _qi < qpt; _qi++){
    qi = q_start + _qi;
    if (qi < nq){
      // iterate
      c0_offset = __float2int_rd(c0_dist * rand_nums[qi][0][0]);

      for (int ki = 0; ki < k; ki++){
        c0_offset = (c0_offset + 1) % c0_dist;
        // c0_offset = __float2int_rd(c0_dist * rand_nums[qi][ki][0]);
        pi_offset = 0;//__float2int_rd(ps * rand_nums[qi][0][0]);
        pj_offset = 0;//__float2int_rd(ps * rand_nums[qi][0][0]);

        for (int pk = 0; pk < pt; pk++){
          for (int _pi = 0; _pi < ps; _pi++){
            for (int _pj = 0; _pj < ps; _pj++){

              // -- compute patch position with offsets --
              pi = (_pi + pi_offset) % ps;
              pj = (_pj + pj_offset) % ps;
              
              // -- standard accumulation --
              ti = bounds(inds[qi][ki][0] + pk,nframes);
              _hi = inds[qi][ki][1] + dilation*(pi - psHalf + adj);
              _wi = inds[qi][ki][2] + dilation*(pj - psHalf + adj);
              hi = use_bounds ? bounds(_hi,height) : _hi;
              wi = use_bounds ? bounds(_wi,width) : _wi;
              valid_h = (hi >= 0) && (hi < height);
              valid_w = (wi >= 0) && (wi < width);
              valid = valid_h && valid_w;

              for (int _c0 = c0_start; _c0 < c0_end; _c0++){
                c0 = (_c0 + c0_offset) % c0_dist + c0_start;
                pix = grad_patches[qi][ki][pk][c0][pi][pj];
                if (valid){
                  vid[ti][c0][hi][wi] += pix;
                }
              }
            }
            
          }
        }
      }
    }
  }
}


void dnls_cuda_unfoldk_backward(
    torch::Tensor vid, torch::Tensor grad_patches,
    torch::Tensor inds, int dilation, bool exact, int adj, bool use_bounds) {

  // unpack params
  int numQueries = inds.size(0);
  int k = inds.size(1);
  int pt = grad_patches.size(2);
  int colors = grad_patches.size(3);
  int ps = grad_patches.size(4);
  assert(pt == 1);

  // num of threads
  int max_nthreads = 1024;
  int color_threads = 1;//min(8,colors);
  int block_threads = max_nthreads/color_threads;
  int cpt = (colors-1)/color_threads+1;
  block_threads = exact ? 1 : block_threads;
  color_threads = exact ? colors : color_threads;
  dim3 nthreads = dim3(block_threads,color_threads);

  // num of blocks
  int max_nblocks = 32;
  int num_per_block = 16;
  int total_per_block = block_threads * num_per_block;
  int nblocks = ((numQueries - 1) / total_per_block) + 1;
  nblocks = min(nblocks,max_nblocks);
  int total_pb = (numQueries - 1) / nblocks + 1;
  int bpb = (total_pb-1) / block_threads + 1;

  // if exact
  if (exact){
    cpt = 1;
    nblocks = 1;
    block_threads = 1;
    bpb = numQueries;
  }
  // fprintf(stdout,"block_threads,color_threads: %d,%d\n",block_threads,color_threads);
  // fprintf(stdout,"bpb,cpt: %d,%d\n",bpb,cpt);

  // fprintf(stdout,"exact: %d, bpb: %d, nthreads.x .y: %d, %d\n",
  //         exact,bpb,nthreads.x,nthreads.y);

  // -- allocate random memory --
  auto cu_index = vid.device().index();
  auto options = torch::TensorOptions().device(torch::kCUDA, cu_index).dtype(torch::kFloat32);
  torch::Tensor rand_nums = torch::rand({numQueries,1,1},options);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "dnls_unfoldk_backward_kernel", ([&] {
    dnls_unfoldk_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dilation, adj, use_bounds, bpb, cpt);
  }));

}

/*********************************************

     Backward Pass (Efficient Attempt)

*********************************************/


template <typename scalar_t>
__global__ void dnls_unfoldk_backward_kernel_eff(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_patches,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    int dilation, int adj, bool use_bounds, int qpt) {

  // shape
  int nq =    grad_patches.size(0);
  int k =     grad_patches.size(1);
  int pt =    grad_patches.size(2);
  int color = grad_patches.size(3);
  int ps =    grad_patches.size(4);
  int qi,ti,hi,wi;
  scalar_t pix;
  int height = vid.size(2);
  int width = vid.size(3);
  int psHalf = ps/2;
  bool valid_h,valid_w,valid;
  int _hi,_wi;

  // get indices
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;
  int q_start = qpt*(tidx + bidx * blockDim.x);
  
  for (int _qi = 0; _qi < qpt; _qi++){
    qi = q_start + _qi;
    if (qi < nq){
      // iterate
      for (int ki = 0; ki < k; ki++){
        for (int pk = 0; pk < pt; pk++){
          for (int pi = 0; pi < ps; pi++){
            for (int pj = 0; pj < ps; pj++){
              ti = inds[qi][ki][0] + pk;
              _hi = inds[qi][ki][1] + dilation*(pi - psHalf + adj);
              _wi = inds[qi][ki][2] + dilation*(pj - psHalf + adj);
              hi = bounds(_hi,height);
              wi = bounds(_wi,width);
              // hi = use_bounds ? bounds(hi,height) : hi;
              // wi = use_bounds ? bounds(wi,width) : wi;
              valid_h = (hi >= 0) && (hi < height);
              valid_w = (wi >= 0) && (wi < width);
              valid = valid_h && valid_w;
              for (int ci = 0; ci < color; ci++){
                pix = grad_patches[qi][ki][pk][ci][pi][pj];
                if(valid){
                  vid[ti][ci][hi][wi] += pix;
                }
              }
            }
          }
        }
      }
    }
  }
}


void dnls_cuda_unfoldk_backward_eff(
    torch::Tensor vid, torch::Tensor grad_patches,
    torch::Tensor inds, int dilation, bool exact, int adj, bool use_bounds) {

  // launch params
  int numQueries = inds.size(0);
  int k = inds.size(1);
  int pt = grad_patches.size(2);
  int color = grad_patches.size(3);
  int ps = grad_patches.size(4);
  assert(pt == 1);

  int qpt = 10;
  int nthreads = 1024;
  int queries_per_block = nthreads * qpt;
  int nblocks = ((numQueries - 1) / queries_per_block) + 1;
  if (exact){
    nthreads = 1;
    nblocks = 1;
    qpt = numQueries;
  }

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "dnls_unfoldk_backward_kernel_eff", ([&] {
    dnls_unfoldk_backward_kernel_eff<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        dilation, adj, use_bounds, qpt);
  }));

}


