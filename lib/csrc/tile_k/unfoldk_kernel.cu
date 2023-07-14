
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
__global__ void stnls_unfoldk_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> inds,
    int dilation, int ps_offset, bool reflect, int qpt, int kpt){

    // -- shapes --
    int bsize = vid.size(0);
    int nframes = vid.size(1);
    int colors = vid.size(2);
    int height = vid.size(3);
    int width = vid.size(4);
    int nq = patches.size(1);
    int k = patches.size(2);
    int pt = patches.size(3);
    int ps = patches.size(6);
    int psHalf = (int)ps/2;

    // -- cuda threads --
    int pi = threadIdx.y;
    int pj = threadIdx.z;

    // -- batching --
    int query_start = blockIdx.x*qpt;
    int k_start = threadIdx.x*kpt;
    int bi = blockIdx.y;

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
        ti = inds[bi][qi][ki][0];
        hi = inds[bi][qi][ki][1];
        wi = inds[bi][qi][ki][2];

        // -- fill across cuda threads --
        vi_h = hi+dilation*(pi - ps_offset);
        vi_w = wi+dilation*(pj - ps_offset);
        vi_h = reflect ? bounds(vi_h,height) : vi_h;
        vi_w = reflect ? bounds(vi_w,width) : vi_w;

        // -- spatially valid --
        valid_hw = (vi_h >= 0) && (vi_h < height);
        valid_hw = valid_hw && (vi_w >= 0) && (vi_w < width);

        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- check valid --
          vi_t = bounds(ti + pk,nframes);
          valid_t = (vi_t >= 0) && (vi_t < nframes);
          valid = valid_hw && valid_t;
          if (!valid){ continue; }

          // -- colors --
          for(int ci = 0; ci < colors; ci++){
            pix = vid[bi][vi_t][ci][vi_h][vi_w];
            patches[bi][qi][ki][pk][ci][pi][pj] = pix;
          }
          
        }
      }
    }
}

void stnls_cuda_unfoldk_forward(
    torch::Tensor vid, torch::Tensor patches, torch::Tensor inds,
    int dilation, int adj, bool reflect) {

  // -- indexing --
  int ps = patches.size(6);
  int ps_offset = (adj > 0) ? 0 : ps/2;

  // -- kernel blocks --
  int bsize = inds.size(0);
  int numQueries = inds.size(1);
  int k = inds.size(2);
  int qpt = 10;
  int nblocks_queries = (numQueries-1)/qpt+1;
  dim3 nblocks(nblocks_queries,bsize);

  // -- kernel threads --
  int MAX_THREADS = 1024;
  int dim = ps*ps;
  int kpb = MAX_THREADS/dim; // num of "k" managed per block
  int kpt = ((k - 1)/kpb) + 1; // num of "k" per thread
  dim3 nthreads(kpb,ps,ps);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "stnls_unfoldk_forward_kernel", ([&] {
    stnls_unfoldk_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        dilation, ps_offset, reflect, qpt, kpt);
    }));
}


/****************************

   Backward Pass (Simple)

****************************/


template <typename scalar_t, bool USE_ATOMIC>
__global__ void stnls_unfoldk_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> grad_patches,
    const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> inds,
    // const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int dilation, int adj, bool reflect, int qpt, int cpt){

  // shape
  int bsize =  grad_patches.size(0);
  int nq =    grad_patches.size(1);
  int k =     grad_patches.size(2);
  int pt =    grad_patches.size(3);
  int colors = grad_patches.size(4);
  int ps =    grad_patches.size(5);
  int nframes = vid.size(1);
  int qi,ti,hi,wi;
  float pix;
  int height = vid.size(3);
  int width = vid.size(4);
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
  int q_start = qpt * (thread_x + block_x * blockDim.x);
  int bi = blockIdx.y;
  
  for (int _qi = 0; _qi < qpt; _qi++){
    qi = q_start + _qi;
    if (qi < nq){
      // iterate
      c0_offset = 0;//__float2int_rd(c0_dist * rand_nums[qi][0][0]);

      for (int ki = 0; ki < k; ki++){
        // c0_offset = (c0_offset + 1) % c0_dist;
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
              ti = bounds(inds[bi][qi][ki][0] + pk,nframes);
              _hi = inds[bi][qi][ki][1] + dilation*(pi - psHalf + adj);
              _wi = inds[bi][qi][ki][2] + dilation*(pj - psHalf + adj);
              hi = reflect ? bounds(_hi,height) : _hi;
              wi = reflect ? bounds(_wi,width) : _wi;
              valid_h = (hi >= 0) && (hi < height);
              valid_w = (wi >= 0) && (wi < width);
              valid = valid_h && valid_w;

              for (int _c0 = c0_start; _c0 < c0_end; _c0++){
                c0 = (_c0 + c0_offset) % c0_dist + c0_start;
                pix = grad_patches[bi][qi][ki][pk][c0][pi][pj];		
                if (valid){
                  if (USE_ATOMIC){
                    atomicAdd(&vid[bi][ti][c0][hi][wi],pix);
                  }else{
                    vid[bi][ti][c0][hi][wi] += pix;
                  }
                }
              }
            }

          }
        }
      }
    }
  }
}


void stnls_cuda_unfoldk_backward(
    torch::Tensor vid, torch::Tensor grad_patches,
    torch::Tensor inds, int dilation, bool exact, int adj,
    bool reflect, bool use_atomic) {

  // unpack params
  int bsize = inds.size(0);
  int numQueries = inds.size(1);
  int k = inds.size(2);
  int pt = grad_patches.size(3);
  int colors = grad_patches.size(4);
  int ps = grad_patches.size(5);
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
  int nblocks_queries = ((numQueries - 1) / total_per_block) + 1;
  nblocks_queries = min(nblocks_queries,max_nblocks);
  int total_pb = (numQueries - 1) / nblocks_queries + 1;
  int bpb = (total_pb-1) / block_threads + 1;

  // if exact
  if (exact){
    cpt = 1;
    nblocks_queries = 1;
    block_threads = 1;
    bpb = numQueries;
  }
  dim3 nblocks(nblocks_queries,bsize);

  // fprintf(stdout,"block_threads,color_threads: %d,%d\n",block_threads,color_threads);
  // fprintf(stdout,"bpb,cpt: %d,%d\n",bpb,cpt);

  // fprintf(stdout,"exact: %d, bpb: %d, nthreads.x .y: %d, %d\n",
  //         exact,bpb,nthreads.x,nthreads.y);

  // -- allocate random memory --
  // auto cu_index = vid.device().index();
  // auto options = torch::TensorOptions().device(torch::kCUDA, cu_index).dtype(torch::kFloat32);
  // torch::Tensor rand_nums = torch::rand({numQueries,1,1},options);

  // launch kernel
  if (use_atomic){
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "stnls_unfoldk_backward_kernel", ([&] {
	stnls_unfoldk_backward_kernel<scalar_t,true><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_patches.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        // rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dilation, adj, reflect, bpb, cpt);
  }));
  }else{
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "stnls_unfoldk_backward_kernel", ([&] {
	stnls_unfoldk_backward_kernel<scalar_t,false><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_patches.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        // rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dilation, adj, reflect, bpb, cpt);
  }));

  }

}

/*********************************************

     Backward Pass (Efficient Attempt)

*********************************************/


template <typename scalar_t>
__global__ void stnls_unfoldk_backward_kernel_eff(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> grad_patches,
    const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> inds,
    int dilation, int adj, bool reflect, int qpt) {

  // shape
  int bsize = grad_patches.size(0);
  int nq =    grad_patches.size(1);
  int k =     grad_patches.size(2);
  int pt =    grad_patches.size(3);
  int color = grad_patches.size(4);
  int ps =    grad_patches.size(5);
  int qi,ti,hi,wi;
  scalar_t pix;
  int height = vid.size(3);
  int width = vid.size(4);
  int psHalf = ps/2;
  bool valid_h,valid_w,valid;
  int _hi,_wi;

  // get indices
  int bi = blockIdx.y;
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
              ti = inds[bi][qi][ki][0] + pk;
              _hi = inds[bi][qi][ki][1] + dilation*(pi - psHalf + adj);
              _wi = inds[bi][qi][ki][2] + dilation*(pj - psHalf + adj);
              hi = bounds(_hi,height);
              wi = bounds(_wi,width);
              // hi = reflect ? bounds(hi,height) : hi;
              // wi = reflect ? bounds(wi,width) : wi;
              valid_h = (hi >= 0) && (hi < height);
              valid_w = (wi >= 0) && (wi < width);
              valid = valid_h && valid_w;
              for (int ci = 0; ci < color; ci++){
                pix = grad_patches[bi][qi][ki][pk][ci][pi][pj];
                if(valid){
                  vid[bi][ti][ci][hi][wi] += pix;
                }
              }
            }
          }
        }
      }
    }
  }
}


void stnls_cuda_unfoldk_backward_eff(
    torch::Tensor vid, torch::Tensor grad_patches,
    torch::Tensor inds, int dilation, bool exact, int adj, bool reflect) {

  // launch params
  int bsize = inds.size(0);
  int numQueries = inds.size(1);
  int k = inds.size(2);
  int pt = grad_patches.size(3);
  int color = grad_patches.size(4);
  int ps = grad_patches.size(5);
  assert(pt == 1);

  int qpt = 10;
  int nthreads = 1024;
  int queries_per_block = nthreads * qpt;
  int nblocks_queries = ((numQueries - 1) / queries_per_block) + 1;
  if (exact){
    nthreads = 1;
    nblocks_queries = 1;
    qpt = numQueries;
  }
  dim3 nblocks(nblocks_queries,bsize);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "stnls_unfoldk_backward_kernel_eff", ([&] {
    stnls_unfoldk_backward_kernel_eff<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_patches.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        dilation, adj, reflect, qpt);
  }));

}


