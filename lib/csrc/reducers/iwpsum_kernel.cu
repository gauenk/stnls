
// #include <torch/extension.h>
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
__global__ void iwpsum_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid2fill,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    int ps, int pt, int h_off, int w_off, int dilation, int adj,
    bool reflect_bounds, int qpt, int cpt){

    // -- shapes --
    int nframes = vid.size(0);
    int colors = vid.size(1);
    int height = vid.size(2);
    int width = vid.size(3);
    int nq = vid2fill.size(0);
    int k = inds.size(1);
    int psHalf = (int)ps/2;
    int center_ti,center_hi,center_wi;

    // -- cuda threads --
    int pi = threadIdx.y;
    int pj = threadIdx.z;

    // -- batching --
    int query_start = blockIdx.x*qpt;
    int c_start = threadIdx.x*cpt;

    // inits
    int qi,ki,ci;
    int ti,hi,wi;
    bool valid_hw,valid_t,valid;
    scalar_t pix,dist;

    // -- range --
    for(int _qi = 0; _qi < qpt; _qi++){

      // -- query index --
      qi = query_start + _qi;
      if (qi >= nq){ continue; }

      for(int ki = 0; ki < k; ki++){

        // -- reference center --
        center_ti = inds[qi][ki][0];
        center_hi = inds[qi][ki][1];
        center_wi = inds[qi][ki][2];
        dist = dists[qi][ki];

        // -- reference patch location --
        if (reflect_bounds){
          hi = bounds((center_hi-h_off)+dilation*(pi - psHalf + adj),height);
          wi = bounds((center_wi-w_off)+dilation*(pj - psHalf + adj),width);
        }else{
          hi = (center_hi-h_off)+dilation*(pi - psHalf + adj);
          wi = (center_wi-w_off)+dilation*(pj - psHalf + adj);
        }

        // -- spatially valid --
        valid_hw = (hi >= 0) && (hi < height);
        valid_hw = valid_hw && (wi >= 0) && (wi < width);

        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- check valid --
          ti = bounds(center_ti + pk,nframes);
          valid_t = (ti >= 0) && (ti < nframes);
          valid = valid_hw && valid_t;

          // -- colors --
          for(int _ci = 0; _ci < cpt; _ci++){

            // -- color index --
            ci = c_start + _ci;

            // -- fill without warp divergence --
            if (valid && (ci < colors)){
              pix = dist*vid[ti][ci][hi][wi];
              // patches[qi][0][pk][ci][pi][pj] += pix;
            }

          }
        }
      }
    }
}

void cuda_iwpsum_forward(
    torch::Tensor vid, torch::Tensor vid2fill,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt,
    int h_off, int w_off, int dilation, int adj, bool reflect_bounds){

  // -- kernel blocks --
  int nqueries = inds.size(0);
  int qpt = 10;
  int nblocks = (nqueries-1)/qpt+1;

  // -- kernel threads --
  int k = inds.size(1);
  int colors = vid.size(1);
  int MAX_THREADS = 1024;
  int dim = ps*ps;
  int cpb = MAX_THREADS/dim; // num of colors per block
  int cpt = ((colors - 1)/cpb) + 1; // num of colors per thread
  dim3 nthreads(cpb,ps,ps);
  // printf("colors: %d, cpt: %d, cpb: %d, ps: %d, nblocks: %d, rbounds: %d\n",
  //        colors,cpt,cpb,ps,nblocks,(int)reflect_bounds);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "iwpsum_forward_kernel", ([&] {
    iwpsum_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        vid2fill.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        ps, pt, h_off, w_off, dilation, adj, reflect_bounds, qpt, cpt);
    }));
}

/********************************

     Backward Pass (for Vid)

********************************/


template <typename scalar_t>
__global__ void iwpsum_backward_vid_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid2fill_grad,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int ps, int pt,
    int h_off, int w_off, int dilation, int adj, bool reflect_bounds, int qpt, int cpt){

  // shape
  int nq =    inds.size(0);
  int k =     inds.size(1);
  int colors = vid_grad.size(1);
  int qi,ti,hi,wi;
  float weight,pix;
  int height = vid_grad.size(2);
  int width = vid_grad.size(3);
  int psHalf = ps/2;
  bool valid_h,valid_w,valid;
  int center_ti,center_hi,center_wi;

  // color indices
  int c0_start = threadIdx.y*cpt;
  int c0_end = min(c0_start + cpt,colors);
  int c0 = 0;
  int c0_offset = 0;
  int c0_dist = c0_end - c0_start;

  // block indices
  int thread_x = threadIdx.x;
  int block_x = blockIdx.x;
  int q_start = qpt*( thread_x + block_x * blockDim.x);
  
  for (int _qi = 0; _qi < qpt; _qi++){
    qi = q_start + _qi;
    if (qi < nq){
      c0_offset = __float2int_rd(c0_dist * rand_nums[qi][0][0]);
      // iterate
      for (int ki = 0; ki < k; ki++){
        c0_offset = (c0_offset + 1) % c0_dist;
        center_ti = inds[qi][ki][0];
        center_hi = inds[qi][ki][1];
        center_wi = inds[qi][ki][2];
        for (int pk = 0; pk < pt; pk++){
          for (int pi = 0; pi < ps; pi++){
            for (int pj = 0; pj < ps; pj++){
              ti = center_ti + pk;
              hi = (center_hi-h_off) + dilation*(pi - psHalf + adj);
              wi = (center_wi-w_off) + dilation*(pj - psHalf + adj);
              hi = reflect_bounds ? bounds(hi,height) : hi;
              wi = reflect_bounds ? bounds(wi,width) : wi;
              valid_h = (hi >= 0) && (hi < height);
              valid_w = (wi >= 0) && (wi < width);
              valid = valid_h && valid_w;
              weight = dists[qi][ki];
              for (int _c0 = c0_start; _c0 < c0_end; _c0++){
                c0 = (_c0 + c0_offset) % c0_dist + c0_start;
                pix = weight;// * patches_grad[qi][0][pk][c0][pi][pj];
                if (valid){
                  vid_grad[ti][c0][hi][wi] += pix;
                }
              }
            }
          }
        }
      }
    }
  }
}

void cuda_iwpsum_backward_vid(
    torch::Tensor vid_grad, torch::Tensor vid2fill_grad, 
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt,
    int h_off, int w_off, int dilation, int adj, bool reflect_bounds, bool exact){

  // unpack params
  int numQueries = inds.size(0);
  int k = dists.size(1);
  int colors = vid_grad.size(1);
  assert(pt == 1);

  // num of threads
  int max_nthreads = 1024;
  int color_threads = 1;
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

  // exact gradient
  if (exact){
    cpt = 1;
    nblocks = 1;
    block_threads = 1;
    bpb = numQueries;
  }

  // -- viz --
  // fprintf(stdout,"nblocks,block_threads,color_threads: %d,%d,%d\n",nblocks,block_threads,color_threads);
  // fprintf(stdout,"bpb,cpt: %d,%d\n",bpb,cpt);

  // -- allocate random memory --
  auto cu_index = vid_grad.device().index();
  auto options = torch::TensorOptions().device(torch::kCUDA, cu_index).dtype(torch::kFloat32);
  torch::Tensor rand_nums = torch::rand({numQueries,1,1},options);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid_grad.type(), "iwpsum_backward_vid_kernel", ([&] {
    iwpsum_backward_vid_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        vid2fill_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        ps, pt, h_off,w_off,dilation, adj, reflect_bounds, bpb, cpt);
  }));
    
}


/********************************

    Backward Pass (for Dists)

********************************/


template <typename scalar_t>
__global__ void iwpsum_backward_dists_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dists_grad,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid2fill_grad,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    int ps, int pt, int h_off, int w_off, int dilation, int adj, bool reflect_bounds){

  // -- shapes --
  int nq = dists_grad.size(0);
  int k = dists_grad.size(1);
  int colors = vid2fill_grad.size(1);
  int height = vid.size(2);
  int width = vid.size(3);
  int psHalf = ps/2;

  // -- init registers --
  int ti,hi,wi;
  float pix_n,pix_m;
  bool valid_h,valid_w,valid;

  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;

  if ((qi < nq) && (ki < k)) { // -- if valid --
    int center_ti = inds[qi][ki][0];
    int center_hi = inds[qi][ki][1];
    int center_wi = inds[qi][ki][2];
    for (int pk = 0; pk < pt; pk++){
      ti = center_ti + pk;
      for (int pi = 0; pi < ps; pi++){
        hi = (center_hi-h_off) + dilation*(pi - psHalf + adj);
        hi = reflect_bounds ? bounds(hi,height) : hi;
        valid_h = (hi >= 0) && (hi < height);
        for (int pj = 0; pj < ps; pj++){
          wi = (center_wi-w_off) + dilation*(pj - psHalf + adj);
          wi = reflect_bounds ? bounds(wi,width) : wi;
          valid_w = (wi >= 0) && (wi < width);
          valid = valid_h && valid_w;
          for (int c0 = 0; c0 < colors; c0++){
              pix_n = 0.;//patches_grad[qi][0][pk][c0][pi][pj];
              pix_m = valid ? vid[ti][c0][hi][wi] : 0;
              dists_grad[qi][ki] += valid ? pix_n * pix_m : 0.;
          }
        }
      }
    }
  }

}

void cuda_iwpsum_backward_dists(
    torch::Tensor dists_grad, torch::Tensor vid2fill_grad,
    torch::Tensor vid, torch::Tensor inds,
    int ps, int pt, int h_off, int w_off, int dilation, int adj,
    bool reflect_bounds, bool exact){

  // const int NQ,NK = 4,4;
  int nq = dists_grad.size(0);
  int k = dists_grad.size(1);
  dim3 threadsPerBlock(32,32);
  dim3 blocksPerGrid(1, 1);
  blocksPerGrid.x = ceil(double(nq)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));


  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "iwpsum_backward_dists_kernel", ([&] {
    iwpsum_backward_dists_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
        dists_grad.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        vid2fill_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        ps,pt,h_off,w_off,dilation, adj, reflect_bounds);
  }));
    
}

