
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
__global__ void wpsum_heads_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int h_off, int w_off, int dilation, int adj, bool reflect_bounds, int qpt, int cpt){

    // -- shapes --
    int nheads = dists.size(1);
    int nframes = vid.size(2);
    int colors = vid.size(3);
    int height = vid.size(4);
    int width = vid.size(5);
    int nq = inds.size(2);
    int k = inds.size(3);
    int pt = patches.size(3);
    int ps = patches.size(5);
    int psHalf = (int)ps/2;
    int center_ti,center_hi,center_wi;

    // -- head indices --
    int head_index = blockIdx.y;
    int vid_nheads = vid.size(1);
    int inds_nheads = inds.size(1);
    int vid_head,inds_head;
    inds_head = head_index % inds_nheads;
    vid_head = head_index % vid_nheads;

    // -- cuda threads --
    int pi = threadIdx.y;
    int pj = threadIdx.z;

    // -- batching --
    int query_start = blockIdx.x*qpt;
    int c_start = threadIdx.x*cpt;
    int bi = blockIdx.z;

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
        center_ti = inds[bi][inds_head][qi][ki][0];
        center_hi = inds[bi][inds_head][qi][ki][1];
        center_wi = inds[bi][inds_head][qi][ki][2];
        dist = dists[bi][head_index][qi][ki];

        // -- reference patch location --
        hi = (center_hi-h_off)+dilation*(pi - psHalf + adj);
        wi = (center_wi-w_off)+dilation*(pj - psHalf + adj);
        if (reflect_bounds){
          hi = bounds(hi,height);
          wi = bounds(wi,width);
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
              pix = dist*vid[bi][vid_head][ti][ci][hi][wi];
              patches[bi][head_index][qi][pk][ci][pi][pj] += pix;
            }

          }
        }
      }
    }
}

void cuda_wpsum_heads_forward(
    torch::Tensor vid, torch::Tensor patches,
    torch::Tensor dists, torch::Tensor inds,
    int h_off, int w_off, int dilation, int adj, bool reflect_bounds){

  // -- unpack --
  int bsize = dists.size(0);
  int nheads = dists.size(1);
  int nqueries = inds.size(2);
  int k = inds.size(3);
  int colors = vid.size(3);
  int ps = patches.size(6);
  
  // -- kernel blocks --
  int qpt = 1;
  int query_nblocks = (nqueries-1)/qpt+1;
  int head_nblocks = nheads;
  dim3 nblocks(query_nblocks,head_nblocks,bsize);

  // -- kernel threads --
  int MAX_THREADS = 1024;
  int dim = ps*ps;
  int color_nthreads = MAX_THREADS/dim; // num of colors per block
  color_nthreads = min(color_nthreads,colors);
  int cpt = ((colors - 1)/color_nthreads) + 1; // num of colors per thread
  dim3 nthreads(color_nthreads,ps,ps);
  // printf("colors: %d, cpt: %d, color_nthreads: %d, ps: %d, nblocks: %d,%d, rbounds: %d\n",
  //        colors,cpt,color_nthreads,ps,query_nblocks,head_nblocks,reflect_bounds);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "wpsum_heads_forward_kernel", ([&] {
    wpsum_heads_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        h_off, w_off, dilation, adj, reflect_bounds, qpt, cpt);
    }));
}

/********************************

     Backward Pass (for Vid)

********************************/


template <typename scalar_t>
__global__ void wpsum_heads_backward_vid_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid_grad,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> patches_grad,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> rand_nums,
    int h_off, int w_off, int dilation, int adj, bool reflect_bounds,
    int qpt, int hpb, int cpt){

  // shape
  int qi,ti,hi,wi;
  float weight,pix;
  int nheads = dists.size(1);
  int nq =    patches_grad.size(2);
  int k =     inds.size(3);
  int pt =    patches_grad.size(3);
  int colors = patches_grad.size(4);
  int ps =    patches_grad.size(5);
  int height = vid_grad.size(4);
  int width = vid_grad.size(5);
  int psHalf = ps/2;
  bool valid_h,valid_w,valid;
  int center_ti,center_hi,center_wi;
  float rand_num;

  // color indices
  int c0_start = threadIdx.y*cpt;
  int c0_end = min(c0_start + cpt,colors);
  int c0 = 0;
  int c0_offset = 0;
  c0_start = 0;
  c0_end = colors;
  int c0_dist = c0_end - c0_start;

  // -- head indices --
  int head_start = blockIdx.y * hpb;
  int head_end = min(head_start + hpb,nheads);
  int vid_nheads = vid_grad.size(1);
  int inds_nheads = inds.size(1);
  int vid_head,inds_head;

  // -- batch --
  int bi = blockIdx.z;

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
        for (int head_index = head_start; head_index < head_end; head_index++){
          inds_head = head_index % inds_nheads;
          vid_head = head_index % vid_nheads;

          center_ti = inds[bi][inds_head][qi][ki][0];
          center_hi = inds[bi][inds_head][qi][ki][1];
          center_wi = inds[bi][inds_head][qi][ki][2];
          weight = dists[bi][head_index][qi][ki];

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
                for (int _c0 = c0_start; _c0 < c0_end; _c0++){
                  c0 = (_c0 + c0_offset + head_index) % c0_dist + c0_start;
                  pix = weight * patches_grad[bi][head_index][qi][pk][c0][pi][pj];
                  if (valid){
                    vid_grad[bi][vid_head][ti][c0][hi][wi] += pix;
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

void cuda_wpsum_heads_backward_vid(
    torch::Tensor vid_grad, torch::Tensor patches_grad, 
    torch::Tensor dists, torch::Tensor inds,
    int h_off, int w_off, int dilation, int adj,
    bool reflect_bounds, bool use_rand, bool exact){

  // unpack params
  int bsize = dists.size(0);
  int nheads = dists.size(1);
  int numQueries = inds.size(2);
  int k = dists.size(3);
  int pt = patches_grad.size(3);
  int colors = patches_grad.size(4);
  int ps = patches_grad.size(5);
  assert(pt == 1);

  // num of threads
  int max_nthreads = 1024;
  int color_threads = 1;
  int block_threads = max_nthreads/color_threads;
  int cpt = (colors-1)/color_threads+1;
  block_threads = exact ? 1 : block_threads;
  color_threads = exact ? colors : color_threads;
  dim3 nthreads(block_threads,color_threads);

  // -- head blocks --
  int nheads_vid = vid_grad.size(1);
  // bool span_heads = nheads_vid == nheads;
  // span_heads = span_heads || !exact;
  // int head_nblocks = span_heads ? nheads : 1;
  // int hpb = exact ? nheads : 1;
  int head_nblocks = nheads_vid;
  int hpb = 1;

  // -- query blocks --
  int max_nblocks = 32;
  int num_per_block = 16;
  int total_per_block = block_threads * num_per_block;
  int query_nblocks = ((numQueries - 1) / total_per_block) + 1;
  query_nblocks = min(query_nblocks,max_nblocks);
  int total_pb = (numQueries - 1) / query_nblocks + 1;
  int bpb = (total_pb-1) / block_threads + 1;

  // exact gradient
  if (exact){
    cpt = 1;
    query_nblocks = 1;
    block_threads = 1;
    bpb = numQueries;
  }

  // -- decl blocks --
  dim3 nblocks(query_nblocks,head_nblocks,bsize);

  // -- viz --
  // fprintf(stdout,"nblocks,block_threads,color_threads: %d,%d,%d\n",nblocks,block_threads,color_threads);
  // fprintf(stdout,"bpb,cpt: %d,%d\n",bpb,cpt);
  // fprintf(stdout,"ps: %d\n",ps);

  // -- allocate random memory --
  auto cu_index = vid_grad.device().index();
  auto options = torch::TensorOptions().device(torch::kCUDA,
                                               cu_index).dtype(torch::kFloat32);
  torch::Tensor rand_nums;
  if (use_rand){
    rand_nums = torch::rand({numQueries,1,1},options);
  }else{
    rand_nums = torch::zeros({numQueries,1,1},options);
  }

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid_grad.type(), "wpsum_heads_backward_vid_kernel", ([&] {
    wpsum_heads_backward_vid_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        patches_grad.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        rand_nums.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        h_off,w_off,dilation, adj, reflect_bounds, bpb, hpb, cpt);
  }));
    
}


/********************************

    Backward Pass (for Dists)

********************************/


template <typename scalar_t>
__global__ void wpsum_heads_backward_dists_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists_grad,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> patches_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int h_off, int w_off, int dilation, int adj, bool reflect_bounds){

  // -- shapes --
  int bsize = dists_grad.size(0);
  int nq = dists_grad.size(2);
  int k = dists_grad.size(3);
  int pt =    patches_grad.size(3);
  int colors = patches_grad.size(4);
  int ps =    patches_grad.size(5);
  int height = vid.size(4);
  int width = vid.size(5);
  int psHalf = ps/2;

  // -- init registers --
  int ti,hi,wi;
  float pix_n,pix_m;
  bool valid_h,valid_w,valid;

  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int head_index = blockIdx.z/bsize;
  int bi = blockIdx.z % bsize;

  // -- head indices --
  int vid_nheads = vid.size(1);
  int inds_nheads = inds.size(1);
  int vid_head = head_index % vid_nheads;
  int inds_head = head_index % inds_nheads;

  if ((qi < nq) && (ki < k)) { // -- if valid --
    int center_ti = inds[bi][inds_head][qi][ki][0];
    int center_hi = inds[bi][inds_head][qi][ki][1];
    int center_wi = inds[bi][inds_head][qi][ki][2];
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
              pix_n = patches_grad[bi][head_index][qi][pk][c0][pi][pj];
              pix_m = valid ? vid[bi][vid_head][ti][c0][hi][wi] : 0;
              dists_grad[bi][head_index][qi][ki] += valid ? pix_n * pix_m : 0.;
          }
        }
      }
    }
  }

}

void cuda_wpsum_heads_backward_dists(
    torch::Tensor dists_grad, torch::Tensor patches_grad,
    torch::Tensor vid, torch::Tensor inds,
    int h_off, int w_off, int dilation, int adj, bool reflect_bounds, bool exact){

  // const int NQ,NK = 4,4;
  int bsize = dists_grad.size(0);
  int nheads = dists_grad.size(1);
  int nq = dists_grad.size(2);
  int k = dists_grad.size(3);
  dim3 threadsPerBlock(32,32);
  dim3 blocksPerGrid(1, 1, nheads*bsize);
  blocksPerGrid.x = ceil(double(nq)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
  // blocksPerGrid.z = nheads;
  // fprintf(stdout,"bsize,nheads,nq,k: %d,%d,%d,%d\n",bsize,nheads,nq,k);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "wpsum_heads_backward_dists_kernel", ([&] {
    wpsum_heads_backward_dists_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
        dists_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches_grad.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        h_off,w_off,dilation, adj, reflect_bounds);
  }));
    
}




