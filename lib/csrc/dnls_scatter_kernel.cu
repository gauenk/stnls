
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/****************************

       Helper Funcs

****************************/

__inline__ __device__ int bounds(int val, int lim ){
  if (val < 0){
    val = -val;
  }else if (val >= lim){
    val = 2*lim - val - 2;
  }
  return val;
}

/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void dnls_scatter_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> nlInds,
    int dilation, int qpt, int kpt){

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
        ti = nlInds[qi][ki][0];
        hi = nlInds[qi][ki][1];
        wi = nlInds[qi][ki][2];

        // -- fill across cuda threads --
        vi_h = bounds(hi+dilation*(pi - psHalf),height);
        vi_w = bounds(wi+dilation*(pj - psHalf),width);

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

void dnls_cuda_scatter_forward(
    torch::Tensor vid, torch::Tensor patches, torch::Tensor nlInds,
    int dilation) {

  // -- kernel blocks --
  int numQueries = nlInds.size(0);
  int k = nlInds.size(1);
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
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "dnls_scatter_forward_kernel", ([&] {
    dnls_scatter_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        nlInds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        dilation, qpt, kpt);
    }));
}

/****************************

   Backward Pass

****************************/


template <typename scalar_t>
__global__ void dnls_scatter_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_patches,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> nlDists,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> nlInds,
    int dilation, float lam, int qpt) {

  // shape
  int nq =    grad_patches.size(0);
  int k =     grad_patches.size(1);
  int pt =    grad_patches.size(2);
  int color = grad_patches.size(3);
  int ps =    grad_patches.size(4);
  int qi,ti,hi,wi;
  float weight;
  int height = vid.size(2);
  int width = vid.size(3);
  int psHalf = ps/2;

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
              ti = nlInds[qi][ki][0] + pk;
              hi = nlInds[qi][ki][1] + dilation*(pi - psHalf);
              wi = nlInds[qi][ki][2] + dilation*(pj - psHalf);
              hi = bounds(hi,height);
              wi = bounds(wi,width);
              weight = __expf(-lam * nlDists[qi][ki]);
              for (int ci = 0; ci < color; ci++){
                vid[ti][ci][hi][wi] += weight * grad_patches[qi][ki][pk][ci][pi][pj];
              }
            }
          }
        }
      }
    }
  }
}


void dnls_cuda_scatter_backward(
    torch::Tensor grad_patches, torch::Tensor vid,
    torch::Tensor nlDists, torch::Tensor nlInds,
    int dilation, float lam, bool exact) {

  // launch params
  int numQueries = nlInds.size(0);
  int k = nlDists.size(1);
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
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "dnls_scatter_backward_kernel", ([&] {
    dnls_scatter_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        grad_patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nlDists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        nlInds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        dilation, lam, qpt);
  }));

}


/****************************

   Backward Pass (Simple)

****************************/


template <typename scalar_t>
__global__ void dnls_scatter_backward_kernel_simple(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_patches,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> nlDists,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> nlInds,
    int dilation, float lam, int qpt) {

  // shape
  int nq =    grad_patches.size(0);
  int k =     grad_patches.size(1);
  int pt =    grad_patches.size(2);
  int color = grad_patches.size(3);
  int ps =    grad_patches.size(4);
  int qi,ti,hi,wi;
  float weight;
  int height = vid.size(2);
  int width = vid.size(3);
  int psHalf = ps/2;

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
              ti = nlInds[qi][ki][0] + pk;
              hi = nlInds[qi][ki][1] + dilation*(pi - psHalf);
              wi = nlInds[qi][ki][2] + dilation*(pj - psHalf);
              hi = bounds(hi,height);
              wi = bounds(wi,width);
              weight = __expf(-lam * nlDists[qi][ki]);
              for (int ci = 0; ci < color; ci++){
                vid[ti][ci][hi][wi] += weight * grad_patches[qi][ki][pk][ci][pi][pj];
              }
            }
          }
        }
      }
    }
  }
}


void dnls_cuda_scatter_backward_simple(
    torch::Tensor grad_patches, torch::Tensor vid,
    torch::Tensor nlDists, torch::Tensor nlInds,
    int dilation, float lam, bool exact) {

  // launch params
  int numQueries = nlInds.size(0);
  int k = nlDists.size(1);
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
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "dnls_scatter_backward_kernel_simple", ([&] {
    dnls_scatter_backward_kernel_simple<scalar_t><<<nblocks, nthreads>>>(
        grad_patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nlDists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        nlInds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        dilation, lam, qpt);
  }));

}
