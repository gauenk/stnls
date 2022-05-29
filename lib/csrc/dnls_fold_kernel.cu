
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
__global__ void dnls_fold_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    int qStart, int qStride, int dilation, int qpt) {
}

void dnls_cuda_fold_forward(
    torch::Tensor vid, torch::Tensor patches,
    int qStart, int qStride, int dilation){

  // launch params
  int numQueries = patches.size(0);
  int k = 1;
  int pt = patches.size(2);
  int color = patches.size(3);
  int ps = patches.size(4);
  assert(pt == 1);

  int qpt = 10;
  int nthreads = 1024;
  int queries_per_block = nthreads * qpt;
  int nblocks = ((numQueries - 1) / queries_per_block) + 1;

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_fold_forward_kernel", ([&] {
    dnls_fold_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        qStart,qStride,dilation,qpt);
      }));
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t>
__global__ void dnls_fold_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    int qStart, int qStride, int dilation, int qpt, int kpt) {

  
}

void dnls_cuda_fold_backward(
  torch::Tensor grad_vid,torch::Tensor patches,
  int qStart, int qStride, int dilation) {

  // -- kernel blocks --
  int numQueries = patches.size(0);
  int k = 1;
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
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_fold_backward_kernel", ([&] {
    dnls_fold_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        grad_vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        qStart,qStride,dilation,qpt,kpt);
  }));

}
