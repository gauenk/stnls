
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void dnls_gather_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> wvid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> nlInds,
    float lam, int dilation) {

  // column index
  const int n = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  

}

void dnls_cuda_gather_forward(
    torch::Tensor vid,torch::Tensor patches,torch::Tensor nlInds) {

  // tmp
  float lam;
  int dilation;
  lam = 0.;
  dilation = 0;
  fprintf(stdout,"hi!\n");

  // launch params
  int numQueries = 10;//nlInds.size(0);
  const int threads = 1024;
  const dim3 blocks((numQueries - 1) / threads + 1);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_gather_forward_kernel", ([&] {
    dnls_gather_forward_kernel<scalar_t><<<blocks, threads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        nlInds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        lam,dilation);
      }));
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t>
__global__ void dnls_gather_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> nlInds) {

  //batch index
  const int n = blockIdx.y;

  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;

}

void dnls_cuda_gather_backward(
    torch::Tensor grad_vid,torch::Tensor patches,torch::Tensor nlInds) {

  // launch params
  int numQueries = 10;//nlInds.size(0);
  const int threads = 1024;
  const dim3 blocks((numQueries - 1) / threads + 1);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_gather_backward_kernel", ([&] {
    dnls_gather_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        nlInds.packed_accessor32<int,3,torch::RestrictPtrTraits>());
  }));

}
