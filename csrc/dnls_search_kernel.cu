
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void dnls_search_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> queryInds,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fflow,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> bflow,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> nlInds,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> nlDists,
    int ws, int wt){

  // column index
  const int n = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;

}

void dnls_cuda_search_forward(
    torch::Tensor vid, torch::Tensor patches, torch::Tensor queryInds,
    torch::Tensor fflow, torch::Tensor bflow,
    torch::Tensor nlInds, torch::Tensor nlDists, int ws,int wt){

  // launch params 
  int numQueries = 10;//nlInds.size(0);
  const int threads = 1024;
  const dim3 blocks((numQueries - 1) / threads + 1);

  // launch kernel
  // AT_DISPATCH_FLOATING_TYPES(vid.type(), "dnls_search_forward_kernel", ([&] {
  //   dnls_search_forward_kernel<scalar_t><<<blocks, threads>>>(
  //       vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
  //       patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
  //       queryInds.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
  //       fflow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
  //       bflow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
  //       nlInds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
  //       nlDists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
  //       int ws, int wt);
  //     }));
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t>
__global__ void dnls_search_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_patches,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> nlInds,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> nlDists) {

  //batch index
  const int n = blockIdx.y;

  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;

}

void dnls_cuda_search_backward(
    torch::Tensor grad_patches,torch::Tensor vid,
    torch::Tensor nlInds, torch::Tensor nlDists) {

  // launch params
  int numQueries = 10;//nlInds.size(0);
  const int threads = 1024;
  const dim3 blocks((numQueries - 1) / threads + 1);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "dnls_search_backward_kernel", ([&] {
    dnls_search_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        nlInds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        nlDists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));

}
