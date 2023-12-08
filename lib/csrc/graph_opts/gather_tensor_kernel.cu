/*

  Stack non-local patches into a video

*/

#include <cuda/std/type_traits>
#include <cstdio>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <math.h>
#include <ATen/ATen.h>
#include "../shared_kernel.cu"

using namespace at;


/**************************************

             Forward

**************************************/

template <typename scalar_t>
__global__ void gather_tensor_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> out_tensor,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> in_tensor,
    const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> labels,
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> flows_k,
    int stride0, int stride1, int H, int W){

    // -- unpack --
    int B = flows_k.size(0);
    int HD = flows_k.size(1);
    int T = flows_k.size(2);
    int nH = flows_k.size(3);
    int nW = flows_k.size(4);
    int K = labels.size(3);
    int M = out_tensor.size(4);

    // -- derived --
    int nHW = nH*nW;
    int Q = T*nHW;
    // int H = nH*stride0;
    // int W = nW*stride0;
    int nH1 = (H-1)/stride1+1;
    int nW1 = (W-1)/stride1+1;

    // -- indexing variables --
    int ref_patch[3];
    int nl_patch[3];
    bool valid_patch;

    // -- location to fill --
    int qi = blockIdx.x*blockDim.x+threadIdx.x;
    int ki = blockIdx.y*blockDim.y+threadIdx.y;
    int ihead = blockIdx.z/B;
    int ibatch = (blockIdx.z-ihead*B) % B;
  
    // -- each region --
    if ((qi < Q) && (ki < K)){
  
      //----------------------------------
      //   Reference & Non-Local Pixel
      //----------------------------------
  
      // -- reference index --
      get_pixel_loc(ref_patch,qi,stride0,nW,nHW,H,W);
      int ti = ref_patch[0];
      int h_ref = ref_patch[1];
      int w_ref = ref_patch[2];
      int hi = ref_patch[1]/stride0;
      int wi = ref_patch[2]/stride0;

      // -- non-local index --
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl_patch[_idx] = ref_patch[_idx] + flows_k[ibatch][ihead][ti][hi][wi][ki][_idx];
      }
      int nl_hi = nl_patch[1]/stride1;
      int nl_wi = nl_patch[2]/stride1;
      check_bounds(valid_patch,nl_patch,T,H,W);
      if (not(valid_patch)){ return; }

      int nl_qi = nl_patch[0] * nH1 * nW1 + nl_hi * nW1 + nl_wi;
      int nl_si = labels[ibatch][ihead][qi][ki];
      for (int mi=0; mi < M; mi++){
       //out_tensor[ibatch][ihead][nl_qi][nl_si][mi]=in_tensor[ibatch][ihead][qi][ki][mi];
        out_tensor[ibatch][ihead][qi][ki][mi]=in_tensor[ibatch][ihead][nl_qi][nl_si][mi];
      }
    }
}

void gather_tensor_forward_cuda(
    torch::Tensor out_tensor,
    const torch::Tensor in_tensor,
    const torch::Tensor labels,
    const torch::Tensor flows_k,
    int stride0, int stride1, int H, int W){

  // -- sizes --
  int B = labels.size(0);
  int HD = labels.size(1);
  int Q = labels.size(2);
  int K = labels.size(3);

  // -- launch parameters --
  dim3 threadsPerBlock(156,4);
  dim3 blocksPerGrid(1, 1, HD*B);
  blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(K)/double(threadsPerBlock.y));

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(in_tensor.type(),
                             "gather_tensor_forward_kernel",([&]{
  gather_tensor_forward_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
           out_tensor.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
           in_tensor.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
           labels.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
           flows_k.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
           stride0,stride1,H,W);
      }));


}



/**************************************

             Backward

**************************************/

template <typename scalar_t>
__global__ void gather_tensor_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> in_grad,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> out_grad,
    const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> labels,
    const torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> flows_k,
    int stride0){

    // -- unpack --
    int B = flows_k.size(0);
    int HD = flows_k.size(1);
    int T = flows_k.size(2);
    int nH = flows_k.size(3);
    int nW = flows_k.size(4);
    int K = labels.size(3);
    int M = in_grad.size(4);

    // -- derived --
    int nHW = nH*nW;
    int Q = T*nHW;
    int H = nH*stride0;
    int W = nW*stride0;

    // -- indexing variables --
    int ref_patch[3];
    int nl_patch[3];
    bool valid_patch;

    // -- location to fill --
    int qi = blockIdx.x*blockDim.x+threadIdx.x;
    int ki = blockIdx.y*blockDim.y+threadIdx.y;
    int ihead = blockIdx.z/B;
    int ibatch = (blockIdx.z-ihead*B) % B;
  
    // -- each region --
    if ((qi < Q) && (ki < K)){
  
      //----------------------------------
      //   Reference & Non-Local Pixel
      //----------------------------------
  
      // -- reference index --
      get_pixel_loc(ref_patch,qi,stride0,nW,nHW,H,W);
      int ti = ref_patch[0];
      int h_ref = ref_patch[1];
      int w_ref = ref_patch[2];
      int hi = ref_patch[1]/stride0;
      int wi = ref_patch[2]/stride0;

      // -- non-local index --
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl_patch[_idx] = ref_patch[_idx] + flows_k[ibatch][ihead][ti][hi][wi][ki][_idx];
      }
      check_bounds(valid_patch,nl_patch,T,H,W);
      if (not(valid_patch)){ return; }

      int nl_qi = nl_patch[0] * nH * nW + nl_patch[1]/stride0 + nW + nl_patch[2];
      int nl_si = labels[ibatch][ihead][qi][ki];
      for (int mi=0; mi < M; mi++){
        in_grad[ibatch][ihead][qi][ki][mi] = out_grad[ibatch][ihead][nl_qi][nl_si][mi];
      }
    }
}

void gather_tensor_backward_cuda(
    torch::Tensor in_tensor_grad,
    const torch::Tensor out_tensor_grad,
    const torch::Tensor labels,
    const torch::Tensor flows_k, int stride0){

  // -- sizes --
  int B = labels.size(0);
  int HD = labels.size(1);
  int Q = labels.size(2);
  int K = labels.size(3);

  // -- launch parameters --
  dim3 threadsPerBlock(156,4);
  dim3 blocksPerGrid(1, 1, HD*B);
  blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(K)/double(threadsPerBlock.y));

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(in_tensor_grad.type(),
                             "gather_tensor_backward_kernel",([&]{
  gather_tensor_backward_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
           in_tensor_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
           out_tensor_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
           labels.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
           flows_k.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
           stride0);
                               }));
}


