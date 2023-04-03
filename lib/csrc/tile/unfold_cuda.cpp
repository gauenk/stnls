// [dir of col2im/im2col]: /home/gauenk/pytorch/aten/src/ATen/native/cuda/

// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void stnls_cuda_unfold_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    int qStart, int qStride,
    int dilation);

void stnls_cuda_unfold_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    int qStart, int qStride,
    int dilation);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void stnls_unfold_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    int qStart,
    int qStride, int dilation) {
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  stnls_cuda_unfold_forward(vid,patches,qStart,qStride,dilation);
}

void stnls_unfold_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    int qStart, int qStride,
    int dilation) {
  CHECK_INPUT(grad_vid);
  CHECK_INPUT(patches);
  stnls_cuda_unfold_backward(grad_vid,patches,qStart,qStride,dilation);
}


// python bindings
void init_unfold(py::module &m){
  m.def("unfold_forward", &stnls_unfold_forward, "DNLS Unfold Forward (CUDA)");
  m.def("unfold_backward", &stnls_unfold_backward, "DNLS Unfold Backward (CUDA)");
}

