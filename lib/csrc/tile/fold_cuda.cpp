// [dir of col2im/im2col]: /home/gauenk/pytorch/aten/src/ATen/native/cuda/

// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void dnls_cuda_fold_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    int start, int stride,
    int dilation);

void dnls_cuda_fold_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    int start, int stride,
    int dilation);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*********************************

      Using Raster Ordering

*********************************/

void dnls_fold_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    int start, int stride,
    int dilation) {
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  dnls_cuda_fold_forward(vid,patches,start,stride,dilation);
}

void dnls_fold_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    int start, int stride,
    int dilation) {
  CHECK_INPUT(grad_vid);
  CHECK_INPUT(patches);
  dnls_cuda_fold_backward(grad_vid,patches,start,stride,dilation);
}


// python bindings
void init_fold(py::module &m){
  m.def("fold_forward", &dnls_fold_forward, "DNLS Fold Forward (CUDA)");
  m.def("fold_backward", &dnls_fold_backward, "DNLS Fold Backward (CUDA)");
}

