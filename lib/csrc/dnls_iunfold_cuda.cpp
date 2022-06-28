// [dir of col2im/im2col]: /home/gauenk/pytorch/aten/src/ATen/native/cuda/

// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void dnls_cuda_iunfold_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj);

void dnls_cuda_iunfold_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_iunfold_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj) {
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  dnls_cuda_iunfold_forward(vid,patches,
                            top,left,btm,right,
                            start,stride,dilation,adj);
}

void dnls_iunfold_backward(
    torch::Tensor grad_vid, torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj) {
  CHECK_INPUT(grad_vid);
  CHECK_INPUT(patches);
  dnls_cuda_iunfold_backward(grad_vid,patches,
                             top,left,btm,right,
                             start,stride,dilation,adj);
}


// python bindings
void init_iunfold(py::module &m){
  m.def("iunfold_forward", &dnls_iunfold_forward, "DNLS iUnfold Forward (CUDA)");
  m.def("iunfold_backward", &dnls_iunfold_backward, "DNLS iUfold Backward (CUDA)");
}

