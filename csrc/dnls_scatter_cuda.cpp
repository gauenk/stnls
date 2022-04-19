
// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
#include "pybind.hpp"

// CUDA forward declarations

std::vector<torch::Tensor> dnls_cuda_scatter_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor nlInds);

std::vector<torch::Tensor> dnls_cuda_scatter_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    torch::Tensor nlInds);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_scatter_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor nlInds) {
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(nlInds);
  dnls_cuda_scatter_forward(vid,patches,nlInds);
}

void dnls_scatter_backward(
    torch::Tensor grad_patches,
    torch::Tensor vid,
    torch::Tensor nlInds) {
  CHECK_INPUT(grad_patches);
  CHECK_INPUT(vid);
  CHECK_INPUT(nlInds);
  dnls_cuda_scatter_backward(grad_patches,vid,nlInds);
}

// python bindings
void init_scatter(py::module &m){
  m.def("scatter_forward", &dnls_scatter_forward, "DNLS Scatter Forward (CUDA)");
  m.def("scatter_backward", &dnls_scatter_backward, "DNLS Scatter Backward (CUDA)");
}

