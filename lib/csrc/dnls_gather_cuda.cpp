
// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void dnls_cuda_gather_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor nlInds);

void dnls_cuda_gather_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    torch::Tensor nlInds);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_gather_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor nlInds) {
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(nlInds);
  dnls_cuda_gather_forward(vid,patches,nlInds);
}

void dnls_gather_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    torch::Tensor nlInds) {
  CHECK_INPUT(grad_vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(nlInds);
  dnls_cuda_gather_backward(grad_vid,patches,nlInds);
}


// python bindings
void init_gather(py::module &m){
  m.def("gather_forward", &dnls_gather_forward, "DNLS Gather Forward (CUDA)");
  m.def("gather_backward", &dnls_gather_backward, "DNLS Gather Backward (CUDA)");
}

