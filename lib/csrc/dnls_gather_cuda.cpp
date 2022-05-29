// [dir of col2im/im2col]: /home/gauenk/pytorch/aten/src/ATen/native/cuda/

// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void dnls_cuda_gather_forward(
    torch::Tensor vid,
    torch::Tensor wvid,
    torch::Tensor patches,
    torch::Tensor nlDists,
    torch::Tensor nlInds,
    int dilation, float lam, bool exact);

void dnls_cuda_gather_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    torch::Tensor nlInds,
    int dilation);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_gather_forward(
    torch::Tensor vid,
    torch::Tensor wvid,
    torch::Tensor patches,
    torch::Tensor nlDists,
    torch::Tensor nlInds,
    int dilation, float lam, bool exact) {
  CHECK_INPUT(vid);
  CHECK_INPUT(wvid);
  CHECK_INPUT(patches);
  CHECK_INPUT(nlDists);
  CHECK_INPUT(nlInds);
  dnls_cuda_gather_forward(vid,wvid,patches,nlDists,nlInds,dilation,lam,exact);
}

void dnls_gather_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    torch::Tensor nlDists,
    torch::Tensor nlInds,
    int dilation) {
  CHECK_INPUT(grad_vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(nlDists);
  CHECK_INPUT(nlInds);
  dnls_cuda_gather_backward(grad_vid,patches,nlInds,dilation);
}


// python bindings
void init_gather(py::module &m){
  m.def("gather_forward", &dnls_gather_forward, "DNLS Gather Forward (CUDA)");
  m.def("gather_backward", &dnls_gather_backward, "DNLS Gather Backward (CUDA)");
}

