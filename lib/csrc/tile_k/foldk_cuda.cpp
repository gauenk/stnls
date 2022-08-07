// [dir of col2im/im2col]: /home/gauenk/pytorch/aten/src/ATen/native/cuda/

// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations


void dnls_cuda_foldk_forward(
    torch::Tensor vid,
    torch::Tensor wvid,
    torch::Tensor patches,
    torch::Tensor dists,
    torch::Tensor inds,
    int ws, int wt, int dilation);

void dnls_cuda_foldk_forward_race(
    torch::Tensor vid,
    torch::Tensor wvid,
    torch::Tensor patches,
    torch::Tensor dists,
    torch::Tensor inds,
    int dilation, bool use_rand, bool exact);

void dnls_cuda_foldk_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    torch::Tensor inds,
    int dilation);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_foldk_forward(
    torch::Tensor vid,
    torch::Tensor wvid,
    torch::Tensor patches,
    torch::Tensor dists,
    torch::Tensor inds,
    int ws, int wt, int dilation) {
  CHECK_INPUT(vid);
  CHECK_INPUT(wvid);
  CHECK_INPUT(patches);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  dnls_cuda_foldk_forward(vid,wvid,patches,dists,inds,ws,wt,dilation);
}


void dnls_foldk_forward_race(
    torch::Tensor vid,
    torch::Tensor wvid,
    torch::Tensor patches,
    torch::Tensor dists,
    torch::Tensor inds,
    int dilation, bool use_rand, bool exact) {
  CHECK_INPUT(vid);
  CHECK_INPUT(wvid);
  CHECK_INPUT(patches);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  dnls_cuda_foldk_forward_race(vid,wvid,patches,dists,inds,
                                dilation,use_rand,exact);
}

void dnls_foldk_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    torch::Tensor dists,
    torch::Tensor inds,
    int dilation) {
  CHECK_INPUT(grad_vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  dnls_cuda_foldk_backward(grad_vid,patches,inds,dilation);
}


// python bindings
void init_foldk(py::module &m){
  m.def("foldk_forward", &dnls_foldk_forward, "DNLS Foldk Forward (CUDA)");
  m.def("foldk_forward_race", &dnls_foldk_forward_race,
        "DNLS Foldk Forward with Race Condition (CUDA)");
  m.def("foldk_backward", &dnls_foldk_backward, "DNLS Foldk Backward (CUDA)");
}

