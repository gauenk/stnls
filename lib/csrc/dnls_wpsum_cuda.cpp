#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void dnls_cuda_wpsum_forward(
  torch::Tensor vid, torch::Tensor patches,
  torch::Tensor dists, torch::Tensor inds,
  int dilation, int adj, bool reflect_bounds);

void dnls_cuda_wpsum_backward(
    torch::Tensor vid_grad, torch::Tensor patches_grad,
    torch::Tensor dists, torch::Tensor inds,
    int dilation, int adj, bool reflect_bounds, bool exact);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_wpsum_forward(
  torch::Tensor vid, torch::Tensor patches,
  torch::Tensor dists, torch::Tensor inds,
  int dilation, int adj, bool reflect_bounds){
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  dnls_cuda_wpsum_forward(vid,patches,dists,inds,dilation,adj,reflect_bounds);
}

void dnls_wpsum_backward(
  torch::Tensor vid_grad, torch::Tensor patches_grad,
  torch::Tensor dists, torch::Tensor inds,
  int dilation, int adj, bool reflect_bounds, bool exact){
  CHECK_INPUT(vid_grad);
  CHECK_INPUT(patches_grad);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  dnls_cuda_wpsum_backward(vid_grad,patches_grad,dists,inds,
                           dilation,adj,reflect_bounds,exact);
}

// python bindings
void init_wpsum(py::module &m){
  m.def("wpsum_forward", &dnls_wpsum_forward,"DNLS WeightedPatchSum Forward (CUDA)");
  m.def("wpsum_backward", &dnls_wpsum_backward,"DNLS WeightedPatchSum Backward (CUDA)");
}

