
// imports
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

void dnls_cuda_unfoldk_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor nlInds,
    int dilation, int adj, bool use_bounds);

void dnls_cuda_unfoldk_backward(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation, bool exact,
    int adj, bool use_bounds);

void dnls_cuda_unfoldk_backward_eff(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation, bool exact,
    int adj, bool use_bounds);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_unfoldk_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor nlInds,
    int dilation, int adj, bool use_bounds) {
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(nlInds);
  dnls_cuda_unfoldk_forward(vid,patches,nlInds,dilation,adj,use_bounds);
}

void dnls_unfoldk_backward(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation,  bool exact,
    int adj, bool use_bounds) {
  CHECK_INPUT(vid);
  CHECK_INPUT(grad_patches);
  CHECK_INPUT(nlInds);
  dnls_cuda_unfoldk_backward(vid,grad_patches,nlInds,
                                    dilation,exact,adj,use_bounds);
}

void dnls_unfoldk_backward_eff(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation, float lam, bool exact,
    int adj, bool use_bounds) {
  CHECK_INPUT(vid);
  CHECK_INPUT(grad_patches);
  CHECK_INPUT(nlInds);
  dnls_cuda_unfoldk_backward_eff(vid,grad_patches,nlInds,
                                 dilation,exact,adj,use_bounds);
}


// python bindings
void init_unfoldk(py::module &m){
  m.def("unfoldk_forward", &dnls_unfoldk_forward, "DNLS Unfoldk Forward (CUDA)");
  m.def("unfoldk_backward", &dnls_unfoldk_backward,"DNLS Unfoldk Backward (CUDA)");
  m.def("unfoldk_backward_eff", &dnls_unfoldk_backward_eff, "DNLS Unfoldk Backward (CUDA), An Attempted Efficient Impl.");
}

