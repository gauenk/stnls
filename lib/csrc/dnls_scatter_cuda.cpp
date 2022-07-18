
// imports
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

void dnls_cuda_scatter_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor nlInds,
    int dilation, int adj, bool use_bounds);

void dnls_cuda_scatter_backward(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation, bool exact,
    int adj, bool use_bounds);

void dnls_cuda_scatter_backward_eff(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation, bool exact,
    int adj, bool use_bounds);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_scatter_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor nlInds,
    int dilation, int adj, bool use_bounds) {
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(nlInds);
  dnls_cuda_scatter_forward(vid,patches,nlInds,dilation,adj,use_bounds);
}

void dnls_scatter_backward(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation,  bool exact,
    int adj, bool use_bounds) {
  CHECK_INPUT(vid);
  CHECK_INPUT(grad_patches);
  CHECK_INPUT(nlInds);
  dnls_cuda_scatter_backward(vid,grad_patches,nlInds,
                                    dilation,exact,adj,use_bounds);
}

void dnls_scatter_backward_eff(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation, float lam, bool exact,
    int adj, bool use_bounds) {
  CHECK_INPUT(vid);
  CHECK_INPUT(grad_patches);
  CHECK_INPUT(nlInds);
  dnls_cuda_scatter_backward_eff(vid,grad_patches,nlInds,
                                 dilation,exact,adj,use_bounds);
}


// python bindings
void init_scatter(py::module &m){
  m.def("scatter_forward", &dnls_scatter_forward, "DNLS Scatter Forward (CUDA)");
  m.def("scatter_backward", &dnls_scatter_backward,"DNLS Scatter Backward (CUDA)");
  m.def("scatter_backward_eff", &dnls_scatter_backward_eff, "DNLS Scatter Backward (CUDA), An Attempted Efficient Impl.");
}

