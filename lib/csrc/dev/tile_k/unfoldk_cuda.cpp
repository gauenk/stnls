
// imports
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

void stnls_cuda_unfoldk_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor nlInds,
    int dilation, int adj, bool use_bounds);

void stnls_cuda_unfoldk_backward(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation, bool exact,
    int adj, bool use_bounds, bool use_atomic);

void stnls_cuda_unfoldk_backward_eff(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation, bool exact,
    int adj, bool use_bounds);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void stnls_unfoldk_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor nlInds,
    int dilation, int adj, bool use_bounds) {
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(nlInds);
  stnls_cuda_unfoldk_forward(vid,patches,nlInds,dilation,adj,use_bounds);
}

void stnls_unfoldk_backward(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation,  bool exact,
    int adj, bool use_bounds, bool use_atomic) {
  CHECK_INPUT(vid);
  CHECK_INPUT(grad_patches);
  CHECK_INPUT(nlInds);
  stnls_cuda_unfoldk_backward(vid,grad_patches,nlInds,
			      dilation,exact,adj,
			      use_bounds,use_atomic);
}

void stnls_unfoldk_backward_eff(
    torch::Tensor vid,
    torch::Tensor grad_patches,
    torch::Tensor nlInds,
    int dilation, float lam, bool exact,
    int adj, bool use_bounds) {
  CHECK_INPUT(vid);
  CHECK_INPUT(grad_patches);
  CHECK_INPUT(nlInds);
  stnls_cuda_unfoldk_backward_eff(vid,grad_patches,nlInds,
                                 dilation,exact,adj,use_bounds);
}


// python bindings
void init_unfoldk(py::module &m){
  m.def("unfoldk_forward", &stnls_unfoldk_forward, "DNLS Unfoldk Forward (CUDA)");
  m.def("unfoldk_backward", &stnls_unfoldk_backward,"DNLS Unfoldk Backward (CUDA)");
  m.def("unfoldk_backward_eff", &stnls_unfoldk_backward_eff, "DNLS Unfoldk Backward (CUDA), An Attempted Efficient Impl.");
}

