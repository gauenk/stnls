#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> dnls_cuda_search_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor queryInds,
    torch::Tensor fflow,
    torch::Tensor bflow,
    torch::Tensor nlInds,
    torch::Tensor nlDists,
    int ws, int wt);

std::vector<torch::Tensor> dnls_cuda_search_backward(
    torch::Tensor grad_patches,
    torch::Tensor vid,
    torch::Tensor nlInds,
    torch::Tensor nlDists);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_search_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    torch::Tensor queryInds,
    torch::Tensor fflow,
    torch::Tensor bflow,
    torch::Tensor nlInds,
    torch::Tensor nlDists,
    int ws, int wt){
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(queryInds);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(nlInds);
  CHECK_INPUT(nlDists);
  dnls_cuda_search_forward(vid,patches,queryInds,fflow,bflow,
                                  nlInds,nlDists,ws,wt);
}

void dnls_search_backward(
    torch::Tensor grad_patches,
    torch::Tensor vid,
    torch::Tensor nlInds,
    torch::Tensor nlDists) {
  CHECK_INPUT(grad_patches);
  CHECK_INPUT(vid);
  CHECK_INPUT(nlInds);
  CHECK_INPUT(nlDists);
  dnls_cuda_search_backward(grad_patches,vid,nlInds,nlDists);
}

// python bindings
void init_search(py::module &m){
  m.def("search_forward", &dnls_search_forward, "DNLS Search Forward (CUDA)");
  m.def("search_backward", &dnls_search_backward, "DNLS Search Backward (CUDA)");
}

