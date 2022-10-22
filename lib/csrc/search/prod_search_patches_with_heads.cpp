#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void prod_search_patches_with_heads_forward_cuda(
    torch::Tensor patches0, torch::Tensor patches1,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor access_inds,
    int chnls, int dilation, bool anchor_self);

void prod_search_patches_with_heads_backward_cuda(
    torch::Tensor grad_patches0, torch::Tensor grad_patches1,
    torch::Tensor patches0, torch::Tensor patches1,
    torch::Tensor dists, torch::Tensor inds, torch::Tensor access_inds,
    int qstart, int nheads, int stride0, int n_h0, int n_w0,
    int chnls, int dilation, bool anchor_self);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void prod_search_patches_with_heads_forward(
    torch::Tensor patches0, torch::Tensor patches1,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor access_inds,
    int chnls, int dilation, bool anchor_self){
  CHECK_INPUT(patches0);
  CHECK_INPUT(patches1);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(access_inds);
  prod_search_patches_with_heads_forward_cuda(patches0,patches1,dists,inds,
                                              access_inds,chnls,dilation,
                                              anchor_self);
}

void prod_search_patches_with_heads_backward(
    torch::Tensor grad_patches0, torch::Tensor grad_patches1,
    torch::Tensor patches0, torch::Tensor patches1,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor access_inds, int chnls, int dilation) {
  CHECK_INPUT(grad_patches0);
  CHECK_INPUT(grad_patches1);
  CHECK_INPUT(patches0);
  CHECK_INPUT(patches1);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(access_inds);
  prod_search_patches_with_heads_backward_cuda(grad_patches0,grad_patches1,
                                               patches0,patches1,
                                               dists,inds,access_inds,
                                               chnls,dilation);
}


// python bindings
void init_prod_search_patches_with_heads(py::module &m){
  m.def("prod_search_patches_with_heads_forward",
        &prod_search_patches_with_heads_forward,
        "Product Search Forward with Heads (CUDA)");
  m.def("prod_search_patches_with_heads_backward",
        &prod_search_patches_with_heads_backward,
        "Product Search Backward with Heads (CUDA)");
}

