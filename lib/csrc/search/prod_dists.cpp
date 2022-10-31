#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void prod_dists_forward_cuda(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists, torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int chnls, int dilation, int stride1,
    bool use_adj, bool reflect_bounds, bool search_abs,
    bool anchor_self);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void prod_dists_forward(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists,torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int chnls, int dilation, int stride1,
    bool use_adj, bool reflect_bounds, bool search_abs,
    bool anchor_self){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  prod_dists_forward_cuda(vid0,vid1,dists,inds,
                          qstart, stride0, n_h0, n_w0,
                          h0_off,w0_off,h1_off,w1_off,
                          ps,pt,chnls,dilation,stride1,
                          use_adj,reflect_bounds,search_abs,
                          anchor_self);
}

// python bindings
void init_prod_dists(py::module &m){
  m.def("prod_dists_forward", &prod_dists_forward,
        "Product Dists Forward with Heads (CUDA)");
}

