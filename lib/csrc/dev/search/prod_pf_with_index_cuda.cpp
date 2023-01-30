// -- included for pytorch --
#include <torch/extension.h>
#include <vector>

// -- include cuda_runtime for jax --
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <pybind11/pybind11.h>
using namespace torch::indexing;


// CUDA forward declarations

void search_prod_pf_with_index_forward_cuda(
    const torch::Tensor vid0,const torch::Tensor vid1,
    const torch::Tensor fflow,const torch::Tensor bflow,
    torch::Tensor dists,torch::Tensor inds,
    torch::Tensor self_dists,
    int qstart, int stride0, int n_h0, int n_w0,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int stride, int dilation,
    bool use_search_abs, bool use_bounds, bool use_adj,
    bool full_ws, bool anchor_self, bool use_self,
    int oh0, int ow0, int oh1, int ow1,
    const torch::Tensor tranges,
    const torch::Tensor n_tranges);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void search_prod_pf_with_index_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists,torch::Tensor inds, torch::Tensor self_dists,
    int qstart, int stride0, int n_h0, int n_w0,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int stride, int dilation,
    bool use_search_abs, bool use_bounds, bool use_adj,
    bool full_ws, bool anchor_self, bool use_self,
    int oh0, int ow0, int oh1, int ow1,
    const torch::Tensor tranges,
    const torch::Tensor n_tranges){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(self_dists);
  CHECK_INPUT(tranges);
  CHECK_INPUT(n_tranges);
  search_prod_pf_with_index_forward_cuda(
          vid0,vid1,fflow,bflow,dists,inds,
          self_dists,qstart, stride0, n_h0, n_w0,
          ps,pt,ws_h,ws_w,wt,chnls,stride,dilation,
          use_search_abs, use_bounds, use_adj,
          full_ws, anchor_self, use_self, oh0, ow0, oh1, ow1,
          tranges, n_tranges);
}



// python bindings
void init_prod_pf_with_index_search(py::module &m){
  m.def("search_prod_pf_with_index_forward", &search_prod_pf_with_index_forward,
        "DNLS Search (Prod) Forward with Precomputed Flow Offsets (CUDA)");
}

