#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void dnls_cuda_search_forward(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor qinds, torch::Tensor fflow, torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w, int wt, int chnls,
    int dilation, int stride, bool use_adj,
    bool reflect_bounds, bool search_abs,
    torch::Tensor bufs, torch::Tensor tranges,
    torch::Tensor n_tranges, torch::Tensor min_tranges);


void dnls_cuda_search_backward(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists, torch::Tensor inds, torch::Tensor qinds,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, bool use_adj,
    bool reflect_bounds, bool exact);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_search_forward(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor qinds,torch::Tensor fflow,torch::Tensor bflow,
    torch::Tensor dists,torch::Tensor inds,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int dilation, int stride,
    bool use_adj, bool reflect_bounds, bool search_abs,
    torch::Tensor bufs,torch::Tensor tranges,
    torch::Tensor n_tranges,torch::Tensor min_tranges){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(qinds);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(bufs);
  CHECK_INPUT(tranges);
  CHECK_INPUT(n_tranges);
  CHECK_INPUT(min_tranges);
  dnls_cuda_search_forward(vid0,vid1,qinds,fflow,bflow,dists,inds,
                           h0_off,w0_off,h1_off,w1_off,
                           ps,pt,ws_h,ws_w,wt,chnls,dilation,stride,
                           use_adj,reflect_bounds,search_abs,bufs,tranges,
                           n_tranges,min_tranges);
}

void dnls_search_backward(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists, torch::Tensor inds, torch::Tensor qinds,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps,int pt, int dilation, bool use_adj, bool reflect_bounds, bool exact) {
  CHECK_INPUT(grad_vid0);
  CHECK_INPUT(grad_vid1);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(qinds);
  dnls_cuda_search_backward(grad_vid0,grad_vid1,vid0,vid1,
                            dists,inds,qinds,h0_off,w0_off,h1_off,w1_off,
                            ps,pt,dilation,use_adj,reflect_bounds,exact);
}

// python bindings
void init_search(py::module &m){
  m.def("search_forward", &dnls_search_forward, "DNLS Search Forward (CUDA)");
  m.def("search_backward", &dnls_search_backward, "DNLS Search Backward (CUDA)");
}

