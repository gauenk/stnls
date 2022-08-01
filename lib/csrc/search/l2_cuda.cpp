#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void search_l2_forward_cuda(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor qinds, torch::Tensor fflow, torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w, int wt, int chnls,
    int dilation, int stride, bool use_adj,
    bool reflect_bounds, bool search_abs, bool full_ws,
    torch::Tensor bufs, torch::Tensor tranges,
    torch::Tensor n_tranges, torch::Tensor min_tranges);


void search_l2_backward_cuda(
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

void search_l2_forward(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor qinds,torch::Tensor fflow,torch::Tensor bflow,
    torch::Tensor dists,torch::Tensor inds,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int dilation, int stride,
    bool use_adj, bool reflect_bounds, bool search_abs, bool full_ws,
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
  search_l2_forward_cuda(vid0,vid1,qinds,fflow,bflow,dists,inds,
                           h0_off,w0_off,h1_off,w1_off,
                           ps,pt,ws_h,ws_w,wt,chnls,dilation,stride,
                           use_adj,reflect_bounds,search_abs,full_ws,
                           bufs,tranges,n_tranges,min_tranges);
}

void search_l2_backward(
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
  search_l2_backward_cuda(grad_vid0,grad_vid1,vid0,vid1,
                          dists,inds,qinds,h0_off,w0_off,h1_off,w1_off,
                          ps,pt,dilation,use_adj,reflect_bounds,exact);
}

// python bindings
void init_l2_search(py::module &m){
  m.def("search_l2_forward", &search_l2_forward, "DNLS Search Forward (CUDA)");
  m.def("search_l2_backward", &search_l2_backward, "DNLS Search Backward (CUDA)");
}

