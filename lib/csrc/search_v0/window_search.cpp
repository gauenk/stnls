#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void window_search_forward_cuda(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor fflow, torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w, int wt, int chnls,
    int dilation, int stride1, bool use_adj,
    bool reflect_bounds, bool search_abs,
    bool full_ws, bool anchor_self,
    torch::Tensor tranges, torch::Tensor n_tranges,
    torch::Tensor min_tranges, torch::Tensor partition);

void window_search_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists, torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, bool use_adj,
    bool reflect_bounds, bool use_rand, bool exact);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void window_search_forward(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor fflow,torch::Tensor bflow,
    torch::Tensor dists,torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int dilation, int stride1,
    bool use_adj, bool reflect_bounds, bool search_abs,
    bool full_ws, bool anchor_self,
    torch::Tensor tranges, torch::Tensor n_tranges,
    torch::Tensor min_tranges, torch::Tensor partition){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(tranges);
  CHECK_INPUT(n_tranges);
  CHECK_INPUT(min_tranges);
  window_search_forward_cuda(vid0,vid1,fflow,bflow,dists,inds,
                             qstart, stride0, n_h0, n_w0,
                             h0_off,w0_off,h1_off,w1_off,
                             ps,pt,ws_h,ws_w,wt,chnls,dilation,stride1,
                             use_adj,reflect_bounds,search_abs,
                             full_ws,anchor_self,
                             tranges,n_tranges,min_tranges,partition);
}

void window_search_backward(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists, torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps,int pt, int dilation, bool use_adj, bool reflect_bounds,
    bool use_rand, bool exact) {
  CHECK_INPUT(grad_vid0);
  CHECK_INPUT(grad_vid1);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  window_search_backward_cuda(grad_vid0,grad_vid1,vid0,vid1,
                                     dists,inds,
                                     qstart,stride0,n_h0,n_w0,
                                     h0_off,w0_off,h1_off,w1_off,
                                     ps,pt,dilation,use_adj,reflect_bounds,
                                     use_rand,exact);
}

// python bindings
void init_window_search(py::module &m){
  m.def("window_search_forward", &window_search_forward,
        "DNLS Search Forward with Index (CUDA)");
  m.def("window_search_backward", &window_search_backward,
        "DNLS Search Backward with Index (CUDA)");
}

