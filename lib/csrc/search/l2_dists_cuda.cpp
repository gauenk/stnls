#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void l2_dists_forward_cuda(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists, torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, int chnls,
    bool use_adj, bool reflect_bounds, bool anchor_self);

void l2_dists_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, int chnls,
    bool use_adj, bool reflect_bounds, bool use_rand,
    bool exact);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void l2_dists_forward(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists,torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, int chnls,
    bool use_adj, bool reflect_bounds, bool anchor_self){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  l2_dists_forward_cuda(vid0,vid1,dists,inds,
                        qstart,stride0,n_h0,n_w0,
                        h0_off,w0_off,h1_off,w1_off,
                        ps,pt,dilation,chnls,
                        use_adj,reflect_bounds,anchor_self);
}


void l2_dists_backward(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int dilation, int chnls,
    bool use_adj, bool reflect_bounds, bool use_rand,
    bool exact){
  CHECK_INPUT(grad_vid0);
  CHECK_INPUT(grad_vid1);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(grad_dists);
  CHECK_INPUT(inds);
  l2_dists_backward_cuda(grad_vid0,grad_vid1,vid0,vid1,
                         grad_dists,inds,
                         qstart,stride0,n_h0,n_w0,
                         h0_off,w0_off,h1_off,w1_off,
                         ps,pt,dilation,chnls,
                         use_adj,reflect_bounds,use_rand,exact);
}

// python bindings
void init_l2_dists(py::module &m){
  m.def("l2_dists_forward", &l2_dists_forward, "DNLS Search Forward (CUDA)");
  m.def("l2_dists_backward",&l2_dists_backward, "DNLS Search Backward (CUDA)");
}

