#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void search_prod_forward_cuda(
    torch::Tensor vid0,torch::Tensor vid1,torch::Tensor queryInds,
    torch::Tensor fflow,torch::Tensor bflow,
    torch::Tensor nlDists,torch::Tensor nlInds,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int stride, int dilation,
    bool use_search_abs, bool use_bounds, bool use_adj,
    int oh0, int ow0, int oh1, int ow1,
    torch::Tensor bufs,torch::Tensor tranges,
    torch::Tensor n_tranges,torch::Tensor min_tranges);


void search_prod_backward_cuda(
    torch::Tensor vid0_grad, torch::Tensor vid1_grad,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor qinds, torch::Tensor nlDists, torch::Tensor nlInds,
    int oh0, int ow0, int oh1, int ow1,
    int ps, int pt, float lam, bool use_bounds, bool exact);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void search_prod_forward(
    torch::Tensor vid0,torch::Tensor vid1,torch::Tensor queryInds,
    torch::Tensor fflow,torch::Tensor bflow,
    torch::Tensor nlDists,torch::Tensor nlInds,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int stride, int dilation,
    bool use_search_abs, bool use_bounds, bool use_adj,
    int oh0, int ow0, int oh1, int ow1,
    torch::Tensor bufs,torch::Tensor tranges,
    torch::Tensor n_tranges,torch::Tensor min_tranges){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(queryInds);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(nlDists);
  CHECK_INPUT(nlInds);
  CHECK_INPUT(bufs);
  CHECK_INPUT(tranges);
  CHECK_INPUT(n_tranges);
  CHECK_INPUT(min_tranges);
  search_prod_forward_cuda(vid0,vid1,queryInds,fflow,bflow,nlDists,nlInds,
                            ps,pt,ws_h,ws_w,wt,chnls,stride,dilation,
                            use_search_abs, use_bounds, use_adj,
                            oh0, ow0, oh1, ow1,
                            bufs,tranges,n_tranges,min_tranges);
}

void search_prod_backward(
    torch::Tensor vid0_grad,
    torch::Tensor vid1_grad,
    torch::Tensor vid0,
    torch::Tensor vid1,
    torch::Tensor qinds,
    torch::Tensor nlDists,
    torch::Tensor nlInds,
    int oh0, int ow0, int oh1, int ow1,
    int ps,int pt,float lam, bool use_bounds, bool exact) {
  CHECK_INPUT(vid0_grad);
  CHECK_INPUT(vid1_grad);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(qinds);
  CHECK_INPUT(nlDists);
  CHECK_INPUT(nlInds);
  search_prod_backward_cuda(vid0_grad,vid1_grad,vid0,vid1,qinds,
                             nlDists,nlInds,oh0,ow0,oh1,ow1,
                             ps,pt,lam,use_bounds,exact);
}

// python bindings
void init_prod_search(py::module &m){
  m.def("search_prod_forward", &search_prod_forward, "DNLS Search (Prod) Forward (CUDA)");
  m.def("search_prod_backward", &search_prod_backward, "DNLS Search (Prod) Backward (CUDA)");
}

