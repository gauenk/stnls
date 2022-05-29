#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void dnls_cuda_search_forward(
    torch::Tensor vid,torch::Tensor queryInds,
    torch::Tensor fflow,torch::Tensor bflow,
    torch::Tensor nlDists,torch::Tensor nlInds,
    int ps, int pt, int ws, int wt,
    int chnls, int dilation, int stride,
    torch::Tensor bufs,torch::Tensor tranges,
    torch::Tensor n_tranges,torch::Tensor min_tranges);


void dnls_cuda_search_backward(
    torch::Tensor vid, torch::Tensor nlDists, torch::Tensor nlInds,
    int ps, int pt, float lam);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void dnls_search_forward(
    torch::Tensor vid,torch::Tensor queryInds,
    torch::Tensor fflow,torch::Tensor bflow,
    torch::Tensor nlDists,torch::Tensor nlInds,
    int ps, int pt, int ws, int wt,
    int chnls, int dilation, int stride,
    torch::Tensor bufs,torch::Tensor tranges,
    torch::Tensor n_tranges,torch::Tensor min_tranges){
  CHECK_INPUT(vid);
  CHECK_INPUT(queryInds);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(nlDists);
  CHECK_INPUT(nlInds);
  CHECK_INPUT(bufs);
  CHECK_INPUT(tranges);
  CHECK_INPUT(n_tranges);
  CHECK_INPUT(min_tranges);
  dnls_cuda_search_forward(vid,queryInds,fflow,bflow,nlDists,nlInds,
                           ps,pt,ws,wt,chnls,dilation,stride,
                           bufs,tranges,n_tranges,min_tranges);
}

void dnls_search_backward(
    torch::Tensor vid,
    torch::Tensor nlDists,
    torch::Tensor nlInds,
    int ps,int pt,float lam) {
  CHECK_INPUT(vid);
  CHECK_INPUT(nlDists);
  CHECK_INPUT(nlInds);
  dnls_cuda_search_backward(vid,nlDists,nlInds,ps,pt,lam);
}

// python bindings
void init_search(py::module &m){
  m.def("search_forward", &dnls_search_forward, "DNLS Search Forward (CUDA)");
  m.def("search_backward", &dnls_search_backward, "DNLS Search Backward (CUDA)");
}

