#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void patch_full_connected_forward_cuda(
    torch::Tensor vid, torch::Tensor vid_in,
    torch::Tensor weights, torch::Tensor bias,
    int qstart, int nqueries, int ps,
    int top, int left, int btm, int right,
    int hw_start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void pfc_forward(
    torch::Tensor vid, torch::Tensor vid_in,
    torch::Tensor weights, torch::Tensor bias,
    int qstart, int nqueries, int ps,
    int top, int left, int btm, int right,
    int hw_start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect){
  patch_full_connected_forward_cuda(
                       vid,vid_in,weights,bias,qstart,
                       nqueries,ps,top,left,btm,right,
                       hw_start,stride,dilation,adj,only_full,
                       use_reflect);
}

// python bindings
void init_pfc(py::module &m){
  m.def("pfc_forward", &pfc_forward,
        "PFC Forward (CUDA)");
}

