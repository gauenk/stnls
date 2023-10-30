/***


Anchor the self-patch displace as the first index.

This is a nice ordering for many subsequent routines.

Using Pytorch functions such as "mask_select" consumes huge GPU Mem.

We can't just compute center of "wt,ws,ws" since our search
space is not always, nor should be, centered. This is really
only true at image boundaries...

We also want to compute the center of the offset regions too.

***/


#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void anchor_self_forward_cuda(
     torch::Tensor dists, torch::Tensor inds,
     torch::Tensor order, int stride0, int H, int W);

void anchor_self_time_forward_cuda(
     torch::Tensor dists, torch::Tensor inds,
     torch::Tensor flows, int wt, int stride0, int H, int W);

void anchor_self_refine_forward_cuda(
     torch::Tensor dists, torch::Tensor inds,
     torch::Tensor flows, int stride0, int H, int W);

void anchor_self_paired_forward_cuda(
     torch::Tensor dists, torch::Tensor inds,
     torch::Tensor flows, int stride0, int H, int W);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void anchor_self_forward(
     torch::Tensor dists, torch::Tensor inds,
     torch::Tensor order, int stride0, int H, int W){
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(order);
  anchor_self_forward_cuda(dists,inds,order,stride0,H,W);
}

void anchor_self_time_forward(
     torch::Tensor dists, torch::Tensor inds,
     torch::Tensor flows, int wt, int stride0, int H, int W){
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(flows);
  anchor_self_time_forward_cuda(dists,inds,flows,wt,stride0,H,W);
}

void anchor_self_refine_forward(
     torch::Tensor dists, torch::Tensor inds,
     torch::Tensor flows, int stride0, int H, int W){
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(flows);
  anchor_self_refine_forward_cuda(dists,inds,flows,stride0,H,W);
}

void anchor_self_paired_forward(
     torch::Tensor dists, torch::Tensor inds,
     torch::Tensor flows, int stride0, int H, int W){
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(flows);
  anchor_self_paired_forward_cuda(dists,inds,flows,stride0,H,W);
}

// python bindings
void init_anchor_self(py::module &m){
  m.def("anchor_self", &anchor_self_forward,
        "anchor_self (CUDA)");
  m.def("anchor_self_time", &anchor_self_time_forward,
        "anchor_self (CUDA)");
  m.def("anchor_self_refine", &anchor_self_refine_forward,
        "anchor_self (CUDA)");
  m.def("anchor_self_paired", &anchor_self_paired_forward,
        "anchor_self (CUDA)");

}
