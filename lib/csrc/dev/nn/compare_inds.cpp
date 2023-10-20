#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void compare_inds_forward_cuda(
     torch::Tensor dists, torch::Tensor vid,
     torch::Tensor inds0, torch::Tensor inds1,
     int qstart, int stride0);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void compare_inds_forward(
     torch::Tensor dists,
     torch::Tensor vid,
     torch::Tensor inds0,
     torch::Tensor inds1,
     int qstart, int stride0){
  CHECK_INPUT(dists);
  CHECK_INPUT(vid);
  CHECK_INPUT(inds0);
  CHECK_INPUT(inds1);
  compare_inds_forward_cuda(dists,vid,inds0,inds1,qstart,stride0);
}

// python bindings
void init_compare_inds(py::module &m){
  m.def("compare_inds", &compare_inds_forward,
        "Compare the patch similairy between two sets of indices");
}
