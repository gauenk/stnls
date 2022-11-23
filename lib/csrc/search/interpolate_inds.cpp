#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void interpolate_inds_forward_cuda(
    torch::Tensor inds, torch::Tensor inds_full,
    int scale);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void interpolate_inds_forward(
    torch::Tensor inds,
    torch::Tensor inds_full,
    int scale){
  CHECK_INPUT(inds);
  interpolate_inds_forward_cuda(inds,inds_full,scale);
}

// python bindings
void init_interpolate_inds(py::module &m){
  m.def("interpolate_inds", &interpolate_inds_forward,
        "Interpolate Indices Forward (CUDA)");
}
