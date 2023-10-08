/*******************

  Find an interpolated maximum/minimum distance
  using bicubic interpolaton

 *******************/



#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void nls_bicubic_interp_forward_cuda(
     torch::Tensor dists, torch::Tensor inds);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void nls_bicubic_interp_forward(
     torch::Tensor dists, torch::Tensor inds){
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  nls_bicubic_interp_forward_cuda(dists,inds,qstart,stride0,H,W);
}

// python bindings
void init_nls_bicubic_interp(py::module &m){
  m.def("nls_bicubic_interp", &nls_bicubic_interp,
        "nls_bicubic_interp (CUDA)");
}

