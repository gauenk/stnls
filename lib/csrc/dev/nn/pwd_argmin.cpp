#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void pwd_argmin_forward_cuda(
     const torch::Tensor pwd,
     torch::Tensor mins, torch::Tensor argmins);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void pwd_argmin_forward(const torch::Tensor pwd,
                        torch::Tensor mins,
                        torch::Tensor argmins){
  CHECK_INPUT(pwd);
  CHECK_INPUT(mins);
  CHECK_INPUT(argmins);
  pwd_argmin_forward_cuda(dists,mins,argmins);
}

// python bindings
void init_pwd_argmin(py::module &m){
  m.def("pwd_argmin", &pwd_argmin_forward,
        "Pairwise Distances Argmin of Lower Triangular");
}
