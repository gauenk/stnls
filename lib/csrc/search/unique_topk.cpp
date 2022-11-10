#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void unique_topk_forward_cuda(
    torch::Tensor vals,
    torch::Tensor args,
    int k, int dim);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void unique_topk_forward(torch::Tensor vals,
                         torch::Tensor args,
                         int k, int dim){
  CHECK_INPUT(vals);
  CHECK_INPUT(args);
  unique_topk_forward_cuda(vals,args,k,dim);
}

// python bindings
void init_unique_topk(py::module &m){
  m.def("unique_topk", &unique_topk_forward,
        "Unique Top-K");
}
