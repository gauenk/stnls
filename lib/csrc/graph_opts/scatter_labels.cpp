#include <torch/extension.h>
#include <vector>

// scattering


void scatter_labels_cuda(
    const torch::Tensor flows, const torch::Tensor flows_k,
    torch::Tensor labels, torch::Tensor names,
    int ws, int wt, int stride0, float stride1, bool full_ws);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward/Backward

void scatter_labels_forward(
    const torch::Tensor flows, const torch::Tensor flows_k,
    torch::Tensor labels, torch::Tensor names,
    int ws, int wt, int stride0, float stride1, bool full_ws){
  CHECK_INPUT(flows);
  CHECK_INPUT(flows_k);
  CHECK_INPUT(labels);
  CHECK_INPUT(names);
  scatter_labels_cuda(flows,flows_k,labels,names,
                      ws,wt,stride0,stride1,full_ws);
}

// python bindings
void init_scatter_labels(py::module &m){
  m.def("scatter_labels_forward", &scatter_labels_forward,"Scatter Labels");
}

