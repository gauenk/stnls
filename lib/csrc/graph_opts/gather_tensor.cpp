
#include <torch/extension.h>
#include <vector>

// gathering

void gather_tensor_forward_cuda(torch::Tensor out_tensor,
                                 const torch::Tensor in_tensor,
                                 const torch::Tensor labels,
                                 const torch::Tensor flows_k,
                                 int stride0, int stride1, int H, int W);

void gather_tensor_backward_cuda(torch::Tensor in_tensor_grad,
                                  const torch::Tensor out_tensor_grad,
                                  const torch::Tensor labels,
                                  const torch::Tensor flows_k, int stride0);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward/Backward

void gather_tensor_forward(
    torch::Tensor out_tensor,
    const torch::Tensor in_tensor,
    const torch::Tensor labels,
    const torch::Tensor flows_k,
    int stride0, int stride1, int H, int W){
  CHECK_INPUT(out_tensor);
  CHECK_INPUT(in_tensor);
  CHECK_INPUT(labels);
  CHECK_INPUT(flows_k);
  gather_tensor_forward_cuda(out_tensor,in_tensor,labels,flows_k,
                              stride0,stride1,H,W);
}

void gather_tensor_backward(
    torch::Tensor out_tensor_grad,
    const torch::Tensor in_tensor_grad,
    const torch::Tensor labels,
    const torch::Tensor flows_k, int stride0){
  CHECK_INPUT(in_tensor_grad);
  CHECK_INPUT(out_tensor_grad);
  CHECK_INPUT(labels);
  CHECK_INPUT(flows_k);
  gather_tensor_backward_cuda(in_tensor_grad,out_tensor_grad,labels,flows_k,stride0);
}

// python bindings
void init_gather_tensor(py::module &m){
  m.def("gather_tensor_forward", &gather_tensor_forward,"Gather Labels");
  m.def("gather_tensor_backward", &gather_tensor_backward,"Gather Labels");
}

