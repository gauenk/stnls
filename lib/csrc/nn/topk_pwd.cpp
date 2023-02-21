#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void topk_pwd_forward_cuda(const torch::Tensor vid,
    const torch::Tensor inds0, const torch::Tensor inds1,
    torch::Tensor dists, int ps, int pt,
    int dilation, bool reflect_bounds, bool use_adj,
    int off_H0, int off_W0, int off_H1, int off_W1);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void topk_pwd_forward(const torch::Tensor vid,
                      const torch::Tensor inds0, const torch::Tensor inds1,
                      torch::Tensor dists, int ps, int pt,
                      int dilation, bool reflect_bounds, bool use_adj,
                      int off_H0, int off_W0, int off_H1, int off_W1){
  CHECK_INPUT(vid);
  CHECK_INPUT(inds0);
  CHECK_INPUT(inds1);
  CHECK_INPUT(dists);
  topk_pwd_forward_cuda(vid,inds0,inds1,dists,ps,pt,dilation,reflect_bounds,
                        use_adj,off_H0,off_W0,off_H1,off_W1);
}

// python bindings
void init_topk_pwd(py::module &m){
  m.def("topk_pwd", &topk_pwd_forward,
        "Top-K Pairwise Distances");
}
