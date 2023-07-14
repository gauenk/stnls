// [dir of col2im/im2col]: /home/gauenk/pytorch/aten/src/ATen/native/cuda/

// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void stnls_cuda_nlfold_forward(
    torch::Tensor vid, torch::Tensor zvid,
    torch::Tensor patches,
    int stride, int dilation,
    bool use_adj, bool reflect);

void stnls_cuda_nlfold_backward(
    torch::Tensor grad_patches,
    const torch::Tensor grad_vid,
    // const torch::Tensor vid,
    // const torch::Tensor zvid,
    // const torch::Tensor patches,
    int stride, int dilation,
    bool use_adj, bool reflect);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*********************************

      Using Raster Order

*********************************/

void stnls_nlfold_forward(
    torch::Tensor vid, torch::Tensor zvid,
    torch::Tensor patches,
    int stride, int dilation,
    bool use_adj, bool reflect) {
  CHECK_INPUT(vid);
  CHECK_INPUT(zvid);
  CHECK_INPUT(patches);
  stnls_cuda_nlfold_forward(vid,zvid,patches,
                            stride,dilation,
                            use_adj,reflect);
}

void stnls_nlfold_backward(
    torch::Tensor grad_patches,
    const torch::Tensor grad_vid,
    // const torch::Tensor vid,
    // const torch::Tensor zvid,
    // const torch::Tensor patches,
    int stride, int dilation,
    bool use_adj, bool reflect) {
  CHECK_INPUT(grad_patches);
  CHECK_INPUT(grad_vid);
  // CHECK_INPUT(vid);
  // CHECK_INPUT(zvid);
  // CHECK_INPUT(patches);
  stnls_cuda_nlfold_backward(grad_patches,grad_vid,
                             // vid,zvid,patches,
                             stride,dilation,use_adj,reflect);
}



// python bindings
void init_nlfold(py::module &m){
  m.def("nlfold_forward", &stnls_nlfold_forward, "DNLS Fold Forward (CUDA)");
  m.def("nlfold_backward", &stnls_nlfold_backward, "DNLS Fold Backward (CUDA)");
}

