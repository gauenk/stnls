// -- included for pytorch --
#include <torch/extension.h>
#include <vector>

// -- include cuda_runtime for jax --
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <pybind11/pybind11.h>
#include "../jax_pybind.h"
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cstddef>
// #include <cstdint>


// CUDA forward declarations

void search_prod_with_index_forward_cuda(
    const torch::Tensor vid0,const torch::Tensor vid1,
    const torch::Tensor fflow,const torch::Tensor bflow,
    torch::Tensor nlDists,torch::Tensor nlInds,
    int qstart, int stride0, int n_h0, int n_w0,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int stride, int dilation,
    bool use_search_abs, bool use_bounds, bool use_adj,
    bool full_ws, int oh0, int ow0, int oh1, int ow1,
    const torch::Tensor tranges,
    const torch::Tensor n_tranges,
    const torch::Tensor min_tranges);

void search_prod_with_index_backward_cuda(
    torch::Tensor vid0_grad, torch::Tensor vid1_grad,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor nlDists, torch::Tensor nlInds,
    int qstart, int stride0, int n_h0, int n_w0,
    int ps, int pt, float lam, bool use_adj, bool use_bounds,
    int oh0, int ow0, int oh1, int ow1, bool full_ws,
    bool use_rand, bool exact);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void search_prod_with_index_forward(
    const torch::Tensor vid0,const torch::Tensor vid1,
    const torch::Tensor fflow,const torch::Tensor bflow,
    torch::Tensor nlDists,torch::Tensor nlInds,
    int qstart, int stride0, int n_h0, int n_w0,
    int ps, int pt, int ws_h, int ws_w, int wt,
    int chnls, int stride, int dilation,
    bool use_search_abs, bool use_bounds, bool use_adj,
    bool full_ws, int oh0, int ow0, int oh1, int ow1,
    const torch::Tensor tranges,
    const torch::Tensor n_tranges,
    const torch::Tensor min_tranges){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(nlDists);
  CHECK_INPUT(nlInds);
  CHECK_INPUT(tranges);
  CHECK_INPUT(n_tranges);
  CHECK_INPUT(min_tranges);
  search_prod_with_index_forward_cuda(
          vid0,vid1,fflow,bflow,nlDists,nlInds,
          qstart, stride0, n_h0, n_w0,
          ps,pt,ws_h,ws_w,wt,chnls,stride,dilation,
          use_search_abs, use_bounds, use_adj,
          full_ws, oh0, ow0, oh1, ow1,
          tranges, n_tranges, min_tranges);
}

void search_prod_with_index_backward(
    torch::Tensor vid0_grad, torch::Tensor vid1_grad,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor nlDists, torch::Tensor nlInds,
    int qstart, int stride0, int n_h0, int n_w0,
    int ps,int pt,float lam, bool use_adj, bool use_bounds,
    int oh0, int ow0, int oh1, int ow1, bool full_ws,
    bool use_rand,bool exact) {
  CHECK_INPUT(vid0_grad);
  CHECK_INPUT(vid1_grad);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(nlDists);
  CHECK_INPUT(nlInds);
  search_prod_with_index_backward_cuda(
      vid0_grad,vid1_grad,vid0,vid1,nlDists,nlInds,
      qstart, stride0, n_h0, n_w0,
      ps, pt, lam, use_adj, use_bounds,
      oh0, ow0, oh1, ow1, full_ws, use_rand, exact);
}

// jax wrappers
void jax_search_prod_with_index_forward(cudaStream_t stream, void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len){
  // -- declr consts for now --
  int qstart = 0;
  int stride0 = 1;
  int n_h0 = 1;
  int n_w0 = 1;
  int ps = 1;
  int pt = 1;
  int chnls = 1;
  int stride = 1;
  int dilation = 1;
  bool use_search_abs = false;
  bool use_bounds = true;
  bool use_adj = false;
  bool full_ws = false;
  int oh0 = 0;
  int ow0 = 0;
  int oh1 = 0;
  int ow1 = 0;

  // -- video shape --
  int nframes = 3;
  int color = 3;
  int height = 128;
  int width = 128;
  int nqueries = 10;
  int k = 3;
  int ws_h = 8;
  int ws_w = 8;
  int wt = 0;
  int st = 2*wt+1;

  // -- init memory types --
  auto vid0_ptr = reinterpret_cast<float*>(buffers[0]);
  auto vid1_ptr = reinterpret_cast<float*>(buffers[1]);
  auto fflow_ptr = reinterpret_cast<float*>(buffers[2]);
  auto bflow_ptr = reinterpret_cast<float*>(buffers[3]);
  auto dists_ptr = reinterpret_cast<float*>(buffers[4]);
  auto inds_ptr = reinterpret_cast<int32_t*>(buffers[5]);
  auto tranges_ptr = reinterpret_cast<int32_t*>(buffers[6]);
  auto n_tranges_ptr = reinterpret_cast<int32_t*>(buffers[7]);
  auto min_tranges_ptr = reinterpret_cast<int32_t*>(buffers[8]);

  // -- init options --
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).\
    layout(torch::kStrided).device(torch::kCUDA, 0);
  auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).\
    layout(torch::kStrided).device(torch::kCUDA, 0);

  // -- create writable tensors --
  auto dists = torch::zeros({nqueries,ws_h,ws_w,st},options_f32);
  auto inds = torch::zeros({nqueries,ws_h,ws_w,st,3},options_i32);

  // -- create tensors --
  auto vid0 = torch::from_blob(vid0_ptr,{nframes,color,height,height},options_f32);
  auto vid1 = torch::from_blob(vid1_ptr,{nframes,color,height,height},options_f32);
  auto fflow = torch::from_blob(fflow_ptr,{nframes,2,height,height},options_f32);
  auto bflow = torch::from_blob(bflow_ptr,{nframes,2,height,height},options_f32);
  auto tranges = torch::from_blob(tranges_ptr,{nframes,nframes},options_i32);
  auto n_tranges = torch::from_blob(n_tranges_ptr,{nframes},options_i32);
  auto min_tranges = torch::from_blob(min_tranges_ptr,{nframes},options_i32);

  // -- run program --
  search_prod_with_index_forward(
      vid0,vid1,fflow,bflow,dists,inds,
      qstart, stride0, n_h0, n_w0,
      ps,pt,ws_h,ws_w,wt,chnls,stride,dilation,
      use_search_abs, use_bounds, use_adj,
      full_ws, oh0, ow0, oh1, ow1,
      tranges, n_tranges, min_tranges);
  fprintf(stdout,"hi.\n");
}

py::dict Registrations() {
  py::dict dict;
  dict["example"] = EncapsulateFunction(jax_search_prod_with_index_forward);
  return dict;
}


// void _jax_search_prod_with_index_forward(
//     torch::Tensor vid0,torch::Tensor vid1,
//     torch::Tensor fflow,torch::Tensor bflow,
//     torch::Tensor nlDists,torch::Tensor nlInds,
//     int qstart, int stride0, int n_h0, int n_w0,
//     int ps, int pt, int ws_h, int ws_w, int wt,
//     int chnls, int stride, int dilation,
//     bool use_search_abs, bool use_bounds, bool use_adj,
//     bool full_ws, int oh0, int ow0, int oh1, int ow1,
//     torch::Tensor tranges,
//     torch::Tensor n_tranges,torch::Tensor min_tranges){
//   CHECK_INPUT(vid0);
//   CHECK_INPUT(vid1);
//   CHECK_INPUT(fflow);
//   CHECK_INPUT(bflow);
//   CHECK_INPUT(nlDists);
//   CHECK_INPUT(nlInds);
//   CHECK_INPUT(tranges);
//   CHECK_INPUT(n_tranges);
//   CHECK_INPUT(min_tranges);
//   search_prod_with_index_forward_cuda(
//           vid0,vid1,fflow,bflow,nlDists,nlInds,
//           qstart, stride0, n_h0, n_w0,
//           ps,pt,ws_h,ws_w,wt,chnls,stride,dilation,
//           use_search_abs, use_bounds, use_adj,
//           full_ws, oh0, ow0, oh1, ow1,
//           tranges, n_tranges, min_tranges);
// }




// python bindings
void init_prod_with_index_search(py::module &m){
  m.def("search_prod_with_index_forward", &search_prod_with_index_forward,
        "DNLS Search (Prod) Forward (CUDA)");
  m.def("search_prod_with_index_backward", &search_prod_with_index_backward,
        "DNLS Search (Prod) Backward (CUDA)");
  m.def("reg", &Registrations,"Jax Forward ");
  // m.def("jax_search_prod_with_index_backward", &jax_search_prod_with_index_forward,
  //       "Jax Forward ");
}

