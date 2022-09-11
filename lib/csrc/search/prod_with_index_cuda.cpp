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
void search_prod_with_index_forward_jax(cudaStream_t stream, void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len){
  fprintf(stdout,"hi.\n");

  // -- init memory types --
  auto vid0_ptr = reinterpret_cast<float*>(buffers[0]);
  auto vid1_ptr = reinterpret_cast<float*>(buffers[1]);
  auto fflow_ptr = reinterpret_cast<float*>(buffers[2]);
  auto bflow_ptr = reinterpret_cast<float*>(buffers[3]);
  auto tranges_ptr = reinterpret_cast<int32_t*>(buffers[4]);
  auto n_tranges_ptr = reinterpret_cast<int32_t*>(buffers[5]);
  auto min_tranges_ptr = reinterpret_cast<int32_t*>(buffers[6]);
  auto ishapes_ptr = reinterpret_cast<int32_t*>(buffers[7]);
  auto dists_ptr = reinterpret_cast<float*>(buffers[8]);
  auto inds_ptr = reinterpret_cast<int32_t*>(buffers[9]);


  // -- init options --
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).\
    layout(torch::kStrided).device(torch::kCUDA, 0);
  auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).\
    layout(torch::kStrided).device(torch::kCUDA, 0);

  // -- unpack from ishapes --
  auto ishapes_th = torch::from_blob(ishapes_ptr,{28},options_i32).to(torch::kCPU);
  auto ishapes = ishapes_th.accessor<int,1>();
    // ishapes = jnp.array([0, nqueries,ws_h,ws_w,wt,
    //                      k, ps, pt, chnls,
    //                      stride0, stride1, dilation,
    //                      use_search_abs, reflect_bounds, use_adj,
    //                      oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
    //                      use_rand, exact, nframes, 0, 0, 0],dtype=jnp.int32)

  int qstart = ishapes[0];
  int nqueries = ishapes[1];
  int ws_h = ishapes[2];
  int ws_w = ishapes[3];
  int wt = ishapes[4];
  int k = ishapes[5];
  int ps = ishapes[6];
  int pt = ishapes[7];
  int chnls = ishapes[8];
  int stride0 = ishapes[9];
  int stride1 = ishapes[10];
  int dilation = ishapes[11];
  bool use_search_abs = ishapes[12] == 1;
  bool reflect_bounds = ishapes[13] == 1;
  bool use_adj = ishapes[14] == 1;
  int oh0 = ishapes[15];
  int ow0 = ishapes[16];
  int oh1 = ishapes[17];
  int ow1 = ishapes[18];
  bool remove_self = ishapes[19] == 1;
  bool full_ws = ishapes[20] == 1;

  int nframes = ishapes[24];
  int color = ishapes[25];
  int height = ishapes[26];
  int width = ishapes[27];

  int st = 2*wt+1;
  int n_h0 = height;
  int n_w0 = width;

  fprintf(stdout,"qstart: %d\n",qstart);
  fprintf(stdout,"nqueries: %d\n",nqueries);
  fprintf(stdout,"ws_h: %d\n",ws_h);
  fprintf(stdout,"ws_w: %d\n",ws_w);
  fprintf(stdout,"wt: %d\n",wt);
  fprintf(stdout,"k: %d\n",k);
  fprintf(stdout,"ps: %d\n",ps);
  fprintf(stdout,"pt: %d\n",pt);

  fprintf(stdout,"chnls: %d\n",chnls);
  fprintf(stdout,"stride0: %d\n",stride0);
  fprintf(stdout,"stride1: %d\n",stride1);
  fprintf(stdout,"dilation: %d\n",dilation);

  fprintf(stdout,"use_search_abs: %d\n",use_search_abs);
  fprintf(stdout,"reflect_bounds: %d\n",reflect_bounds);
  fprintf(stdout,"use_adj: %d\n",use_adj);

  fprintf(stdout,"oh0: %d\n",oh0);
  fprintf(stdout,"ow0: %d\n",ow0);
  fprintf(stdout,"oh1: %d\n",oh1);
  fprintf(stdout,"ow1: %d\n",ow1);


  fprintf(stdout,"remove_self: %d\n",remove_self);
  fprintf(stdout,"full_ws: %d\n",full_ws);

  fprintf(stdout,"nframes: %d\n",nframes);
  fprintf(stdout,"color: %d\n",color);
  fprintf(stdout,"height: %d\n",height);
  fprintf(stdout,"width: %d\n",width);

  fprintf(stdout,"n_h0: %d\n",n_h0);
  fprintf(stdout,"n_w0: %d\n",n_w0);

  // -- create writable tensors --
  // auto dists = torch::zeros({nqueries,ws_h,ws_w,st},options_f32);
  // auto inds = torch::zeros({nqueries,ws_h,ws_w,st,3},options_i32);
  auto dists = torch::from_blob(dists_ptr,{nqueries,ws_h,ws_w,st},options_f32);
  auto inds = torch::from_blob(inds_ptr,{nqueries,ws_h,ws_w,st,3},options_i32);

  // -- create tensors --
  auto vid0 = torch::from_blob(vid0_ptr,{nframes,color,height,width},options_f32);
  auto vid1 = torch::from_blob(vid1_ptr,{nframes,color,height,width},options_f32);
  auto fflow = torch::from_blob(fflow_ptr,{nframes,2,height,width},options_f32);
  auto bflow = torch::from_blob(bflow_ptr,{nframes,2,height,width},options_f32);

  auto tranges = torch::from_blob(tranges_ptr,{nframes,nframes},options_i32);
  auto n_tranges = torch::from_blob(n_tranges_ptr,{nframes},options_i32);
  auto min_tranges = torch::from_blob(min_tranges_ptr,{nframes},options_i32);

  // -- run program --
  chnls = chnls <= 0 ? color : chnls;
  search_prod_with_index_forward(
      vid0,vid1,fflow,bflow,dists,inds,
      qstart, stride0, n_h0, n_w0,
      ps,pt,ws_h,ws_w,wt,chnls,stride1,dilation,
      use_search_abs, reflect_bounds, use_adj,
      full_ws, oh0, ow0, oh1, ow1,
      tranges, n_tranges, min_tranges);
  fprintf(stdout,"hi.\n");

  // -- copy result back --
  // dists_b.copy_(dists);
  // inds_b.copy_(inds);
  // std::memcpy(dists_ptr,dists.data_ptr(),sizeof(float)*dists.numel());
  // std::memcpy(inds_ptr,inds.data_ptr(),sizeof(int32_t)*inds.numel());
  // void* start = dists.data_ptr();
  // void* end = start + sizeof(float)*dists.numel();
  // thrust::memcpy(thrust::device,start,end,data_ptr);

  // -- view --
  auto vid0_ = vid0.to(torch::kCPU);
  auto vid0_a = vid0_.accessor<float,4>();
  fprintf(stdout,"%2.3f,%2.3f\n",vid0_a[0][0][0][0],vid0_a[0][0][0][1]);
  fprintf(stdout,"%2.3f,%2.3f\n",vid0_a[0][0][1][0],vid0_a[0][0][1][1]);

  auto dists_ = dists.to(torch::kCPU);
  auto dists_a = dists_.accessor<float,4>();
  fprintf(stdout,"%2.3f,%2.3f\n",dists_a[0][0][0][0],dists_a[0][0][0][1]);
  fprintf(stdout,"%2.3f,%2.3f\n",dists_a[0][0][1][0],dists_a[0][0][1][1]);

}

py::dict search_prod_with_index_jax() {
  py::dict dict;
  dict["forward"] = EncapsulateFunction(search_prod_with_index_forward_jax);
  dict["backward"] = EncapsulateFunction(search_prod_with_index_forward_jax);
  return dict;
}

// python bindings
void init_prod_with_index_search(py::module &m){
  m.def("search_prod_with_index_forward", &search_prod_with_index_forward,
        "DNLS Search (Prod) Forward (CUDA)");
  m.def("search_prod_with_index_backward", &search_prod_with_index_backward,
        "DNLS Search (Prod) Backward (CUDA)");
  m.def("search_prod_with_index_jax", &search_prod_with_index_jax,"Jax Forward/Backward");
}

