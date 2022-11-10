#include <torch/extension.h>
// #include "pybind.hpp"

// void init_basic(py::module &);

// -- searching --
void init_l2_search(py::module &);
void init_l2_dists(py::module &);
void init_l2_with_index_search(py::module &);
void init_l2_search_with_heads(py::module &);
void init_prod_search(py::module &);
void init_prod_with_index_search(py::module &);
void init_window_search(py::module &);
void init_prod_search_with_heads(py::module &);
void init_prod_dists(py::module &);
void init_prod_refine(py::module &);
void init_unique_topk(py::module &);
// void init_prod_search_patches_with_heads(py::module &);

// -- reducing --
void init_wpsum(py::module &);
void init_iwpsum(py::module &);
void init_wpsum_heads(py::module &);
void init_wpsum_heads_2vid(py::module &);

// -- tile k --
void init_foldk(py::module &);
void init_unfoldk(py::module &);

// -- batched fold/unfold --
void init_fold(py::module &);
void init_ifold(py::module &);
void init_ifoldz(py::module &);
void init_unfold(py::module &);
void init_iunfold(py::module &);

// -- nn --
void init_pfc(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_l2_search(m);
  init_l2_dists(m);
  init_l2_with_index_search(m);
  init_l2_search_with_heads(m);
  init_window_search(m);
  init_prod_search(m);
  init_prod_dists(m);
  init_prod_refine(m);
  init_unique_topk(m);
  init_prod_with_index_search(m);
  init_prod_search_with_heads(m);
  // init_prod_search_patches_with_heads(m);

  init_wpsum(m);
  init_iwpsum(m);
  init_wpsum_heads(m);
  init_wpsum_heads_2vid(m);

  init_foldk(m);
  init_unfoldk(m);

  init_fold(m);
  init_ifold(m);
  init_ifoldz(m);
  init_unfold(m);
  init_iunfold(m);

  init_pfc(m);
}


