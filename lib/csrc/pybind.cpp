#include <torch/extension.h>

// -- search --
void init_non_local_search(py::module &);
void init_refinement(py::module &);

// -- nn --
void init_pfc(py::module &);
void init_accumulate_flow(py::module &);
void init_temporal_inds(py::module &);
void init_unique_topk(py::module &);
void init_interpolate_inds(py::module &);
void init_anchor_self(py::module &);
void init_jitter_unique_inds(py::module &);
void init_topk_pwd(py::module &);

// -- reducing --
void init_wpsum(py::module &);
void init_iwpsum(py::module &);
void init_wpsum_heads(py::module &);
void init_wpsum_heads_2vid(py::module &);

// -- tile --
void init_fold(py::module &);
void init_ifold(py::module &);
void init_ifoldz(py::module &);
void init_unfold(py::module &);
void init_iunfold(py::module &);

// -- tile k --
void init_foldk(py::module &);
void init_unfoldk(py::module &);

// -- dev/search --
void init_l2_search(py::module &);
void init_l2_dists(py::module &);
void init_l2_with_index_search(py::module &);
void init_l2_search_with_heads(py::module &);
void init_prod_search(py::module &);
void init_prod_with_index_search(py::module &);
void init_prod_pf_with_index_search(py::module &);
void init_window_search(py::module &);
void init_prod_search_with_heads(py::module &);
void init_prod_dists(py::module &);
void init_prod_refine(py::module &);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  // -- search --
  init_non_local_search(m);
  init_refinement(m);

  // -- nn --
  init_pfc(m);
  init_accumulate_flow(m);
  init_temporal_inds(m);
  init_unique_topk(m);
  init_interpolate_inds(m);
  init_anchor_self(m);
  init_jitter_unique_inds(m);
  init_topk_pwd(m);


  // -- reducers --
  init_wpsum(m);
  init_iwpsum(m);
  init_wpsum_heads(m);
  init_wpsum_heads_2vid(m);

  // -- tile --
  init_fold(m);
  init_ifold(m);
  init_ifoldz(m);
  init_unfold(m);
  init_iunfold(m);

  // -- tile_k --
  init_foldk(m);
  init_unfoldk(m);


  // -- dev/search --
  init_l2_search(m);
  init_l2_dists(m);
  init_l2_with_index_search(m);
  init_l2_search_with_heads(m);
  init_window_search(m);
  init_prod_search(m);
  init_prod_dists(m);
  init_prod_refine(m);
  init_prod_with_index_search(m);
  init_prod_pf_with_index_search(m);
  init_prod_search_with_heads(m);

}


