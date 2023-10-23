#include <torch/extension.h>

// -- search --
void init_non_local_search(py::module &);
void init_refinement(py::module &);
void init_paired_search(py::module &);
void init_n3net_matmult1(py::module &);


// -- nn --
void init_accumulate_flow(py::module &);
void init_search_flow(py::module &);
void init_anchor_self(py::module &);
void init_non_local_inds(py::module &);

// -- agg --
void init_wpsum(py::module &);
void init_non_local_stack(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  // -- search --
  init_non_local_search(m);
  init_refinement(m);
  init_paired_search(m);
  init_n3net_matmult1(m);

  // -- nn --
  init_accumulate_flow(m);
  init_search_flow(m);
  init_anchor_self(m);
  init_non_local_inds(m);

  // -- agg --
  init_wpsum(m);
  init_non_local_stack(m);

}


