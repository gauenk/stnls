#include <torch/extension.h>

// -- search --
void init_non_local_search(py::module &);
void init_refinement(py::module &);
void init_paired_search(py::module &);
void init_paired_refine(py::module &);
void init_n3net_matmult1(py::module &);


// -- nn --
void init_accumulate_flow(py::module &);
void init_search_flow(py::module &);
void init_anchor_self(py::module &);
void init_non_local_inds(py::module &);

// -- agg --
void init_pool(py::module &);
void init_gather(py::module &);
void init_gather_add(py::module &);
void init_scatter(py::module &);
void init_scatter_add(py::module &);

// -- graph opts --
void init_scatter_tensor(py::module &m);
void init_gather_tensor(py::module &m);
void init_scatter_labels(py::module &m);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  // -- search --
  init_non_local_search(m);
  init_refinement(m);
  init_paired_search(m);
  init_paired_refine(m);
  init_n3net_matmult1(m);

  // -- nn --
  init_accumulate_flow(m);
  init_search_flow(m);
  init_anchor_self(m);
  init_non_local_inds(m);

  // -- agg --
  init_pool(m);
  init_gather(m);
  init_gather_add(m);
  init_scatter(m);
  init_scatter_add(m);

  // -- graph opts --
  init_scatter_tensor(m);
  init_gather_tensor(m);
  init_scatter_labels(m);

}


