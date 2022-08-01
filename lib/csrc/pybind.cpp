#include <torch/extension.h>
// #include "pybind.hpp"

// void init_basic(py::module &);

// -- searching --
void init_l2_search(py::module &);
void init_l2_with_index_search(py::module &);
void init_prod_search(py::module &);

// -- reducing --
void init_wpsum(py::module &);

// -- patch db --
void init_gather(py::module &);
void init_scatter(py::module &);

// -- batched fold/unfold --
void init_fold(py::module &);
void init_unfold(py::module &);
void init_iunfold(py::module &);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_l2_search(m);
  init_l2_with_index_search(m);
  init_prod_search(m);

  init_wpsum(m);

  init_gather(m);
  init_scatter(m);

  init_fold(m);
  init_unfold(m);
  init_iunfold(m);

}


