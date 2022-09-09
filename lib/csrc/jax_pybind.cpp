// #include <torch/extension.h>
#include <pybind11/pybind11.h>

// void init_basic(py::module &);

// -- searching --
void init_prod_with_index_search(pybind11::module &);

PYBIND11_MODULE(dnls_jax, m) {
  init_prod_with_index_search(m);
}
