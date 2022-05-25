#include <torch/extension.h>
// #include "pybind.hpp"

// void init_basic(py::module &);
void init_gather(py::module &);
void init_scatter(py::module &);
void init_search(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // init_basic(m);
  init_gather(m);
  init_scatter(m);
  init_search(m);
}


