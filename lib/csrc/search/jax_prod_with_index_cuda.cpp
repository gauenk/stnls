#include <torch/extension.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include <stdio.h>
#include <bit>
#include <iostream>


// jax wrappers
void search_prod_with_index_forward(cudaStream_t stream, void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len){
  fprintf(stdout,"jax-it!\n");
}

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_trivially_constructible<To>::value,
      "This implementation additionally requires destination type to be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["example"] = EncapsulateFunction(search_prod_with_index_forward);
  return dict;
}

// python bindings
void init_prod_with_index_search(pybind11::module &m){
  m.def("search_prod_with_index_backward_reg", &Registrations,
        "Jax Forward ");
}
