#include <torch/extension.h>
// #include <pybind11/pybind11.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>


// void init_basic(py::module &);

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
py::capsule EncapsulateFunction(T* fn) {
  return py::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

// // -- searching --
// void init_prod_with_index_search(pybind11::module &);

// PYBIND11_MODULE(dnls_jax, m) {
//   init_prod_with_index_search(m);
// }
