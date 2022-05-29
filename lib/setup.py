from setuptools import setup
# from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(name='dnls',
      packages=["dnls"],
      ext_modules=[
          CUDAExtension('dnls_cuda', [
              'csrc/dnls_gather_cuda.cpp',
              'csrc/dnls_gather_kernel.cu',
              'csrc/dnls_scatter_cuda.cpp',
              'csrc/dnls_scatter_kernel.cu',
              'csrc/dnls_search_cuda.cpp',
              'csrc/dnls_search_kernel.cu',
              'csrc/dnls_fold_cuda.cpp',
              'csrc/dnls_fold_kernel.cu',
              'csrc/dnls_unfold_cuda.cpp',
              'csrc/dnls_unfold_kernel.cu',
              'csrc/pybind.cpp',
          ])
      ],
      cmdclass={'build_ext': BuildExtension}
)
