from setuptools import setup
# from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(name='dnls',
      packages=["dnls"],
      ext_modules=[
          CUDAExtension('dnls_cuda', [
              'dnls_cuda.cpp',
              'dnls_cuda_kernel.cu',
          ])
      ],
      cmdclass={'build_ext': BuildExtension})
