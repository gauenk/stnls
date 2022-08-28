from setuptools import setup
# from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(name='dnls',
      packages=["dnls"],
      ext_modules=[
          CUDAExtension('dnls_cuda', [
              'csrc/tile_k/foldk_cuda.cpp',
              'csrc/tile_k/foldk_kernel.cu',
              'csrc/tile_k/unfoldk_cuda.cpp',
              'csrc/tile_k/unfoldk_kernel.cu',
              'csrc/search/l2_cuda.cpp',
              'csrc/search/l2_kernel.cu',
              'csrc/search/l2_dists_cuda.cpp',
              'csrc/search/l2_dists_kernel.cu',
              'csrc/search/l2_with_index_cuda.cpp',
              'csrc/search/l2_with_index_kernel.cu',
              'csrc/search/window_search.cpp',
              'csrc/search/window_search_kernel.cu',
              'csrc/search/prod_cuda.cpp',
              'csrc/search/prod_kernel.cu',
              'csrc/search/prod_with_index_cuda.cpp',
              'csrc/search/prod_with_index_kernel.cu',
              'csrc/search/prod_search_with_heads.cpp',
              'csrc/search/prod_search_with_heads_kernel.cu',
              'csrc/tile/fold_cuda.cpp',
              'csrc/tile/fold_kernel.cu',
              'csrc/tile/ifold_cuda.cpp',
              'csrc/tile/ifold_kernel.cu',
              'csrc/tile/ifoldz_cuda.cpp',
              'csrc/tile/ifoldz_kernel.cu',
              'csrc/tile/unfold_cuda.cpp',
              'csrc/tile/unfold_kernel.cu',
              'csrc/tile/iunfold_cuda.cpp',
              'csrc/tile/iunfold_kernel.cu',
              'csrc/reducers/wpsum_cuda.cpp',
              'csrc/reducers/wpsum_kernel.cu',
              'csrc/reducers/iwpsum_cuda.cpp',
              'csrc/reducers/iwpsum_kernel.cu',
              'csrc/reducers/wpsum_heads_cuda.cpp',
              'csrc/reducers/wpsum_heads_kernel.cu',
              'csrc/reducers/wpsum_heads_2vid_cuda.cpp',
              'csrc/reducers/wpsum_heads_2vid_kernel.cu',
              'csrc/pybind.cpp',
          ])
      ],
      cmdclass={'build_ext': BuildExtension},
      extra_cuda_cflags=['-lineinfo']
      #extra_cuda_cflags=['--generate-line-info']
)
