from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='stnls',
      package_dir={"": "lib"},
      packages=find_packages("."),
      package_data={'': ['*.so']},
      include_package_data=True,
      ext_modules=[
          CUDAExtension('stnls_cuda', [
              # -- search --
              'lib/csrc/search/non_local_search.cpp', # Search across Space-Time
              'lib/csrc/search/non_local_search_int_kernel.cu',
              'lib/csrc/search/non_local_search_bilin2d_kernel.cu',
              'lib/csrc/search/refinement.cpp', # Search over K offsets
              'lib/csrc/search/refinement_int_kernel.cu',
              'lib/csrc/search/refinement_bilin2d_kernel.cu',
              'lib/csrc/search/paired_search.cpp', # Paired Search
              'lib/csrc/search/paired_search_kernel.cu',
              'lib/csrc/search/paired_refine.cpp', # Paired Refinement
              'lib/csrc/search/paired_refine_kernel.cu',
              'lib/csrc/search/mat_mult1.cpp', # Space-Time Search (Pair of Frames)
              'lib/csrc/search/mat_mult1_kernel.cu',
              # -- nn --
              'lib/csrc/nn/accumulate_flow.cpp', # Compute All Accumulated Flows
              'lib/csrc/nn/accumulate_flow_kernel.cu',
              'lib/csrc/nn/search_flow.cpp', # Compute Accumulated Flows only for Search
              'lib/csrc/nn/search_flow_kernel.cu',
              'lib/csrc/nn/anchor_self.cpp', # Anchoring Self
              'lib/csrc/nn/anchor_self_kernel.cu',
              'lib/csrc/nn/non_local_inds.cpp', # Compute Non-Local Indices from Params
              'lib/csrc/nn/non_local_inds_kernel.cu',
              # -- agg --
              'lib/csrc/agg/gather.cpp', # Non-Local Gather
              'lib/csrc/agg/gather_int_kernel.cu',
              'lib/csrc/agg/gather_bilin2d_kernel.cu',
              # 'lib/csrc/agg/scatter.cpp', # Non-Local Scatter
              # 'lib/csrc/agg/scatter_int_kernel.cu',
              # 'lib/csrc/agg/scatter_bilin2d_kernel.cu',
              'lib/csrc/agg/wpsum.cpp', # Weighted Patch Sum
              'lib/csrc/agg/wpsum_int_kernel.cu',
              'lib/csrc/agg/wpsum_bilin2d_kernel.cu',
              # -- setup --
              'lib/csrc/pybind.cpp',
          ],
           extra_compile_args={'cxx': ['-g','-w'],
                               'nvcc': ['-O2']})
      ],
      cmdclass={'build_ext': BuildExtension},
)

