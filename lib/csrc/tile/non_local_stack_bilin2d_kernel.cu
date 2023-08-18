/*

  Stack non-local patches into a video


 */


// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
// #include "shared_tile_kernels.cu"
#include "nlstack_bilin2d.cu"


/****************************

       Helper Funcs

****************************/


#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)

__inline__ __device__ int bounds(int val, int lb, int ub ){
  int vval = val;
  if (val < lb){
    vval = 2*lb - val;
  }else if (val >= ub){
    vval = 2*(ub-1) - val;
  }
  return vval;
}


/**************************************

          Forward Pass

**************************************/


template <typename scalar_t>
__global__ void non_local_stack_bilin2d_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> stack,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> counts,
    int ps, int pt, int dilation, int stride0, int patch_offset,
    int nW0, int nHW0, bool reflect_bounds, int q_start,
    int off_H0, int off_H1, int off_W0, int off_W1, int ftrs_per_thread){

    // -- unpack --
    int nbatch = vid.size(0);
    int nheads = vid.size(1);
    int nframes = vid.size(2);
    int nftrs = vid.size(3);
    int height = vid.size(4);
    int width = vid.size(5);
    int Q = inds.size(2);
    int K = inds.size(3);

    // -- indexing variables --
    int qindex,qindex_tmp,iftr;
    int ref_patch[3];
    scalar_t nl_patch[3];
    int ref[3];
    scalar_t nl[3];
    int nl_i[3];
    int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};
    scalar_t pix;
    bool valid;
    bool valid_ref[4];
    bool valid_nl[4];
  
    // -- location to fill --
    int qi = blockIdx.x*blockDim.x+threadIdx.x;
    int ki = blockIdx.y*blockDim.y+threadIdx.y;
    int ihead = blockIdx.z/nbatch;
    int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;
  
    // -- feature chunk --
    int ftr_start = threadIdx.z * ftrs_per_thread;
    int ftr_end = min(nftrs,ftr_start + ftrs_per_thread);
    
    // -- each region --
    if ((qi < Q) && (ki < K)){
  
      //----------------------------------
      //   Reference & Non-Local Pixel
      //----------------------------------
  
      // -- full-resolution video query index --
      qindex = qi + q_start;
  
      // -- reference index --
      get_pixel_loc(ref_patch,qindex,qindex_tmp,stride0,nW0,nHW0,height,width);

      // -- non-local index --
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl_patch[_idx] = inds[ibatch][ihead][qi][ki][_idx];
      }
  
      //----------------------------------
      //      Fill Non-Local Patch
      //----------------------------------

      // scalar_t w = weights[ibatch][ihead][qi][ki];
      fill_non_local_patch_bilin2d<scalar_t>(stack[ibatch][ihead][ki],
                                            counts,vid[ibatch][ihead],
                                            weights[ibatch][ihead][qi][ki],
                                            ps,pt,dilation,reflect_bounds,
                                            ref_patch,nl_patch,ref,nl,nl_i,
                                            valid_ref,valid_nl,valid,
                                            center_offsets,patch_offset,
                                            iftr,ftr_start,ftr_end,
                                            nframes,height,width,pix,qi,ki);

    }

}

void non_local_stack_bilin2d_forward_cuda(
    const torch::Tensor vid, const torch::Tensor weights,
    const torch::Tensor inds, torch::Tensor stack, torch::Tensor counts,
    int ps, int pt, int dilation, int stride0, bool use_adj, bool reflect_bounds,
    int q_start, int off_H0, int off_W0, int off_H1, int off_W1){

  // -- sizes --
  int nbatch = vid.size(0);
  int nheads = vid.size(1);
  int nframes = vid.size(2);
  int nftrs = vid.size(3);
  int height = vid.size(4);
  int width = vid.size(5);

  // -- indexing vars --
  int nH0 = (height-1)/stride0+1;
  int nW0 = (width-1)/stride0+1;
  int nHW0 = nH0*nW0;
  int ps_offset = dilation*(ps/2);
  ps_offset = use_adj ? 0 : -ps_offset;

  // -- launch parameters --
  int nq = inds.size(2);
  int k = inds.size(3);
  int ftr_threads = min(15,nftrs);
  dim3 threadsPerBlock(10,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, nheads*nbatch);
  blocksPerGrid.x = ceil(double(nq)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
  int ftrs_per_thread = (nftrs-1)/ftr_threads+1;

  // -- allocate counts --
  // auto options = torch::TensorOptions()
  //   .dtype(torch::kInt32)
  //   .layout(torch::kStrided)
  //   .device(torch::kCUDA, vid.device().index());
  // auto counts = torch::zeros({height,width},options);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "non_local_stack_bilin2d_forward_kernel", ([&] {
        non_local_stack_bilin2d_forward_kernel<scalar_t>
          <<<blocksPerGrid, threadsPerBlock>>>(
           vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
           weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
           inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
           stack.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
           counts.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
           ps, pt, dilation, stride0, ps_offset, nW0, nHW0, reflect_bounds,
           q_start, off_H0, off_W0, off_H1, off_W1, ftrs_per_thread);
      }));

  // -- normalize --
  // counts = counts.view({1, 1, 1, 1, height, width});
  // stack /= counts;

}


/**************************************

          Backward Pass

**************************************/


template <typename scalar_t>
__global__ void non_local_stack_bilin2d_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_weights,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_inds,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> grad_stack,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
    // const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> counts,
    int ps, int pt, int dilation, int stride0, int patch_offset,
    int nW0, int nHW0, bool reflect_bounds, int q_start,
    int off_H0, int off_H1, int off_W0, int off_W1, int ftrs_per_thread){

    // -- unpack --
    int nbatch = vid.size(0);
    int nheads = vid.size(1);
    int nframes = vid.size(2);
    int nftrs = vid.size(3);
    int height = vid.size(4);
    int width = vid.size(5);
    int Q = inds.size(2);
    int K = inds.size(3);

    // -- indexing variables --
    int qindex,qindex_tmp,iftr;
    int ref_patch[3];
    scalar_t nl_patch[3];
    int ref[3];
    scalar_t nl[3];
    int nl_i[3];
    scalar_t pix;
    bool valid;
    bool valid_nl[4];
    bool valid_ref[4];
    int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};
  
    // -- location to fill --
    int qi = blockIdx.x*blockDim.x+threadIdx.x;
    int ki = blockIdx.y*blockDim.y+threadIdx.y;
    int ihead = blockIdx.z/nbatch;
    int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;
  
    // -- feature chunk --
    int ftr_start = threadIdx.z * ftrs_per_thread;
    int ftr_end = min(nftrs,ftr_start + ftrs_per_thread);
    
    // -- each region --
    if ((qi < Q) && (ki < K)){
  
      //----------------------------------
      //   Reference & Non-Local Pixel
      //----------------------------------
  
      // -- full-resolution video query index --
      qindex = qi + q_start;
  
      // -- reference index --
      get_pixel_loc(ref_patch,qindex,qindex_tmp,stride0,nW0,nHW0,height,width);
  
      // -- non-local index --
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl_patch[_idx] = inds[ibatch][ihead][qi][ki][_idx];
        // nl_patch[_idx] = ref_patch[_idx];//inds[ibatch][ihead][qi][ki][_idx];
      }
  
      //----------------------------------
      //      Fill Non-Local Patch
      //----------------------------------

      fill_non_local_patch_bwd_bilin2d
        <scalar_t>(grad_vid[ibatch][ihead],
                   grad_weights[ibatch][ihead],
                   grad_inds[ibatch][ihead],
                   // counts,
                   grad_stack[ibatch][ihead][ki],
                   // stack[ibatch][ihead][ki],
                   vid[ibatch][ihead],
                   weights[ibatch][ihead][qi][ki],
                   ps,pt,dilation,reflect_bounds,
                   ref_patch,nl_patch,ref,nl,nl_i,
                   valid_ref,valid_nl,valid,
                   center_offsets,patch_offset,
                   iftr,ftr_start,ftr_end,
                   nframes,height,width,pix,qi,ki);

    }

}

void non_local_stack_bilin2d_backward_cuda(
    torch::Tensor grad_vid,
    torch::Tensor grad_weights,
    torch::Tensor grad_inds,
    const torch::Tensor grad_stack,
    const torch::Tensor vid,
    const torch::Tensor weights,
    const torch::Tensor inds,
    const torch::Tensor stack,
    const torch::Tensor counts,
    int ps, int pt, int dilation, int stride0,
    bool use_adj, bool reflect_bounds,
    int off_H0, int off_W0, int off_H1, int off_W1){

  // -- sizes --
  int nbatch = vid.size(0);
  int nheads = vid.size(1);
  int nframes = vid.size(2);
  int nftrs = vid.size(3);
  int height = vid.size(4);
  int width = vid.size(5);

  // -- indexing vars --
  int nW0 = (width-1)/stride0+1;
  int nH0 = (height-1)/stride0+1;
  int nHW0 = nH0*nW0;
  int ps_offset = dilation*(ps/2);
  ps_offset = use_adj ? 0 : -ps_offset;
  int q_start = 0;
  // fprintf(stdout,"ps,pt,stride0: %d,%d,%d\n",ps,pt,stride0);

  // -- launch parameters --
  int nq = inds.size(2);
  int k = inds.size(3);
  int ftr_threads = min(15,nftrs);
  dim3 threadsPerBlock(10,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, nheads*nbatch);
  blocksPerGrid.x = ceil(double(nq)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
  int ftrs_per_thread = (nftrs-1)/ftr_threads+1;


  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "non_local_stack_bilin2d_backward_kernel", ([&] {
        non_local_stack_bilin2d_backward_kernel<scalar_t>
          <<<blocksPerGrid, threadsPerBlock>>>(
           grad_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
           grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
           grad_inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
           grad_stack.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
           vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
           weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
           inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
           // stack.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
           // counts.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
           ps, pt, dilation, stride0, ps_offset, nW0, nHW0, reflect_bounds,
           q_start, off_H0, off_W0, off_H1, off_W1, ftrs_per_thread);
      }));


}
