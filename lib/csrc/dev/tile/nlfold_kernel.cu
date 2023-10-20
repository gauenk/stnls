/*

  Execute ifoldz _with_ a normalization image.

 */


// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "shared_tile_kernels.cu"

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
__global__ void stnls_nlfold_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> zvid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    int stride, int dilation, int ps_offset, bool reflect,
    int nH, int nW, int nHW, int HW, int THW) {

    // -- unpack --
    int B = vid.size(0);
    int T = vid.size(1);
    int F = vid.size(2);
    int H = vid.size(3);
    int W = vid.size(4);
    int Q = patches.size(1);
    int pt = patches.size(2);
    int ps = patches.size(5);

    // -- indexing --
    int psHalf = ps/2;
    int ref[3],pix[3],ref_orig[2];
    bool valid_ref,valid_refl,valid_pix,valid_q;
    int qi,_tmp;

    // -- cuda indexing --
    int ibatch = blockIdx.y;

    CUDA_KERNEL_LOOP(_index, THW) {

      // -- a pixel in video --
      get_pixel_loc(pix, _index, _tmp, 1, W, HW, H, W);
      // pix[1] = bounds(pix[1],0,H);
      // pix[2] = bounds(pix[2],0,W);
      // check_bounds(valid_pix, pix, T, H, W);
      if (!valid_pix){ continue; }

      // -- for each reference location within radius (pt,ps,ps)  --
      for(int fi = 0; fi < F; fi++){
        scalar_t val = 0;
        scalar_t zval = 0;
        for (int pk = 0; pk < pt; pk++){
          for (int pi = 0; pi < ps; pi++){
            for (int pj = 0; pj < ps; pj++){

              // -- potential reference for the pixel --
              ref[0] = bounds(pix[0] + pk,0,T); // maybe "-pk"
              ref[1] = pix[1] + dilation*(pi - ps_offset);
              ref[2] = pix[2] + dilation*(pj - ps_offset);
              check_bounds(valid_ref, ref, T, H, W);
              // ref_orig[0] = ref[1];
              // ref_orig[1] = ref[2];
              // ref[1] = reflect ? bounds(ref[1],0,H) : ref[1];
              // ref[2] = reflect ? bounds(ref[2],0,W) : ref[2];
              // check_bounds(valid_refl, ref, T, H, W);
              // valid_ref = valid_ref && valid_refl;
              // check_bounds(valid_ref, ref, T, H, W);

              // -- testing only --
              // valid_ref = (ref[0]==pix[0]);
              // valid_ref = valid_ref && (ref[1]==pix[1]);
              // valid_ref = valid_ref && (ref[2]==pix[2]);

              // -- valid only if on "ref" grid --
              valid_ref = valid_ref && (ref[1] % stride == 0);
              valid_ref = valid_ref && (ref[2] % stride == 0);
              if (!valid_ref){ continue; }

              // -- query index from reference --
              qi = ref[0] * nHW;
              qi += ((ref[1]/stride) * nW);
              qi += ref[2]/stride;

              // -- index of pixel within anchor patch --
              int h_ip = ps-1-pi;
              int w_ip = ps-1-pj;
              // int h_ip = pj;//ps-1-pj;
              // int w_ip = pi;//ps-1-pi;

              // -- reflect to match --
              // if (ref_orig[0] > ref[1]){
              //   h_ip = pi;
              //   valid_ref = valid_ref && (h_ip < psHalf);
              // }
              // else if(ref_orig[0] < ref[1]){
              //   h_ip = pi;
              //   valid_ref = valid_ref && (h_ip > psHalf);
              // }

              // if (ref_orig[1] > ref[2]){
              //   w_ip = pj;
              //   valid_ref = valid_ref && (w_ip < psHalf);
              // }
              // else if(ref_orig[1] < ref[2]){
              //   w_ip = pj;
              //   valid_ref = valid_ref && (w_ip > psHalf);
              // }

              // -- accumulate --
              valid_q = valid_ref && (qi >= 0) && (qi < Q);
              if (valid_q){
                val += patches[ibatch][qi][pk][fi][h_ip][w_ip];
                zval += 1;
              }
            }
          } // for patch size
        } // for patch size

        vid[ibatch][pix[0]][fi][pix[1]][pix[2]] = val;
        zvid[ibatch][pix[0]][fi][pix[1]][pix[2]] = zval;

      } // for nftrs
    } // for each pixel (with stride)
}

void stnls_cuda_nlfold_forward(
    torch::Tensor vid, torch::Tensor zvid, torch::Tensor patches,
    int stride, int dilation, bool use_adj, bool reflect){

  // batching entire image always
  int nbatch = vid.size(0);
  int T = vid.size(1);
  int F = vid.size(2);
  int H = vid.size(3);
  int W = vid.size(4);
  int ps = patches.size(5);
  int Q = patches.size(1);

  // -- size --
  int nH = (H-1)/stride+1;
  int nW = (W-1)/stride+1;
  int nHW = nH * nW;
  int HW = H * W;
  int THW = T * H * W;
  int ps_offset = use_adj ? 0 : ps/2;
  assert(Q == T*nHW);

  // launch params
  int nthreads = 512;
  int nblocks_queries = (THW-1) / nthreads+1;
  dim3 nblocks(nblocks_queries,nbatch);
  // fprintf(stdout,"ps,stride,reflect: %d,%d,%d\n",ps,stride,reflect);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "stnls_nlfold_forward_kernel", ([&] {
    stnls_nlfold_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        zvid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        stride,dilation,ps_offset,reflect,nH,nW,nHW,HW,THW);
      }));
}

/**************************************

         Backward Pass

**************************************/

template <typename scalar_t>
__global__ void stnls_nlfold_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_patches,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_vid,
    int stride, int dilation, int ps_offset, bool reflect, int nW, int nHW) {

    // -- shapes --
    int T = grad_vid.size(1); // nframes
    int F = grad_vid.size(2); // num features
    int H = grad_vid.size(3); // height
    int W = grad_vid.size(4); // width
    int Q = grad_patches.size(1);
    int pt = grad_patches.size(2);
    int ps = grad_patches.size(4);

    // -- vars --
    int _qindex;
    bool valid_ref,valid_pix,valid_t,valid;
    int ref[3],pix[3];
    scalar_t pix_v;

    // -- cuda indexing --
    int pi = threadIdx.y;
    int pj = threadIdx.z;
    int ibatch = blockIdx.y;

    // -- cuda iterate over queries  --
    CUDA_KERNEL_LOOP(_index, Q) {

      // -- reference index --
      int qi = _index;
      get_pixel_loc(ref, qi, _qindex, stride, nW, nHW, H, W);
      check_bounds(valid_ref, ref, T, H, W);
      if (!valid_ref){ continue; }

      // -- iterate over loop --
      for(int pk = 0; pk < pt; pk++){

        // -- get pixel --
        pix[0] = bounds(ref[0] + pk,0,T); // maybe "-pk"
        pix[1] = ref[1] + dilation*(pi - ps_offset);
        pix[2] = ref[2] + dilation*(pj - ps_offset);
        pix[1] = reflect ? bounds(pix[1],0,H) : pix[1];
        pix[2] = reflect ? bounds(pix[2],0,W) : pix[2];
        check_bounds(valid_pix, pix, T, H, W);
        if (!valid_pix){ continue; }

        // -- nftrs --
        for(int fi = 0; fi < F; fi++){
          pix_v = grad_vid[ibatch][pix[0]][fi][pix[1]][pix[2]];
          grad_patches[ibatch][qi][pk][fi][pi][pj] = pix_v;
        }
      }

    }
}

void stnls_cuda_nlfold_backward(
    torch::Tensor grad_patches,
    const torch::Tensor grad_vid,
    // const torch::Tensor vid,
    // const torch::Tensor zvid,
    // const torch::Tensor patches,
    int stride, int dilation,
    bool use_adj, bool reflect) {

  // -- kernel blocks --
  int B = grad_vid.size(0);
  int nbatch = B;
  int H = grad_vid.size(3);
  int W = grad_vid.size(4);
  int Q = grad_patches.size(1);

  // int k = 1;
  // int qpt = 10;
  // int nblocks_q = (Q-1)/qpt+1;
  int pt = grad_patches.size(2);
  int ps = grad_patches.size(5);
  int ps_offset = use_adj ? 0 : ps/2;
  assert(pt == 1);

  // -- size --
  int nH = (H-1)/stride+1;
  int nW = (W-1)/stride+1;
  int nHW = nH * nW;

  // -- launch params --
  int nthreads_q = 1024 / (ps*ps);
  int nblocks_queries = (Q-1) / nthreads_q+1;
  dim3 nblocks(nblocks_queries,nbatch);
  dim3 nthreads(nthreads_q,ps,ps);
  // fprintf(stdout,"ps,stride,dilation,reflect: %d,%d,%d,%d\n",
  //         ps,stride,dilation,reflect);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(grad_patches.type(), "stnls_nlfold_backward_kernel", ([&] {
    stnls_nlfold_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        grad_patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        grad_vid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        // vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        // zvid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        // patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        stride, dilation, ps_offset, reflect, nW, nHW);
  }));

}
