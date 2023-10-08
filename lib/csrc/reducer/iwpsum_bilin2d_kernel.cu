
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "share_kernel.cu"


/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void iwpsum_bilin2d_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> out_vid,
    torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> out_vidz,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> inds,
    int ps, int stride0, int pt, int dilation, bool reflect_bounds, int psOffset,
    int nH, int nW, int nHW, int q_per_thread, int f_per_thread){

    // -- shapes --
    int nbatch = in_vid.size(0);
    int nheads = in_vid.size(1);
    int nframes = in_vid.size(2);
    int nfeatures = in_vid.size(3);
    int height = in_vid.size(4);
    int width = in_vid.size(5);
    int nq = inds.size(2);
    int k = inds.size(3);

    // -- batching --
    int query_start = blockIdx.x*q_per_thread;
    int ibatch = blockIdx.y;
    int ihead = blockIdx.z;

    // -- cuda threads --
    int pi = threadIdx.y;
    int pj = threadIdx.z;
    int ftr_start = threadIdx.x*f_per_thread;
    int ftr_end = min(nfeatures,ftr_start + f_per_thread);

    // -- pixel locations --
    // int center_ti,center_hi,center_wi;
    int qi;
    bool valid,valid_ref;
    scalar_t pix,weight;
    int ref_t,ref_h,ref_w;
    int ref_ti,ref_hi,ref_wi;
    int qloc[3];
    scalar_t kloc[3];
    int ti;
    scalar_t hi,wi;
    int h_interp,w_interp;

    // -- range --
    for(int _qi = 0; _qi < q_per_thread; _qi++){

      // -- query index --
      qi = query_start + _qi;
      if (qi >= nq){ continue; }
      get_query_loc(qloc,qi,nW,nHW);

      // -- reference patch --
      // ref_t = inds[ibatch][ihead][qi][0][0];
      // ref_h = inds[ibatch][ihead][qi][0][1];
      // ref_w = inds[ibatch][ihead][qi][0][2];
      ref_t = qloc[0];
      ref_h = stride0*qloc[1];
      ref_w = stride0*qloc[2];

      // -- reference pixel index --
      ref_hi = ref_h+dilation*(pi + psOffset);
      ref_wi = ref_w+dilation*(pj + psOffset);
      ref_hi = reflect_bounds ? bounds(ref_hi,height) : ref_hi;
      ref_wi = reflect_bounds ? bounds(ref_wi,width) : ref_wi;

      // -- valid reference only --
      valid_ref = (ref_hi < height) && (ref_hi >= 0);
      valid_ref = valid_ref && (ref_wi < width) && (ref_wi >= 0);
      if (not valid_ref){ continue;}

      for(int ki = 0; ki < k; ki++){

        // -- non-local patch center --
        kloc[0] = inds[ibatch][ihead][qloc[0]][qloc[1]][qloc[2]][ki][0];
        kloc[1] = inds[ibatch][ihead][qloc[0]][qloc[1]][qloc[2]][ki][1];
        kloc[2] = inds[ibatch][ihead][qloc[0]][qloc[1]][qloc[2]][ki][2];
        weight = dists[ibatch][ihead][qloc[0]][qloc[1]][qloc[2]][ki];

        // -- non-local pixel index --
        hi = kloc[1]+dilation*(pi + psOffset);
        wi = kloc[2]+dilation*(pj + psOffset);
        hi = reflect_bounds ? bounds(hi,height) : hi;
        wi = reflect_bounds ? bounds(wi,width) : wi;

        // -- valid non-local patches only --
        valid = (hi >= 0) && (hi < height);
        valid = valid && (wi >= 0) && (wi < width);
        if (not valid){ continue; }

        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- time is always valid --
          ref_ti = bounds(ref_t + pk,nframes);
          ti = bounds(__float2int_rd(kloc[0]) + pk,nframes);

          // -- channels --
          for(int iftr = ftr_start; iftr < ftr_end; iftr++){

            // -- assign --
            bilin2d_interpolate(pix,hi,wi,height,width,
                                in_vid[ibatch][ihead][ti][iftr]);
            pix = weight*pix;

            // -- accumulate
            atomicAdd(&out_vid[ibatch][ihead][ref_ti][iftr][ref_hi][ref_wi],pix);

          } // channel-loop
        } // pt-loop
      } // k-loop

      // -- normalize --
      if ((threadIdx.x==0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (ref_t==0)){
        if (valid_ref){
          atomicAdd(&out_vidz[0][0][0][0][ref_hi][ref_wi],1);
        }
      }

    } // query-loop
}

void iwpsum_bilin2d_forward_cuda(
    torch::Tensor in_vid,
    torch::Tensor out_vid,
    torch::Tensor out_vidz,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj){

  // -- kernel blocks --
  int nbatch = inds.size(0);
  int nheads = inds.size(1);
  int nqueries = inds.size(2);
  int q_per_thread = 2;
  int qblocks = (nqueries-1)/q_per_thread+1;
  dim3 nblocks(qblocks,nbatch,nheads);

  // -- kernel threads --
  int nftrs = in_vid.size(3);
  int MAX_THREADS = 1024;
  int dim = ps*ps;
  int f_per_block = MAX_THREADS/dim; // num of nftrs per block
  int f_per_thread = ((nftrs - 1)/f_per_block) + 1; // num of nftrs per thread
  dim3 nthreads(f_per_block,ps,ps);

  // -- viz --
  // fprintf(stdout,"qblocks,nbatch,nheads: %d,%d,%d\n",qblocks,nbatch,nheads);
  // fprintf(stdout,"f_per_block,ps,ps: %d,%d,%d\n",f_per_block,ps,ps);
  
  // -- derived quantities --
  int nH = inds.size(3);
  int nW = inds.size(4);
  int nHW = nH * nW;
  int H = in_vid.size(4);
  int stride0 = ceil(H / nH);

  // -- shared --
  int psHalf = ps/2;
  int adj = use_adj ? psHalf : 0;
  int psOffset = adj - psHalf;

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(in_vid.type(), "iwpsum_bilin2d_forward_kernel", ([&] {
    iwpsum_bilin2d_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        in_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        out_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        out_vidz.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        ps, stride0, pt, dilation, reflect_bounds, psOffset,
        nH, nW, nHW, q_per_thread, f_per_thread);
    }));
}

