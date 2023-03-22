
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>

template <typename scalar_t>
__global__ void compare_inds_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds0,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds1,
    int qstart, int stride0, int q_per_thread){

  // -- starting qi for thread --
  int B = dists.size(0);
  int HD = dists.size(1);
  int Q = dists.size(2);
  int K = dists.size(3);
  int H = vid.size(4);
  int W = vid.size(5);

  // -- threads --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int ki = threadIdx.y;
  int qi_start = q_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int qi_end = min(qi_start+q_per_thread,Q);

  // -- fwd decl --
  int self_index = 0;
  bool eq_loc;
  int loc0[3];
  int loc1[3];
  int i_tmp[3];
  // scalar_t d_tmp;
  int qi,qindex,i_mod;


  // -- fwd decl patch dists --
  scalar_t dist,_dist;
  int loc0_pix[3];
  int loc1_pix[3];
  scalar_t pix0,pix1;
  bool valid0[4];
  bool valid1[4];

  // ref_pix, prop_pix, valid_ref, valid_prop,
  // ps,pt,dilation,reflect_bounds,
  // patch_offset,center_offsets,invalid,
  // T,C,H,W,pix0,pix1,_dist);

  // -- for each location --
  for (int qi = qi_start; qi < qi_end; qi++){

    // -- unpack pixel locs --
    // qindex = qi + qstart;
    // i_mod = qindex % nHW;
    // loc[0] = qindex / nHW;
    // loc[1] = ((i_mod / nW) * stride0) % H;
    // loc[2] = ((i_mod % nW) * stride0) % W;

    // -- unroll indices --
#pramga unroll    
    for (int i=0; i < 3; i++){
      loc0[i] = inds0[ibatch][ihead][qi][ki][i];
      loc1[i] = inds1[ibatch][ihead][qi][ki][i];
    }

    compute_dist<scalar_t,DIST_TYPE>(dist,
                vid0[ibatch][ihead],vid1[ibatch][ihead],
                loc0, loc1, loc0_pix, loc1_pix,
                valid0, valid1,
		ps,pt,dilation,reflect_bounds,
                patch_offset,center_offsets,invalid,
                T,C,H,W,pix0,pix1,_dist);


  }
}

void compare_inds_forward_cuda(
     torch::Tensor dists, torch::Tensor vid,
     torch::Tensor inds0, torch::Tensor inds1,
     int qstart, int stride0){
  
  // -- unpack --
  int B = dists.size(0);
  int HD = dists.size(1);
  int Q = dists.size(2);
  int K = dists.size(3);
  int HD = vid.size(1);
  int H = vid.size(4);
  int W = vid.size(5);

  // -- num 2 run --
  int nRun = Q;

  // -- kernel params --
  int q_per_thread = 1;
  int _nthreads = 512/K+1;
  dim3 nthreads(_nthreads,K);
  int _nblocks = (nRun-1)/(_nthreads*q_per_thread)+1;
  dim3 nblocks(_nblocks,B,Hd);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
  // fprintf(stdout,"nH,nW,stride0: %d,%d,%d\n",nH,nW,stride0);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(dists.type(), "compare_inds_kernel", ([&] {
     compare_inds_kernel<scalar_t><<<nblocks, nthreads>>>(
       dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
       vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
       inds0.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
       inds1.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
       qstart,stride0,q_per_thread);
      }));

}

