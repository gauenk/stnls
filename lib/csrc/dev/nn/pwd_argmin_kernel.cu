

/*******

        Incomplete!

 *******/

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>
// #include "shared_nn_utils.cu"
#include "../search/shared_kernel.cu"


template <typename scalar_t>
__global__ void pwd_argmin_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> pwds,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> mins,
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> argmins,
    int q_per_thread){

  // -- unpack shape --
  int B = pwds.size(0);
  int HD = pwds.size(1);
  int Q = pwds.size(3);
  int nPWD = pwds.size(3);
  int K = mins.size(3);

  // invalid value 
  float invalid = __int_as_float(0x7f800000);

  // -- search region offsets --
  int psHalf = (ps)/2;
  int adj = use_adj ? psHalf : 0;

  // -- cuda index --
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;
  int qi_start = (threadIdx.x + blockIdx.x*blockDim.x)*q_per_thread;
  int qi_end = min(qi_start+q_per_thread,Q);
  int row = threadIdx.y;

  // decls
  int arg[2];
  int val[2];
  
  // 2dim -> 1dim (pwd indices)
  int k0,k1;
  int xform_p,pwd_i;
  bool swap;

  // sorting
  int kidx;
  scalar_t dist,dmax,dcurr;

  // -- for each location --
  for (int qi = qi_start; qi < qi_end; qi++){
    auto pwds_q = pwds[ibatch][ihead][qi];
    auto mins_q = mins[ibatch][ihead][qi];
    auto argmins_q = argmins[ibatch][ihead][qi];
    for (int col = 0; col < K-1; col++){

      // -- convert (row,col) in [0,1,...,K] to (k0,k1) in lower_triangular  --
      col = (col == row) ? col + 1 : col;
      swap = row > col;
      k0 = swap ? row : col;
      k1 = swap ? col : row;

      // -- convert (k0,k1) to (pwd_i) --
      xform_p = k0 - 1;
      pwd_i = k1 + int(xform_p*(xform_p+1)/2.);

      // -- read --
      dist = pwds_q[pwd_i];

      if (dist < dmax){
        kidx = (K-1)-1;
        dcurr = dmax;
        while( dist < dcurr && kidx > 0){
          kidx -= 1;
          dcurr = mins_q[kidx];
        }
        if (kidx != 0){ kidx += 1; }
        else if (dist > dcurr){ kidx += 1; }

        // shift values up
        for (int sidx = K-2; sidx > kidx; --sidx){
          mins_q[sidx] = (float)mins_q[sidx-1];
          argmins_q[sidx] = (int)argmins_q[sidx-1];
        }

        // assign new values
        mins_q[kidx] = inVal;
        argmins_q[kidx] = (int)kidx;

      }          

      
    }
  }
}


void pwd_argmin_forward_cuda(const torch::Tensor pwds,
                             torch::Tensor mins, torch::Tensor argmins){

  // -- unpack --
  int B = pwds.size(0);
  int HD = pwds.size(1);
  int Q = pwds.size(2);
  int nPWD = pwds.size(3);
  int K = mins.size(3);
  // mins.shape = (B,HD,Q,K)
  int _nPWD = K*(K-1)/2;
  assert(nPWD == _nPWD);

  // -- kernel params --
  int q_per_thread = 1;
  int nthreads_q = 1024/(K-1);
  int nthreads_k = K-1;
  int nblocks_q = (Q-1)/(nthreads_q*q_per_thread)+1;
  dim3 nthreads(nthreads_q,nthreads_k);
  dim3 nblocks(_nblocks_q,B,HD);
  // fprintf(stdout,"ps: %d\n",ps);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(pwds.type(),"non_local_search_forward_kernel", ([&] {
        pwd_argmin_kernel<scalar_t><<<nblocks, nthreads>>>(
          pwds.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          mins.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          argmins.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
          q_per_thread);
      }));
}
