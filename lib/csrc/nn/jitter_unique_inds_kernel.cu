
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>

__global__ void jitter_unique_inds_kernel(
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> inds,
    int K, int tgt_K, int H, int W, int sqrt_K, int sqrt_K2, int q_per_thread){

  // -- shared memory --
  extern __shared__ bool s[];
  bool* repl = (bool*)&s[0]; // size (K,)
  bool* avail = (bool *)(&s[K]); // size (sqrt_K,sqrt_K)

  // -- alloc --
  int qi;
  int Q = inds.size(1);
  bool check;
  int deltas[2];
  int refs[2];
  int a_index;

  // -- cuda threads --
  int bi = blockIdx.y;
  int qi_thread = q_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);

  // -- for each location --
  for (int qi_ix = 0; qi_ix < q_per_thread; qi_ix++){

    // -- current query --
    qi = qi_thread + qi_ix;
    if (qi >= Q){ continue; }

    // -- init --
    for(int i = 0; i < sqrt_K; i++){
      for(int j = 0; j < sqrt_K; j++){
        avail[i+sqrt_K*j] = (i != sqrt_K2) && (j != sqrt_K2);
      }
    }
    for(int i = 0; i < K; i++){
      repl[i] = 0;
    }
    for(int i = 0; i < 2; i++){
      refs[i] = inds[bi][qi][0][i+1];
    }

    // -- for each neighbor --
    for(int i = 0; i < K; i++){

      // -- mark "not available" if within radius --
      if(i > 0){
        check = true;
        for(int l = 0; l < 2; l++){
          deltas[l] = refs[l] - inds[bi][qi][i][l+1];
          check = check & (abs(deltas[l]) < sqrt_K2);
        }
        if (check){
          a_index = deltas[0]+sqrt_K2;
          a_index += sqrt_K*(deltas[1]+sqrt_K2);
          avail[a_index] = 0;
        }
      }

      // -- mark as "duplicate" if repeated --
      for(int j = i+1; j < K; j++){
        check = true;
        for(int l = 1; l < 3; l++){
          check = check & (inds[bi][qi][i][l] == inds[bi][qi][j][l]);
        }
        if (check){
          repl[j] = 1;
        }
      }

    }

    // -- replaced marked neighbors --
    // index in a spiral; [credit: stackoverflow.com #3706219]

    int delta_i = 0;
    int delta_j = -1;
    int avail_i = 0;
    int avail_j = 0;
    a_index = sqrt_K2 + sqrt_K * sqrt_K2;

    for (int i = 0; i < K; i++){
      if (repl[i] == 0){continue;}
      
      // -- loop in a sprial until we find an open spot --
      while (avail[a_index] == 0){
        check = (avail_i == avail_j);
        check = check || ((avail_i < 0) && (avail_i == -avail_j));
        check = check || ((avail_i > 0) && (avail_i == (1-avail_j)));
        if (check){
          delta_i = -delta_j;
          delta_j = delta_j;
        }
        avail_i += delta_i;
        avail_j += delta_j;
        a_index = avail_i + sqrt_K2;
        a_index += sqrt_K*(avail_j + sqrt_K2);
      }

      // -- fill --
      inds[bi][qi][i][1] = refs[0] + avail_i;
      inds[bi][qi][i][2] = refs[1] + avail_j;

    }

  }
}


void jitter_unique_inds_cuda(
     torch::Tensor inds,
     int tgt_K, int H, int W){
  
  // -- unpack --
  int B = inds.size(0);
  int Q = inds.size(1);
  int K = inds.size(2);
  int sqrt_K = std::sqrt(tgt_K);
  int sqrt_K2 = sqrt_K/2;

  // -- num 2 run --
  int nRun = Q;

  // -- kernel params --
  int q_per_thread = 1;
  int _nthreads = 256;
  dim3 nthreads(_nthreads);
  int _nblocks = (nRun-1)/(_nthreads*q_per_thread)+1;
  dim3 nblocks(_nblocks,B);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);

  // -- shared memory size --
  int SMEM = _nthreads * K * sqrt_K * sqrt_K;

  // -- launch kernel --
  jitter_unique_inds_kernel<<<nblocks, nthreads, SMEM>>>(
       inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
       K,tgt_K,H,W,sqrt_K,sqrt_K2,q_per_thread);

}
