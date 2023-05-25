// Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

// This file is part of the implementation as described in the NIPS 2018 paper:
// Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
// Please see the file LICENSE.txt for the license governing this code.

#include <math.h>
#include "stdio.h"
#include "iostream"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <ATen/ATen.h>

using namespace std;

const int N_THREADS_M = 256;
const int N_THREADS_O = 1024 / N_THREADS_M;
const int N_THREADS_N = 256;
const int N_THREADS_E = 1024 / N_THREADS_N;


// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//             Forward
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

__global__
void matmul1_fwd_kernel(float *mat_x, float *mat_y, long *mat_i,
                        float *mat_o, int m, int n, int e, int o, int batch_size){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.z *blockDim.z + threadIdx.z;

	float sum = 0;

	if (batch >= batch_size || row >= m || col >= o){
	  return;
	}

	// Fetch x indices
	int pos_i = (batch * m * o) + (row * o) + col;
	int xind_col = mat_i[pos_i];

	// Mat mult
	for (int i = 0; i < e; i++) {
	    int pos_y = (batch * m * e) + (row * e + i);
	    int pos_x = (batch * n * e) + (xind_col * e + i);
	    sum += mat_y[pos_y] * mat_x[pos_x];
	}

	int pos = (batch * m * o) + (row * o + col);
	mat_o[pos] = sum;	 
}

void matmul1_fwd_cuda(at::Tensor mat_x, at::Tensor mat_y, at::Tensor mat_i,
                      at::Tensor out, int n, int m, int e, int o, int b) {
		// Set array and CUDA block/grid sizes


  dim3 block(N_THREADS_O, N_THREADS_M, 1);
  dim3 grid((int)ceil(((float)o)/N_THREADS_O), (int)ceil(((float)m)/N_THREADS_M), b);
  // fprintf(stdout,"m,n,e,o,b: %d,%d,%d,%d,%d\n",m,n,e,o,b);
		
  // Call kernel
  matmul1_fwd_kernel<<<grid, block>>>(mat_x.data_ptr<float>(),
				      mat_y.data_ptr<float>(),
				      mat_i.data_ptr<long>(),
				      out.data_ptr<float>(), m, n, e, o, b);
  return;
}


// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//             Backward
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

// #include <math.h>
// #include <vector>
// #include "stdio.h"
// #include "iostream"
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <ATen/ATen.h>
// #include <chrono>

// using namespace std;

__device__
void matmul1_xgrad(float *grad, float *mat_y, long *mat_i, float *mat_ox,
                   int m, int n, int e, int o, int batch_size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z *blockDim.z + threadIdx.z;


    if (batch >= batch_size || row >= m || col >= o)
        return;

	int pos_i = (batch  * m * o) + (row * o) + col;	
	int idx = mat_i[pos_i];
	float g = grad[pos_i];

	for (int j = 0; j < e; j++) {
		int pos_y = (batch * m * e) + (row * e) + j;
	    int pos_ox = (batch * n * e) + (idx * e) + j;
	    atomicAdd(mat_ox + pos_ox, mat_y[pos_y] * g);
	}
}

__device__
void matmul1_ygrad(float *grad, float *mat_x, long *mat_i, float *mat_o,
                   int m, int n, int e, int o, int batch_size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z *blockDim.z + threadIdx.z;


    if (batch >= batch_size || row >= m || col >= e)
        return;

    float sum = 0.0;

    for (int i = 0; i < o; i++) {

      int pos_i = (batch * m * o) + (row * o) + i;
      int xind = mat_i[pos_i];
      // int pos_g = (batch * m * o) + (row * o) + i;
      float g = grad[pos_i];

      int pos_x = (batch * n * e) + (xind * e) + col;
      sum = sum + (mat_x[pos_x] * g);
    }
    int pos_o = (batch * m * e) + (row * e) + col;
    mat_o[pos_o] = sum;
}


__global__
void matmul1_bwd_kernel_xgrad(float *gradients, float *mat_x, float *mat_y, long *mat_i, float *mat_ox, int m, int n,  int e, int o, int batch_size){
		matmul1_xgrad(gradients, mat_y, mat_i, mat_ox, m, n, e, o, batch_size);
}


__global__
void matmul1_bwd_kernel_ygrad(float *gradients, float *mat_x, float *mat_y, long *mat_i, float *mat_oy, int m, int n,  int e, int o, int batch_size){
    matmul1_ygrad(gradients, mat_x, mat_i, mat_oy, m, n, e, o, batch_size);
}

void matmul1_bwd_cuda(at::Tensor gradients, at::Tensor mat_x, at::Tensor mat_y, at::Tensor mat_i, at::Tensor out_x, at::Tensor out_y, int m, int n, int e, int o, int b){
	// Set array and CUDA block/grid sizes

	dim3 block(N_THREADS_E, N_THREADS_N, 1);
	dim3 grid((int)ceil(((float)e)/N_THREADS_E), (int)ceil(((float)std::max(n, m))/N_THREADS_N), b);

	// Call kernel
	matmul1_bwd_kernel_ygrad<<<grid, block>>>(gradients.data_ptr<float>(), mat_x.data_ptr<float>(), mat_y.data_ptr<float>(), mat_i.data_ptr<long>(), out_y.data_ptr<float>(), m, n, e, o, b);


	dim3 block_xgrad(N_THREADS_E, N_THREADS_N, 1);
	dim3 grid_xgrad((int)ceil(((float)e)/N_THREADS_E), (int)ceil(((float)std::max(n, m))/N_THREADS_N), b);

	// Call kernel
	matmul1_bwd_kernel_xgrad<<<grid_xgrad, block_xgrad>>>(gradients.data_ptr<float>(), mat_x.data_ptr<float>(), mat_y.data_ptr<float>(), mat_i.data_ptr<long>(), out_x.data_ptr<float>(), m, n, e, o, b);

	return;
}

