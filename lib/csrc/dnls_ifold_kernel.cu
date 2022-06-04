
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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

       Forward Pass (with Inds)

**************************************/


template <typename scalar_t>
__global__ void dnls_ifold_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int num_kernels) {

    // -- unpack --
    int nframes = vid.size(0);
    int colors = vid.size(1);
    int height = vid.size(2);
    int width = vid.size(3);
    int pt = patches.size(2);
    int ps = patches.size(5);
    int numQueries = patches.size(0);
    int psHalf = ps/2;
    int hw = height*width;
    int fill_pad = psHalf * dilation;
    // int width_s = width/stride;
    // int hw_s = (height/stride)*(width/stride);
    // int num_kernels = inds.size(0);
    bool valid,valid_q;
    // bool is_edge;
    // int nhits,nhits_q;
    // int ndim = ps*ps*pt;

    // -- make square --
    int sq_h = btm - top;
    int sq_w = right - left;
    int sq_hw = sq_h * sq_w;

    // -- strided size --
    int n_h = int((sq_h-1) / stride) + 1;
    int n_w = int((sq_w-1) / stride) + 1;
    int n_hw = n_h * n_w;

    CUDA_KERNEL_LOOP(_index, num_kernels) {

      int index = (_index);
      const int64_t t_im = (index / sq_hw);
      const int64_t i_mod = index % sq_hw;
      const int64_t w_im = (i_mod % sq_w) + left;
      const int64_t h_im = ((i_mod / sq_w) % sq_h) + top;

      // const int64_t t_im = (index / n_hw);
      // const int64_t w_im = (index % n_w) + left;
      // const int64_t h_im = ((index / n_w) % n_h) + top;


      // const int64_t t_im = (index / n_hw);
      // int i_mod = index % (n_hw);
      // const int64_t h_im = (stride*(i_mod / n_w) + top);// % height;
      // const int64_t w_im = (stride*(i_mod % n_w) + left);// % width;

      // -- allow partial nhits if edge --
      // int padf = dilation*ps;
      // bool is_edge = (w_im < padf) || (w_im > (width-padf));
      // is_edge = is_edge || (h_im < padf) || (h_im > (height-padf));

      for(int ci = 0; ci < colors; ci++){
        scalar_t val = 0;
        for (int pk = 0; pk < pt; pk++){
          for (int pi = 0; pi < ps; pi++){
            for (int pj = 0; pj < ps; pj++){

              // -- offsets for ni --
              int _wi = w_im + dilation*(pi - psHalf);
              int _hi = h_im + dilation*(pj - psHalf);
              int ti = t_im + pk;

              // -- check bounds --
              // NOTE; this will not work for dilation > 1
              // valid = (_wi >= -fill_pad) && (_wi < (width+fill_pad));
              // valid = valid && (_hi >= -fill_pad) && (_hi < (height+fill_pad));
              valid = (_wi >= (left-fill_pad)) && (_wi < (right+fill_pad));
              valid = valid && (_hi >= (top-fill_pad)) && (_hi < (btm+fill_pad));
              // valid = (_wi >= (left-fill_pad)) && (_wi < (right+fill_pad));
              // valid = valid && (_hi >= (top-fill_pad)) && (_hi < (btm+fill_pad));
              int wi = bounds(_wi,left,right);
              int hi = bounds(_hi,top,btm);

              // -- only if proposed index is aligned with stride --
              valid = valid && ((hi-top) % stride == 0) && ((wi-left) % stride == 0);

              // -- compute ni --
              // int qi = ti * hw + hi * width + wi; // maybe stride here?
              // qi = (int) (qi / (1.*stride));
              // qi -= start;
              int qi = ti * n_hw;
              qi += (((hi-top)/stride) * n_w);
              qi += ((wi-left)/stride);
              qi -= start;

              // -- patch indexing --
              int w_ip = ps-1-pi;
              int h_ip = ps-1-pj;

              // -- reflect to match --
              if (_wi > wi){
                w_ip = pi;
                valid = valid && (w_ip < psHalf);
              }
              else if(_wi < wi){
                w_ip = pi;
                valid = valid && (w_ip > psHalf);
              }

              if (_hi > hi){
                h_ip = pj;
                valid = valid && (h_ip < psHalf);
              }
              else if(_hi < hi){
                h_ip = pj;
                valid = valid && (h_ip > psHalf);
              }

              // -- accumulate --
              valid_q = valid && (qi >= 0) && (qi < numQueries);
              if (valid_q){
                val += patches[qi][0][0][ci][h_ip][w_ip];
              }

            }
          } // for patch size
        } // for patch size
        vid[t_im][ci][h_im][w_im] = val;
      } // for colors
    } // for each pixel (with stride)
}

void dnls_cuda_ifold_forward(
    torch::Tensor vid, torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation){

  // batching entire image always
  int nframes = vid.size(0);
  int colors = vid.size(1);
  int height = vid.size(2);
  int width = vid.size(3);

  // coords
  int sq_h = btm - top;
  int sq_w = right - left;
  int sq_hw = sq_h * sq_w;

  // launch params
  int nthreads = 512;
  int num_kernels = nframes*sq_hw;
  int nblocks = (num_kernels-1) / nthreads+1;

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_ifold_forward_kernel", ([&] {
    dnls_ifold_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        top,left,btm,right,start,stride,dilation,num_kernels);
      }));
}

/**************************************

       Backward Pass (with Coords)

**************************************/

template <typename scalar_t>
__global__ void dnls_ifold_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> vid, // grad
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int qpt, int kpt) {

    // -- shapes --
    int nframes = vid.size(0);
    int colors = vid.size(1);
    int height = vid.size(2);
    int width = vid.size(3);
    int nq = patches.size(0);
    int k = patches.size(1);
    int pt = patches.size(2);
    int ps = patches.size(4);
    int psHalf = (int)ps/2;
    int height_width = height*width;
    int hw = height*width;

    // -- cuda threads --
    int pi = threadIdx.y;
    int pj = threadIdx.z;

    // -- batching --
    int query_start_block = blockIdx.x*qpt;
    int k_start = threadIdx.x*kpt;

    // -- unpack --
    int sq_h = btm - top;
    int sq_w = right - left;
    int sq_hw = sq_h * sq_w;

    // -- strided size --
    int n_h = int((sq_h-1) / stride) + 1;
    int n_w = int((sq_w-1) / stride) + 1;
    int n_hw = n_h*n_w;

    // inits
    int qIndex,_qIndex;
    int qi,ki,ti,hi,wi,_qi,qi_mod;
    int vi_h,vi_w,vi_t;
    bool valid_hw,valid_t,valid;
    scalar_t pix;
    int stride2 = stride*stride;

    // -- range --
    for(int _qi = 0; _qi < qpt; _qi++){

      // -- query index --
      qi = _qi + query_start_block;
      if (qi >= nq){ continue; }

      for(int _ki = 0; _ki < kpt; _ki++){

        // -- k index --
        ki = k_start + _ki;
        if (ki >= k){ continue; }

        // -- indices --
        // qIndex = stride2*(qi + start);
        // ti = (qIndex/hw);
        // _qIndex = qIndex % hw;
        // hi = (stride)*(_qIndex / (stride*width)) % height;
        // wi = (_qIndex/stride) % width;
        // hi = (_qIndex/width) % height;
        // wi = _qIndex % width;

        // -- new inds --
        qIndex = qi + start;
        ti = qIndex / (n_hw);
        qi_mod = qIndex % (n_hw);
        hi = (qi_mod / n_w) * stride + top;
        wi = (qi_mod % n_w) * stride + left;

        // -- valid ind --
        valid_hw = (hi >= top) && (hi < btm);
        valid_hw = valid_hw && (wi >= left) && (wi < right);
        valid_hw = valid_hw && (ti   >= 0) && (ti < nframes);

        // -- fill across cuda threads --
        // vi_h = bounds(hi+dilation*(pi - psHalf),height);
        // vi_w = bounds(wi+dilation*(pj - psHalf),width);
        vi_h = hi+dilation*(pi - psHalf);
        vi_w = wi+dilation*(pj - psHalf);

        // -- spatially valid --
        valid_hw = valid_hw && (vi_h >= top) && (vi_h < btm);
        valid_hw = valid_hw && (vi_w >= left) && (vi_w < right);
        // valid_hw = valid_hw && (vi_h >= 0) && (vi_h < height);
        // valid_hw = valid_hw && (vi_w >= 0) && (vi_w < width);
        // valid_hw = valid_hw && (ti   >= 0) && (ti < nframes);
        // valid_hw = (vi_h >= left) && (vi_h < right);
        // valid_hw = valid_hw && (vi_w >= top) && (vi_w < btm);
        // valid_hw = valid_hw && (ti   >= 0) && (ti < nframes);

        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- check valid --
          vi_t = bounds(ti + pk,0,nframes);
          valid_t = (vi_t >= 0) && (vi_t < nframes);
          valid = valid_hw && valid_t;

          // -- colors --
          for(int ci = 0; ci < colors; ci++){
            if (valid){
              pix = vid[vi_t][ci][vi_h][vi_w];
            }else{
              pix = 0.;
            }
            patches[qi][ki][pk][ci][pi][pj] = pix;
          }
        }
      }
    }
}

void dnls_cuda_ifold_backward(
  torch::Tensor grad_vid,
  torch::Tensor patches,
  int top, int left, int btm, int right,
  int start, int stride, int dilation) {

  // -- kernel blocks --
  int numQueries = patches.size(0);
  int k = 1;
  int qpt = 10;
  int nblocks = (numQueries-1)/qpt+1;
  int pt = patches.size(2);
  assert(pt == 1);

  // -- kernel threads --
  int ps = patches.size(5);
  int MAX_THREADS = 1024;
  int dim = ps*ps;
  int kpb = MAX_THREADS/dim; // num of "k" managed per block
  int kpt = ((k - 1)/kpb) + 1; // num of "k" per thread
  dim3 nthreads(kpb,ps,ps);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_ifold_backward_kernel", ([&] {
    dnls_ifold_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        grad_vid.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        top, left, btm, right, start, stride, dilation, qpt, kpt);
  }));

}
