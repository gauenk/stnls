
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
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> patches,
    int top, int left, int btm, int right, 
    int start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect, int num_kernels) {

    // -- unpack --
    int bsize = vid.size(0);
    int nframes = vid.size(1);
    int colors = vid.size(2);
    int height = vid.size(3);
    int width = vid.size(4);
    int pt = patches.size(3);
    int ps = patches.size(6);
    int numQueries = patches.size(1);
    int psOffset = (ps-1)/2;
    int psHalf = ps/2;
    int hw = height*width;
    int fill_pad = psHalf * dilation;
    int dil = dilation;
    // int width_s = width/stride;
    // int hw_s = (height/stride)*(width/stride);
    // int num_kernels = inds.size(0);
    bool valid,valid_q;
    // bool is_edge;
    // int nhits,nhits_q;
    // int ndim = ps*ps*pt;

    // -- coords with pads --
    int pad = dilation*(ps/2);
    pad = (adj > 0) ? 0 : pad; // suspect line.
    int top_p = std::max(top-pad,0);
    int left_p = std::max(left-pad,0);
    int btm_p = std::min(btm+pad,height);
    int right_p = std::min(right+pad,width);

    // -- get batch index --
    int bindex = blockIdx.y;
  
    // coords
    int sq_hp = btm_p - top_p;
    int sq_wp = right_p - left_p;
    int sq_hwp = sq_hp * sq_wp;

    // -- adjust endpoint for "adj" --
    // no spilling over right-hand boundary
    int right_a = right - (ps-1)*dil;
    int btm_a = btm - (ps-1)*dil;
    int right_bnd = (only_full) ? right_a : right;
    int btm_bnd = (only_full) ? btm_a : btm;
  
    // -- make square --
    int sq_h = btm - top;
    int sq_w = right - left;
    int sq_hw = sq_h * sq_w;

    // -- strided size --
    int n_h = int((sq_h-1) / stride) + 1;
    int n_w = int((sq_w-1) / stride) + 1;
    if (only_full){
      n_h = (sq_h - (ps-1)*dil - 1)/stride + 1;
      n_w = (sq_w - (ps-1)*dil - 1)/stride + 1;
    }
    int n_hw = n_h * n_w;

    CUDA_KERNEL_LOOP(_index, num_kernels) {

      // index to pixel location
      int index = (_index);
      const int64_t t_im = (index / sq_hwp);
      const int64_t i_mod = index % sq_hwp;
      const int64_t w_im = (i_mod % sq_wp) + left_p;
      const int64_t h_im = ((i_mod / sq_wp) % sq_hp) + top_p;

      // Which patches (qi) impact me (t_im,w_im,h_im)?
      for(int ci = 0; ci < colors; ci++){
        scalar_t val = 0;
        for (int pk = 0; pk < pt; pk++){
          for (int pi = 0; pi < ps; pi++){
            for (int pj = 0; pj < ps; pj++){

              // -- offsets for ni --
              // use "psOffset" instead of "psHalf" because of reflection.
              int _wi = w_im + dilation*(pi - psOffset - adj);
              int _hi = h_im + dilation*(pj - psOffset - adj);
              int ti = t_im + pk;

              // -- check bounds (we need the patch for the pixel!) --
              valid = (_wi >= left) && (_wi < right_bnd);
              valid = valid && (_hi >= top) && (_hi < btm_bnd);
              int wi = use_reflect ? bounds(_wi,left,right) : _wi;
              int hi = use_reflect ? bounds(_hi,top,btm) : _hi;

              // -- only if proposed index is aligned with stride --
              valid = valid && ((hi-top) % stride == 0) && ((wi-left) % stride == 0);

              // -- compute ni --
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
                val += patches[bindex][qi][0][0][ci][h_ip][w_ip];
              }

            }
          } // for patch size
        } // for patch size
        vid[bindex][t_im][ci][h_im][w_im] = val;
      } // for colors
    } // for each pixel (with stride)
}

void dnls_cuda_ifold_forward(
    torch::Tensor vid, torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect){

  // batching entire image always
  int bsize = vid.size(0);
  int nframes = vid.size(1);
  int colors = vid.size(2);
  int height = vid.size(3);
  int width = vid.size(4);
  int ps = patches.size(6);

  // -- coords with pads --
  int pad = dilation*(ps/2);
  pad = (adj > 0) ? 0 : pad;
  int top_p = std::max(top-pad,0);
  int left_p = std::max(left-pad,0);
  int btm_p = std::min(btm+pad,height);
  int right_p = std::min(right+pad,width);

  // coords
  int sq_hp = btm_p - top_p;
  int sq_wp = right_p - left_p;
  int sq_hwp = sq_hp * sq_wp;

  // launch params
  int nthreads = 512;
  // int num_kernels = nframes*sq_hw;
  int num_kernels = nframes*sq_hwp;
  int nblocks_queries = (num_kernels-1) / nthreads+1;
  dim3 nblocks(nblocks_queries,bsize);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_ifold_forward_kernel", ([&] {
    dnls_ifold_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        top,left,btm,right,start,stride,dilation,adj,only_full,use_reflect,num_kernels);
      }));
}

/**************************************

       Backward Pass (with Coords)

**************************************/

template <typename scalar_t>
__global__ void dnls_ifold_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid, // grad
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> patches,
    int top, int left, int btm, int right, int start, int stride,
    int dilation, int adj, bool only_full, bool use_reflect, int qpt, int kpt) {

    // -- shapes --
    int bsize = vid.size(0);
    int nframes = vid.size(1);
    int colors = vid.size(2);
    int height = vid.size(3);
    int width = vid.size(4);
    int nq = patches.size(1);
    int k = patches.size(2);
    int pt = patches.size(3);
    int ps = patches.size(5);
    int psHalf = (int)ps/2;
    int psOffset = (int)(ps-1)/2; // convention to decided center
    int height_width = height*width;
    int hw = height*width;
    int dil = dilation;

    // -- cuda threads --
    int pi = threadIdx.y;
    int pj = threadIdx.z;

    // -- batching --
    int query_start_block = blockIdx.x*qpt;
    int k_start = threadIdx.x*kpt;
    int bindex = blockIdx.y;

    // -- only fully contained patches count --
    int right_a = right - (ps-1)*dil;
    int btm_a = btm - (ps-1)*dil;
    int right_bnd = (adj > 0) ? right_a : right;
    int btm_bnd = (adj > 0) ? btm_a : btm;

    // -- unpack --
    int sq_h = btm - top;
    int sq_w = right - left;
    int sq_hw = sq_h * sq_w;

    // -- strided size --
    int n_h = int((sq_h-1) / stride) + 1;
    int n_w = int((sq_w-1) / stride) + 1;
    if (adj > 0){
      n_h = (sq_h - (ps-1)*dil - 1)/stride + 1;
      n_w = (sq_w - (ps-1)*dil - 1)/stride + 1;
    }
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
        valid_hw = (hi >= top) && (hi < btm_bnd);
        valid_hw = valid_hw && (wi >= left) && (wi < right_bnd);
        valid_hw = valid_hw && (ti   >= 0) && (ti < nframes);

        // -- fill across cuda threads --
        // vi_h = bounds(hi+dilation*(pi - psHalf),height);
        // vi_w = bounds(wi+dilation*(pj - psHalf),width);
        vi_h = hi+dilation*(pi - psHalf + adj);
        vi_w = wi+dilation*(pj - psHalf + adj);

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
              pix = vid[bindex][vi_t][ci][vi_h][vi_w];
            }else{
              pix = 0.;
            }
            patches[bindex][qi][ki][pk][ci][pi][pj] = pix;
          }
        }
      }
    }
}

void dnls_cuda_ifold_backward(
  torch::Tensor grad_vid,
  torch::Tensor patches,
  int top, int left, int btm, int right,
  int start, int stride, int dilation, int adj,
  bool only_full, bool use_reflect) {

  // patches.shape
  // = [batch size, num queries, neighbors, patch_t, channels, patch_h, patch_w]

  // -- unpack --
  int bsize = patches.size(0);
  int numQueries = patches.size(1);
  int pt = patches.size(3);
  int ps = patches.size(6);

  // -- kernel blocks --
  int k = 1;
  int qpt = 10;
  int nblocks_queries = (numQueries-1)/qpt+1;
  assert(pt == 1);
  dim3 nblocks(nblocks_queries,bsize);

  // -- kernel threads --
  int MAX_THREADS = 1024;
  int dim = ps*ps;
  int kpb = MAX_THREADS/dim; // num of "k" managed per block
  int kpt = ((k - 1)/kpb) + 1; // num of "k" per thread
  dim3 nthreads(kpb,ps,ps);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "dnls_ifold_backward_kernel", ([&] {
    dnls_ifold_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        grad_vid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        top, left, btm, right, start, stride, dilation, adj,
        only_full, use_reflect, qpt, kpt);
  }));

}
