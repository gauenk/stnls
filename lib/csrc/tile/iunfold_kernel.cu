


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

/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void stnls_iunfold_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t,5,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor64<scalar_t,7,torch::RestrictPtrTraits> patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int patch_offset,
    bool only_full, bool use_reflect, int qpt, int kpt) {

    // -- shapes --
    int nframes = vid.size(1);
    int colors = vid.size(2);
    int height = vid.size(3);
    int width = vid.size(4);
    int nq = patches.size(1);
    int k = patches.size(2);
    int pt = patches.size(3);
    int ps = patches.size(6);
    int height_width = height*width;
    int fill_pad = dilation * (ps/2);
    int dil = dilation;

    // -- cuda threads --
    int pi = threadIdx.y;
    int pj = threadIdx.z;

    // -- batching --
    int bi = blockIdx.y;
    int query_start_block = blockIdx.x*qpt;
    int k_start = threadIdx.x*kpt;

    // -- unpack --
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
    int n_hw = n_h*n_w;

    // inits
    int qIndex,_qIndex;
    int qi,ki,ti,hi,wi,qi_mod;
    int vi_h,vi_w,vi_t;
    bool valid_hw,valid_t,valid;
    scalar_t pix;
    int stride2 = stride * stride;

    // -- range --
    for(int _qi = 0; _qi < qpt; _qi++){

      // -- query index --
      qi = _qi + query_start_block;
      if (qi >= nq){ continue; }

      for(int _ki = 0; _ki < kpt; _ki++){

        // -- k index --
        ki = k_start + _ki;
        if (ki >= k){ continue; }

        // -- new inds --
        qIndex = qi + start;
        ti = qIndex / (n_hw);
        qi_mod = qIndex % (n_hw);
        hi = (qi_mod / n_w) * stride + top;
        wi = (qi_mod % n_w) * stride + left;

        // -- valid ind --
        valid_hw = (hi >= (top-fill_pad)) && (hi < (btm+fill_pad));
        valid_hw = valid_hw && (wi >= (left-fill_pad)) && (wi < (right+fill_pad));
        valid_hw = valid_hw && (ti   >= 0) && (ti < nframes);

        // -- fill across cuda threads --
        vi_h = hi+dilation*(pi + patch_offset);
        vi_w = wi+dilation*(pj + patch_offset);
        vi_h = use_reflect ? bounds(vi_h,top,btm) : vi_h;
        vi_w = use_reflect ? bounds(vi_w,left,right) : vi_w;

        // -- spatially valid --
        valid_hw = valid_hw && (vi_h >= 0) && (vi_h < height);
        valid_hw = valid_hw && (vi_w >= 0) && (vi_w < width);
        // valid_hw = valid_hw && (vi_h >= top) && (vi_h < btm);
        // valid_hw = valid_hw && (vi_w >= left) && (vi_w < right);


        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- check valid --
          vi_t = bounds(ti + pk,0,nframes);
          valid_t = (vi_t >= 0) && (vi_t < nframes);
          valid = valid_hw && valid_t;

          // -- colors --
          for(int ci = 0; ci < colors; ci++){
            if (valid){
              pix = vid[bi][vi_t][ci][vi_h][vi_w];
            }else{
              pix = 0.;
            }
            patches[bi][qi][ki][pk][ci][pi][pj] = pix;
          }
        }
      }
    }
}

void stnls_cuda_iunfold_forward(
    torch::Tensor vid, torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation,
    int adj, bool only_full, bool use_reflect){

  // -- kernel blocks --
  int batchsize = patches.size(0);
  int numQueries = patches.size(1);
  int k = 1;
  int qpt = 2;
  int nblocks_queries = (numQueries-1)/qpt+1;
  dim3 nblocks(nblocks_queries,batchsize);
  int pt = patches.size(3);
  assert(pt == 1);

  // -- kernel threads --
  int ps = patches.size(6);
  int MAX_THREADS = 960;
  int dim = ps*ps;
  int kpb = MAX_THREADS/dim; // num of "k" managed per block
  int kpt = ((k - 1)/kpb) + 1; // num of "k" per thread
  dim3 nthreads(kpb,ps,ps);
  // fprintf(stdout,"top,left,btm,right: %d,%d,%d,%d\n",top,left,btm,right);

  // -- shared --
  int patch_offset = adj - ps/2;

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "stnls_iunfold_forward_kernel", ([&] {
    stnls_iunfold_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
        patches.packed_accessor64<scalar_t,7,torch::RestrictPtrTraits>(),
        top, left, btm, right, start, stride, dilation,
        patch_offset, only_full, use_reflect, qpt, kpt);
      }));
}


/****************************

       Backward Pass

****************************/

template <typename scalar_t>
__global__ void stnls_iunfold_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> patches,
    int top, int left, int btm, int right, int start, int stride, int dilation,
    int adj, bool only_full, bool use_reflect, int num_kernels) {

    // -- unpack --
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
    bool valid,valid_q,is_edge;
    int dil = dilation;

    // -- boundary --
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

    // -- pick batch --
    int bi = blockIdx.y;

    CUDA_KERNEL_LOOP(_index, num_kernels) {

      int index = (_index);
      const int64_t t_im = (index / sq_hw);
      const int64_t i_mod = index % sq_hw;
      const int64_t w_im = i_mod % sq_w + left;
      const int64_t h_im = (i_mod / sq_w) % sq_h + top;
      // int index = (_index);
      // const int64_t w_im = index % width;
      // const int64_t h_im = (index / width) % height;
      // const int64_t t_im = (index / hw);

      // -- allow partial nhits if edge --
      // int padf = dilation*ps;
      // bool is_edge = (w_im < padf) || (w_im > (width-padf));
      // is_edge = is_edge || (h_im < padf) || (h_im > (height-padf));
        
      for(int ci = 0; ci < colors; ci++){
        // nhits = 0;
        // nhits_q = 0;
        scalar_t val = 0;
        for (int pk = 0; pk < pt; pk++){
          for (int pi = 0; pi < ps; pi++){
            for (int pj = 0; pj < ps; pj++){

              // -- offsets for ni --
              int _wi = w_im + dilation*(pi - psOffset - adj);
              int _hi = h_im + dilation*(pj - psOffset - adj);
              // int _wi = w_im + dilation*(pi - psHalf);
              // int _hi = h_im + dilation*(pj - psHalf);
              int ti = t_im + pk;

              // -- check bounds --
              valid = (_wi >= left) && (_wi < (right_bnd));
              valid = valid && (_hi >= top) && (_hi < (btm_bnd));

              // valid = (_wi >= (left-fill_pad)) && (_wi < (right+fill_pad));
              // valid = valid && (_hi >= (top-fill_pad)) && (_hi < (btm+fill_pad));

              // valid = (_wi >= -fill_pad) && (_wi < (width+fill_pad));
              // valid = valid && (_hi >= -fill_pad) && (_hi < (height+fill_pad));
              // int wi = bounds(_wi,left,right);
              // int hi = bounds(_hi,top,btm);
              int wi = _wi;
              int hi = _hi;
              // int wi = bounds(_wi,left,right);
              // int hi = bounds(_hi,top,btm);
              // int wi = bounds(_wi,0,width);
              // int hi = bounds(_hi,0,height);

              // -- compute ni --
              // int qi = ti * n_hw + ((hi/stride) * n_W)+ (wi/stride);
              // qi -= start;
              int qi = ti * n_hw;
              qi += (((hi-top)/stride) * n_w);
              qi += ((wi-left)/stride);
              qi -= start;

              // -- only if qi is aligned with center --
              valid = valid && ((hi-top) % stride == 0) && ((wi-left) % stride == 0);

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
                val += patches[bi][qi][0][0][ci][h_ip][w_ip];
                // nhits_q += 1;
              }
              // if(valid){
              //   nhits += 1;
              // }
            }
          } // for patch size
        } // for patch size
        // bool eq_hits = nhits == nhits_q;
        // bool hit_req = true;//((not is_edge) && (nhits == ndim)) || is_edge;
        // if (eq_hits){
        //   vid[t_im][ci][h_im][w_im] =  val;
        // }
        vid[bi][t_im][ci][h_im][w_im] = val;
        // vid[t_im][ci][h_im][w_im] += val;
      } // for colors
    }
}

void stnls_cuda_iunfold_backward(
  torch::Tensor grad_vid,torch::Tensor patches,
  int top, int left, int btm, int right,
  int start, int stride, int dilation,
  int adj, bool only_full, bool use_reflect) {

  // -- kernel blocks --
  // int numQueries = patches.size(0);
  // int k = 1;
  int batchsize = grad_vid.size(0);
  int nframes = grad_vid.size(1);
  int height = grad_vid.size(3);
  int width = grad_vid.size(4);

  // make square
  int sq_h = btm - top;
  int sq_w = right - left;
  int sq_hw = sq_h * sq_w;


  // threads and blocks
  int nthreads = 512;
  // int num_kernels = patches.size(0);//nframes*height*width;
  // int num_kernels = nframes*height*width;
  int num_kernels = nframes*sq_hw;
  int nblocks_pix = (num_kernels-1) / nthreads+1;
  dim3 nblocks(nblocks_pix,batchsize);

  // get starting pixel

  // -- kernel threads --
  // int ps = patches.size(5);
  // int MAX_THREADS = 1024;
  // int dim = ps*ps;
  // int kpb = MAX_THREADS/dim; // num of "k" managed per block
  // int kpt = ((k - 1)/kpb) + 1; // num of "k" per thread
  // dim3 nthreads(kpb,ps,ps);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(patches.type(), "stnls_iunfold_backward_kernel", ([&] {
    stnls_iunfold_backward_kernel<scalar_t>
      <<<nblocks, nthreads>>>(
        grad_vid.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
        top, left, btm, right, start, stride, dilation,
        adj, only_full, use_reflect, num_kernels);
  }));

}
