//pass
//--blockDim=[512,512] --gridDim=1
// https://github.com/TianhuaTao/Megatron-DeepSpeed/commit/5478d67ef2048481e651a05053487fba029c3210
// SOURCE: https://github.com/NVIDIA/Megatron-LM/commit/5478d67ef2048481e651a05053487fba029c3210

template<typename T, typename U, typename V> __global__
void cuComputeGradInput(
    const V* __restrict__ dout,
    const T* __restrict__ input,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    const V* gamma,
    T* grad_input,
    int has_gamma)
{
  __requires(n2 == 1024);
  extern __shared__ U buf[];
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const U c_mean = mean[i1];
    const U c_invvar = invvar[i1];
    const T* k_input = input + i1*n2;
    const V* k_dout = dout + i1*n2;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (has_gamma) {
      int l = 4*thrx;
      for (;  l < n2 - 3;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_input[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          sum_loss1 += c_loss * gamma[l+k];
          sum_loss2 += c_loss * gamma[l+k] * (c_h - c_mean) * c_invvar;
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss * gamma[l];
        sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
      }
    } else {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_input[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          sum_loss1 += c_loss;
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss;
        sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
      sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      //U* buf = shared.getPointer(); 
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2*wrt_i] = sum_loss1;
          buf[2*wrt_i+1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2*read_i];
          sum_loss2 += buf[2*read_i+1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2*threadIdx.x] = sum_loss1;
        buf[2*threadIdx.x+1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        sum_loss1 = buf[2*threadIdx.x];
        sum_loss2 = buf[2*threadIdx.x+1];
      } 
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T* k_grad_input = grad_input + i1*n2;
    if (has_gamma) {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss * gamma[l];
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads(); // <- BUG
  }
}
