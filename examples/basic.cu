#include <cuda.h>

template<typename T>
__global__ sum(T* arr) {
  __shared__ answer[BLOCK_SIZE];

  int nthreads = blockDim.x;

  while(nthreads > 1) {
    int half_point = nthreads >> 1;

    if (threadIdx.x < half_point) {
      thread2 = threadIdx.x + half_point;

      temp = answer[thread2];
      answer[threadIdx.x] += temp;
    }
    __syncthreads();

    nthreads = half_point;
  }
}
