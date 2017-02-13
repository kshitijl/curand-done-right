#include <curand-done-right/curanddr.hxx>

// from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

const int block_size = 1024;

__global__ void estimate_pi(uint* scratch, unsigned long size, uint seed) {
  /* Later we will be reducing in shared memory, during which we will
   * read ALL indices in the declared shared memory block, not just
   * those for which index < size. I did it this way to avoid a branch
   * in that loop. So I initialize all locations with sentinel 0
   * values. */
  long tid = threadIdx.x;
  __shared__ uint answer[block_size];
  answer[tid] = 0;
  
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size) {
    /* Generate two uniformly distributed random numbers in [0,1]: a
     * point in the square [0,1]x[0,1]. */
    auto randoms = curanddr::uniforms<2>(uint3{0,0,0},
                                         uint2{index, seed});

    /* Does it lie within the quarter of the unit circle that falls
     * within the rectangle we're sampling, [0,1]x[0,1]? */
    float xx = randoms[0], yy = randoms[1];
    int integrand = 0;
    if(xx*xx + yy*yy < 1)
      integrand = 1;
    
    int nthreads = blockDim.x;

    answer[tid] = integrand;

    /* We will reduce the integrands within this block in shared
     * memory. Barrier to wait for all threads to finish writing their
     * answers to shared memory. */
    __syncthreads();

    /* A standard log-tree reduction. */
    while(nthreads > 1) {
      int half_point = nthreads >> 1;

      if (threadIdx.x < half_point) {
        int thread2 = threadIdx.x + half_point;

        uint temp = answer[thread2];
        answer[threadIdx.x] += temp;
      }
      __syncthreads();

      nthreads = half_point;
    }

    /* Write the answer for this block into global memory. We will do
     * a final reduction over all blocks on the host. */
    if(tid == 0) {
      scratch[blockIdx.x] = answer[0];
    }
  }
}

int main(int argc, char **argv) {
  unsigned long size = 10;
  uint seed = 1;
  if(argc > 1)
    sscanf(argv[1], "%lu", &size);

  uint *d_output, *h_output;

  int n_blocks = (size + block_size - 1)/block_size;

  /* Allocate space for the computed sum for each block. */
  gpuErrchk(cudaMalloc(&d_output, n_blocks*sizeof(uint)));
  h_output = (uint*)malloc(n_blocks*sizeof(uint));

  /* Launch CTA to count samples within quarter-circle for each
   * block. */
  estimate_pi<<<n_blocks,block_size>>>(d_output, size, seed);
  gpuErrchk( cudaPeekAtLastError() );

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_output, d_output, n_blocks*sizeof(uint),
                       cudaMemcpyDeviceToHost));

  long final_result = 0;
  for(int ii = 0; ii < n_blocks; ++ii)
    final_result += h_output[ii];

  float estimate = 4 * (float)final_result/size;
  printf("%ld %u\n", final_result, h_output[n_blocks-1]);  
  printf("%f\n", estimate);
  
  return 0;
}
