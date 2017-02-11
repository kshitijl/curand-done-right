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

const int block_size = 32;

__constant__ uint d_seed;
__global__ void estimate_pi(uint* scratch, int size) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size) {
    /* Generate two uniformly distributed random numbers in [0,1]: a
     * point in the square [0,1]x[0,1]. */
    auto randoms = curanddr::uniforms<2>(uint3{0,0,0},
                                         uint2{index, d_seed});
    float xx = randoms[0], yy = randoms[1];
    int integrand = 0;
    if(xx*xx + yy*yy < 1)
      integrand = 1;
    
    int tid = threadIdx.x;

    __shared__ uint answer[block_size];
    int nthreads = blockDim.x;

    answer[tid] = integrand;
    __syncthreads();

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

    if(tid == 0)
      scratch[blockIdx.x] = answer[0];
  }
}

int main(int argc, char **argv) {
  uint size = 10;
  uint h_seed = 1;
  if(argc > 1)
    sscanf(argv[1], "%d", &size);

  uint *d_output, *h_output;

  int n_blocks = (size + block_size - 1)/block_size;

  gpuErrchk(cudaMalloc(&d_output, n_blocks*sizeof(uint)));
  h_output = (uint*)malloc(n_blocks*sizeof(uint));

  cudaMemcpyToSymbol(d_seed, &h_seed, sizeof(uint));
  estimate_pi<<<n_blocks,block_size>>>(d_output, size);

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_output, d_output, n_blocks*sizeof(uint),
                       cudaMemcpyDeviceToHost));

  long final_result = 0;
  for(int ii = 0; ii < n_blocks; ++ii)
    final_result += h_output[ii];

  float estimate = 4 * (float)final_result/size;
  printf("%f\n", estimate);
  
  return 0;
}
