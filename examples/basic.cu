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

#define BLOCK_SIZE 8

__constant__ uint d_seed;
__global__ void estimate_pi(int* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;  
  auto randoms = curanddr::uniforms<1>(uint3{0,0,0},
                                       uint2{index, d_seed});
  output[0] = 1000*randoms[0];
/*  __shared__ int answer[BLOCK_SIZE];

  int nthreads = blockDim.x;

  while(nthreads > 1) {
    int half_point = nthreads >> 1;

    if (threadIdx.x < half_point) {
      int thread2 = threadIdx.x + half_point;

      T temp = answer[thread2];
      answer[threadIdx.x] += temp;
    }
    __syncthreads();

    nthreads = half_point;
    } */
}

int main(int argc, char **argv) {
  int size = 1000;
  uint h_seed = 1;
  if(argc > 1)
    sscanf(argv[1], "%d", &h_seed);

  int *d_output, h_output;
  
  gpuErrchk(cudaMalloc(&d_output, sizeof(int)));

  cudaMemcpyToSymbol(d_seed, &h_seed, sizeof(uint));
  estimate_pi<<<1,1>>>(d_output);

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));
  printf("%d\n", h_output);
  
  return 0;
}
