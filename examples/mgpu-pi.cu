#include <moderngpu/kernel_reduce.hxx>
#include <curand-done-right/curanddr.hxx>

int main(int argc, char **argv) {
  int size = 100;
  uint seed = 1;

  if(argc > 1)
    sscanf(argv[1], "%d", &size);

  mgpu::standard_context_t context;

  mgpu::mem_t<int> answer(1, context);
  mgpu::transform_reduce(
    [=]__device__(uint index) {
      auto randoms = curanddr::uniforms<2>(uint3{0,0,0},
                                           uint2{index, seed});
      float xx = randoms[0], yy = randoms[1];
      int integrand = 0;
      if(xx*xx + yy*yy < 1)
        integrand = 1;

      return integrand;
    },
    size,
    answer.data(),
    mgpu::plus_t<int>(),
    context);

  float estimate = 4 * (float)(mgpu::from_mem(answer)[0])/size;
  printf("%f\n", estimate);
  
  return 0;
}
