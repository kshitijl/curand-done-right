#include <moderngpu/kernel_reduce.hxx>
#include <curand-done-right/curanddr.hxx>

int main() {
  int size = 100000;

  mgpu::standard_context_t context;

  mgpu::mem_t<float2> answer(1, context);
  mgpu::transform_reduce(
    []__device__(uint index) {
      auto randoms = curanddr::gaussians<1>(uint3{0,0,0},
                                          uint2{index, 0});
      return float2{randoms[0], randoms[0]*randoms[0]};
    },
    size,
    answer.data(),
    mgpu::plus_t<float2>(),
    context);

  
  return 0;
}
