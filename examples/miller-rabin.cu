#include <moderngpu/kernel_reduce.hxx>
#include <curand-done-right/curanddr.hxx>

using ulong = unsigned long;

int main(int argc, char**argv) {
  uint size = 1;
  int input_n = 46349;

  if(argc > 1)
    sscanf(argv[1], "%d", &input_n);

  /* Write n-1 as 2^k * d, where k is maximal. In other words, find
   * the rightmost 1 in the bitwise representation of n-1; the number
   * with just that bit set is 2^k. */
  int nm1 = input_n - 1;
  int two_to_k = nm1 & -nm1;
  int dd = nm1 / two_to_k;
  assert(dd*two_to_k == nm1);

  mgpu::standard_context_t context;

  mgpu::mem_t<int> result(1, context);
  mgpu::transform_reduce(
    [=]__device__(uint index) {
      auto randoms = curanddr::uniforms<1>(uint3{0,0,0},
                                           uint2{0, index});
      //int base_a = 2 + randoms[0]*(input_n-4);
      int base_a = 46342;
      
      uint xx = 1;
      for(int ii = 0; ii < dd; ++ii)
        xx = xx*base_a % input_n;

      if(xx == 1 or xx == input_n - 1) {
        return 0;
      }

      for(int power = two_to_k; power > 2; power /= 2) {
        xx = xx*xx % input_n;

        if(xx == 1) {
          printf("Witness loop %d\n", base_a);
          return 1;
        }
        if(xx == input_n-1)
          return 0;
      }

      printf("Witness doop %d\n", base_a);
      return 1;
    },
    size,
    result.data(),
    mgpu::plus_t<int>(),
    context);

  if(mgpu::from_mem(result)[0] > 0)
    printf("Composite\n");
  else
    printf("Probably prime\n");
  return 0;
}
