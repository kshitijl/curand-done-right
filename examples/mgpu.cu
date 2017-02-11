/*
  Compute the mean and variance of 1 million normally-distributed
  random floats. These should be close to 0 and 1 if everything is
  working well.
*/

#include <moderngpu/kernel_reduce.hxx>
#include <curand-done-right/curanddr.hxx>

/*
  We will compute both the sum and sum-of-squares simultaneously,
  accumulating into a float2.
 */
struct plus_float2 : public std::binary_function<float2, float2, float2> {
  __host__ __device__ float2 operator()(float2 a, float2 b) const {
    return float2{a.x+b.x, a.y+b.y};
  }  
};

int main() {
  int size = 1e6;

  mgpu::standard_context_t context;

  mgpu::mem_t<float2> answer(1, context);
  mgpu::transform_reduce(
    []__device__(uint index) {
      /* Generate 1 gaussian random number. This will always be a
         standard normal gaussian, which you can then transform to the
         appropriate mean and variance.

         Note that four of the five possible integer inputs are 0, and
         only one varies between items.
       */
      auto randoms = curanddr::gaussians<1>(uint3{0,0,0},
                                          uint2{index, 0});
      /* Sum x and x**2 simultaneously */
      return float2{randoms[0], randoms[0]*randoms[0]};
    },
    size,
    answer.data(),
    plus_float2(),
    context);

  float2 moments = mgpu::from_mem(answer)[0];
  float mean = moments.x/size;
  float variance = moments.y/size - mean*mean;

  printf("%f %f\n", mean, variance);
  
  return 0;
}
