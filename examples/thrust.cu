#include <stdio.h>

#include <thrust/tabulate.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>

#include <curand-done-right/curanddr.hxx>

using mf = thrust::tuple<float, float>;

/*
  We will compute both the sum and sum-of-squares simultaneously,
  accumulating into a float2.
 */
struct mfp : public thrust::binary_function<mf, mf, mf> {
  __host__ __device__ mf operator()(const mf& a, const mf& b) const {
    return mf{thrust::get<0>(a) + thrust::get<0>(b),
        thrust::get<1>(a) + thrust::get<1>(b)};
  }  
};

int main() {
  int size = 1000000;
  
  thrust::device_vector<float> samples(size);
  thrust::tabulate(thrust::device,
                   samples.begin(), samples.end(),
                   []__device__(uint index) {
                     auto random_normals = curanddr::gaussians<1>(uint3{0,0,0},
                                                                  uint2{index, 0});
                     return random_normals[0];                     
                   });
  auto moments = thrust::transform_reduce(samples.begin(), samples.end(),
                                          []__device__(float xx) {
                                            return mf{xx,xx*xx};
                                          },
                                          mf{0,0},
                                          mfp());


  float mean = thrust::get<0>(moments)/size;
  float variance = thrust::get<1>(moments)/size - mean*mean;
  printf("%f %f\n", mean, variance);
  return 0;
}

