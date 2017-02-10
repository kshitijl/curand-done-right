#include <iostream>

#include <thrust/tabulate.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <curand-done-right/curanddr.hxx>

int main() {
  int size = 1000;
  
  thrust::device_vector<float> samples(size);
  thrust::tabulate(thrust::device,
                   samples.begin(), samples.end(),
                   []__device__(uint index) {
                     auto random_normals = curanddr::gaussians<1>(uint3{0,0,0},
                                                                  uint2{0, index});
                     return random_normals[0];                     
                   });
  
  // float2 sum = thrust::reduce(samples.begin(), samples.end(), float2{0},
  //                             []__device__(float2 b, float2 a) {
  //                               return float2{a+b.x, a+b.y};
  //                             });
  // std::cout << sum.x/size << "\n";
  return 0;
}

