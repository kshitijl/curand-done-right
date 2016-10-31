#include <iostream>

#include <thrust/tabulate.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

int main() {
  int size = 10;
  
  thrust::device_vector<float> samples(size);
  thrust::tabulate(thrust::device,
                   samples.begin(), samples.end(),
                   []__device__(uint index) {
                     return index;
                   });

  float sum = thrust::reduce(samples.begin(), samples.end(), float(0),
                             thrust::plus<float>());
  std::cout << sum << "\n";
  return 0;
}

