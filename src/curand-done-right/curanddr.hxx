#pragma once

#include <curand_kernel.h>
#include <curand_normal.h>

namespace curanddr {
  template <int Arity, typename num_t = float>
  struct alignas(8) vector_t {
    num_t values[Arity];
    __device__ num_t operator[] (size_t n) const {
      return values[n];
    }    
    
  };

  // from moderngpu meta.hxx
  template<int i, int count, bool valid = (i < count)>
  struct iterate_t {
    template<typename func_t>
    __device__ static void eval(func_t f) {
      f(i);
      iterate_t<i+1, count>::eval(f);
    }
  };

  template<int i, int count>
  struct iterate_t<i, count, false> {
    template<typename func_t>
    __device__ static void eval(func_t f) { }
  };

  template<int count, typename func_t>
  __device__ void iterate(func_t f) {
    iterate_t<0, count>::eval(f);
  }
  
  template<int Arity>
  __device__ vector_t<Arity> gaussians(uint3 counter, uint2 key) {
    enum { n_blocks = (Arity + 4 - 1)/4 };

    float scratch[n_blocks * 4];
  
    iterate<n_blocks>([&](uint index) {
        uint4 local_counter{counter.x, counter.y, counter.z, index};
        uint4 result = curand_Philox4x32_10(local_counter, key);

        float2 hi = _curand_box_muller(result.x, result.y);
        float2 lo = _curand_box_muller(result.z, result.w);

        uint ii = index*4;
        scratch[ii] = hi.x;
        scratch[ii+1] = hi.y;
        scratch[ii+2] = lo.x;
        scratch[ii+3] = lo.y;
      });

    vector_t<Arity> answer;

    iterate<Arity>([&](uint index) {
        answer.values[index] = scratch[index];
      });
  
    return answer;
  }

  template<int Arity>
  __device__ vector_t<Arity> uniforms(uint3 counter, uint2 key) {
    enum { n_blocks = (Arity + 4 - 1)/4 };

    float scratch[n_blocks * 4];
  
    iterate<n_blocks>([&](uint index) {
        uint4 local_counter{counter.x, counter.y, counter.z, index};
        uint4 result = curand_Philox4x32_10(local_counter, key);

        uint ii = index*4;
        scratch[ii]   = _curand_uniform(result.x);
        scratch[ii+1] = _curand_uniform(result.y);
        scratch[ii+2] = _curand_uniform(result.z);
        scratch[ii+3] = _curand_uniform(result.w);
      });

    vector_t<Arity> answer;

    iterate<Arity>([&](uint index) {
        answer.values[index] = scratch[index];
      });
  
    return answer;
  }  
}
