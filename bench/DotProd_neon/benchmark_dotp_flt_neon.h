/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

// Benchmark
#include <benchmark/benchmark.h>

// Std
#include <cstdint>
#include <cstdlib>
#include <vector>

// Utils
#include "Utils/generators.h"
#include "Utils/compiler_utils.h"

#ifndef HAS_NEON_
  #error "Minimum SIMD support for ARM is NEON"
#endif


// Data alignment optimizations
#define DOTPFLT_NEON_SIZE_MULTIPLE 16
#include "DotProd_neon/dotp_flt_neon.h"

// Constants
#ifndef INNER_LOOP
  #define INNER_LOOP 50
#endif
#ifndef SRAND_SEED
  #define SRAND_SEED 55150
#endif


//
void BM_DotPFLT_Scalar(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrdf<float>(1, N, -1.f, 1.f);
  float ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_flt_neon_scalar(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
void BM_DotPFLT_NaiveNeon(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrdf<float>(1, N, -1.f, 1.f);
  float ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_flt_neon_naive(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
void BM_DotPFLT_Neon(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrdf<float>(1, N, -1.f, 1.f);
  float ttl = 0;
 
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_flt_neon(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}


//
BENCHMARK(BM_DotPFLT_Scalar)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
BENCHMARK(BM_DotPFLT_NaiveNeon)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
BENCHMARK(BM_DotPFLT_Neon)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
