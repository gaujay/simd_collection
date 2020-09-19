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
#define DOTP168_NEON_SIZE_MULTIPLE 16
#include "DotProd_neon/dotp_i16i8_neon.h"

// Constants
#ifndef INNER_LOOP
  #define INNER_LOOP 50
#endif
#ifndef SRAND_SEED
  #define SRAND_SEED 55150
#endif


//
void BM_DotP168_Scalar(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int16_t, int8_t>(1, N, -16, 16);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i16i8_neon_scalar(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
void BM_DotP168_NaiveNeon(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int16_t, int8_t>(1, N, -16, 16);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i16i8_neon_naive(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
void BM_DotP168_Neon(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int16_t, int8_t>(1, N, -16, 16);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i16i8_neon(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}


//
BENCHMARK(BM_DotP168_Scalar)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
BENCHMARK(BM_DotP168_NaiveNeon)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
BENCHMARK(BM_DotP168_Neon)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
