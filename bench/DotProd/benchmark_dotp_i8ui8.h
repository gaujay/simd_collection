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

#ifndef HAS_SSE2_
  #error "Minimum SIMD support is SSE2"
#endif
#ifndef HAS_SSSE3_
  #warning "Benchmarking SSE2 version (SSSE3 recommended)"
#endif


// Data alignment optimizations
#define DOTP88_SIZE_MULTIPLE 64
//#define DOTP88_128_ALIGNED
//#define DOTP88_256_ALIGNED
#include "DotProd/dotp_i8ui8.h"

// Constants
#ifndef INNER_LOOP
  #define INNER_LOOP 50
#endif
#ifndef SRAND_SEED
  #define SRAND_SEED 55150
#endif


//
void BM_DotP88_ForcedScalar(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int8_t, uint8_t>(1, N, 0, 32);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i8ui8_scalarforced(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
void BM_DotP88_Scalar(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int8_t, uint8_t>(1, N, 0, 32);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i8ui8_scalar(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
#ifdef HAS_SSSE3_
void BM_DotP88_NaiveSSE(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int8_t, uint8_t>(1, N, 0, 32);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i8ui8_sse_naive(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}
#endif

//
#ifdef HAS_SSSE3_
void BM_DotP88_SSE(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int8_t, uint8_t>(1, N, 0, 32);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i8ui8_sse(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}
#endif

//
#ifdef HAS_AVX2_
void BM_DotP88_AVX2(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int8_t, uint8_t>(1, N, 0, 32);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i8ui8_avx2(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}
#endif


//
//BENCHMARK(BM_DotP88_ForcedScalar)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
BENCHMARK(BM_DotP88_Scalar)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#ifdef HAS_SSSE3_
  BENCHMARK(BM_DotP88_NaiveSSE)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
  BENCHMARK(BM_DotP88_SSE)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#endif
#ifdef HAS_AVX2_
  BENCHMARK(BM_DotP88_AVX2)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#endif
