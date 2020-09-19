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
#ifndef HAS_SSE4_1_
  #warning "Benchmarking SSE2 version (SSE4.1 recommended)"
#endif


// Data alignment optimizations
#define DOTP32_SIZE_MULTIPLE 32
//#define DOTP32_128_ALIGNED
//#define DOTP32_256_ALIGNED
#include "DotProd/dotp_i32.h"

// Constants
#ifndef INNER_LOOP
  #define INNER_LOOP 50
#endif
#ifndef SRAND_SEED
  #define SRAND_SEED 55150
#endif


//
void BM_DotP32_ForcedScalar(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int32_t, int32_t>(1, N, -16, 16);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i32_scalarforced(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
void BM_DotP32_Scalar(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int32_t, int32_t>(1, N, -16, 16);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i32_scalar(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
#ifdef HAS_SSSE3_
void BM_DotP32_NaiveSSE(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int32_t, int32_t>(1, N, -16, 16);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i32_sse_naive(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}
#endif

//
void BM_DotP32_SSE(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int32_t, int32_t>(1, N, -16, 16);
  int32_t ttl = 0;
  
  for (auto _ : state)
  {
   ttl = 0;
   for (size_t i=0; i<INNER_LOOP; ++i)
     benchmark::DoNotOptimize(ttl += dotProduct_i32_sse(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
#ifdef HAS_AVX2_
void BM_DotP32_AVX2(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrd<int32_t, int32_t>(1, N, -16, 16);
  int32_t ttl = 0;
 
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_i32_avx2(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}
#endif


//
//BENCHMARK(BM_DotP32_ForcedScalar)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
BENCHMARK(BM_DotP32_Scalar)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#ifdef HAS_SSSE3_
  BENCHMARK(BM_DotP32_NaiveSSE)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#endif
BENCHMARK(BM_DotP32_SSE)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#ifdef HAS_AVX2_
  BENCHMARK(BM_DotP32_AVX2)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#endif
