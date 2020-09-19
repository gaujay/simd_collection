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


// Data alignment optimizations
#define DOTPDBL_SIZE_MULTIPLE 16
//#define DOTPDBL_128_ALIGNED
//#define DOTPDBL_256_ALIGNED
#include "DotProd/dotp_dbl.h"

// Constants
#ifndef INNER_LOOP
  #define INNER_LOOP 50
#endif
#ifndef SRAND_SEED
  #define SRAND_SEED 55150
#endif


//
void BM_DotPDBL_ForcedScalar(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrdf<double>(1, N, -1., 1.);
  double ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_dbl_scalarforced(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
void BM_DotPDBL_Scalar(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrdf<double>(1, N, -1., 1.);
  double ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_dbl_scalar(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
#ifdef HAS_SSE3_
void BM_DotPDBL_NaiveSSE(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrdf<double>(1, N, -1., 1.);
  double ttl = 0;
  
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_dbl_sse_naive(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}
#endif

//
void BM_DotPDBL_SSE(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrdf<double>(1, N, -1., 1.);
  double ttl = 0;
 
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_dbl_sse(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}

//
#ifdef HAS_AVX_
void BM_DotPDBL_AVX(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrdf<double>(1, N, -1., 1.);
  double ttl = 0;
 
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_dbl_avx(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}
#endif

//
#ifdef HAS_FMA_
void BM_DotPDBL_FMA(benchmark::State& state) {
  const size_t N = (size_t)state.range(0);
  std::srand(SRAND_SEED);
  auto dv = dual_vec_rrdf<double>(1, N, -1., 1.);
  double ttl = 0;
 
  for (auto _ : state)
  {
    ttl = 0;
    for (size_t i=0; i<INNER_LOOP; ++i)
      benchmark::DoNotOptimize(ttl += dotProduct_dbl_fma(dv[0].u.data(), dv[0].v.data(), N));
  }
  benchmark::DoNotOptimize(ttl);
}
#endif


//
//BENCHMARK(BM_DotPDBL_ForcedScalar)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
BENCHMARK(BM_DotPDBL_Scalar)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#ifdef HAS_SSE3_
  BENCHMARK(BM_DotPDBL_NaiveSSE)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#endif
BENCHMARK(BM_DotPDBL_SSE)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#ifdef HAS_AVX_
  BENCHMARK(BM_DotPDBL_AVX)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#endif
#ifdef HAS_FMA_
  BENCHMARK(BM_DotPDBL_FMA)->BM_RANGE(BM_PARAM, BM_MIN, BM_MAX);
#endif
