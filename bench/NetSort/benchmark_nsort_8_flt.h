/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

// Benchmark
#include <benchmark/benchmark.h>

// Std
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

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
//#define NSORT_8_FLT_EARLY_EXIT // Enable if array rarely need inter-lanes sorting (first 4 vs last 4)
//#define NSORT_8_FLT_128_ALIGNED
//#define NSORT_8_FLT_256_ALIGNED
#include "NetSort/nsort_8_flt.h"

// Constants
#ifndef INNER_LOOP
  #define INNER_LOOP 50
#endif
#ifndef SRAND_SEED
  #define SRAND_SEED 55150
#endif


//
void BM_NSort_8FLT_QSORT_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  float v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrdf(v0, 8*INNER_LOOP, -1.f, 1.f);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(float));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_flt_qsort(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8FLT_QSORT_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  float v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0.f);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(float));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_flt_qsort(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8FLT_STDSORT_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  float v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrdf(v0, 8*INNER_LOOP, -1.f, 1.f);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(float));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      std::sort(v1 + i*8, v1+8 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8FLT_STDSORT_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  float v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0.f);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(float));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      std::sort(v1+i*8, v1+8 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8FLT_SSE_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  float v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrdf(v0, 8*INNER_LOOP, -1.f, 1.f);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(float));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_flt_sse(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8FLT_SSE_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  float v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0.f);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(float));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_flt_sse(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

#ifdef HAS_AVX_
//
void BM_NSort_8FLT_AVX_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  float v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrdf(v0, 8*INNER_LOOP, -1.f, 1.f);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(float));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_flt_avx(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8FLT_AVX_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  float v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0.f);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(float));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_flt_avx(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}
#endif


//
BENCHMARK(BM_NSort_8FLT_QSORT_RND);
BENCHMARK(BM_NSort_8FLT_QSORT_SEQ);
BENCHMARK(BM_NSort_8FLT_STDSORT_RND);
BENCHMARK(BM_NSort_8FLT_STDSORT_SEQ);
BENCHMARK(BM_NSort_8FLT_SSE_RND);
BENCHMARK(BM_NSort_8FLT_SSE_SEQ);
#ifdef HAS_AVX_
  BENCHMARK(BM_NSort_8FLT_AVX_RND);
  BENCHMARK(BM_NSort_8FLT_AVX_SEQ);
#endif
