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
//#define NSORT_8_I32_EARLY_EXIT // Enable if array rarely need inter-lanes sorting (first 4 vs last 4)
//#define NSORT_8_I32_128_ALIGNED
//#define NSORT_8_I32_256_ALIGNED
#include "NetSort/nsort_8_i32.h"

// Constants
#ifndef INNER_LOOP
  #define INNER_LOOP 50
#endif
#ifndef SRAND_SEED
  #define SRAND_SEED 55150
#endif


//
void BM_NSort_8I32_QSORT_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  int32_t v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrd(v0, 8*INNER_LOOP, -5000, 5000);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(int32_t));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_i32_qsort(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8I32_QSORT_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  int32_t v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(int32_t));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_i32_qsort(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8I32_STDSORT_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  int32_t v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrd(v0, 8*INNER_LOOP, -5000, 5000);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(int32_t));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      std::sort(v1 + i*8, v1+8 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8I32_STDSORT_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  int32_t v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(int32_t));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      std::sort(v1+i*8, v1+8 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

#ifdef HAS_SSE4_1_
//
void BM_NSort_8I32_SSE_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  int32_t v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrd(v0, 8*INNER_LOOP, -5000, 5000);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(int32_t));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_i32_sse(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8I32_SSE_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  int32_t v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(int32_t));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_i32_sse(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}
#endif // HAS_SSE4_1_

#ifdef HAS_AVX2_
//
void BM_NSort_8I32_AVX2_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  int32_t v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrd(v0, 8*INNER_LOOP, -5000, 5000);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(int32_t));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_i32_avx2(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8I32_AVX2_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  int32_t v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(int32_t));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_i32_avx2(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}
#endif // HAS_AVX2_


//
BENCHMARK(BM_NSort_8I32_QSORT_RND);
BENCHMARK(BM_NSort_8I32_QSORT_SEQ);
BENCHMARK(BM_NSort_8I32_STDSORT_RND);
BENCHMARK(BM_NSort_8I32_STDSORT_SEQ);
#ifdef HAS_SSE4_1_
  BENCHMARK(BM_NSort_8I32_SSE_RND);
  BENCHMARK(BM_NSort_8I32_SSE_SEQ);
#endif
#ifdef HAS_AVX2_
  BENCHMARK(BM_NSort_8I32_AVX2_RND);
  BENCHMARK(BM_NSort_8I32_AVX2_SEQ);
#endif
