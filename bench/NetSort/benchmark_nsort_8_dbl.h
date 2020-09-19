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

#ifndef HAS_AVX_
  #warning "Benchmarking base version (AVX recommended)"
#endif


// Data alignment optimizations
//#define NSORT_8_DBL_EARLY_EXIT // Enable if array rarely need inter-lanes sorting (first 4 vs last 4)
//#define NSORT_8_DBL_256_ALIGNED
#include "NetSort/nsort_8_dbl.h"

// Constants
#ifndef INNER_LOOP
  #define INNER_LOOP 50
#endif
#ifndef SRAND_SEED
  #define SRAND_SEED 55150
#endif


//
void BM_NSort_8DBL_QSORT_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  double v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrdf(v0, 8*INNER_LOOP, -1., 1.);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(double));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_dbl_qsort(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8DBL_QSORT_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  double v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0.);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(double));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_dbl_qsort(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8DBL_STDSORT_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  double v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrdf(v0, 8*INNER_LOOP, -1., 1.);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(double));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      std::sort(v1 + i*8, v1+8 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8DBL_STDSORT_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  double v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0.);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(double));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      std::sort(v1+i*8, v1+8 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

#ifdef HAS_AVX_
//
void BM_NSort_8DBL_AVX_RND(benchmark::State& state) {
  std::srand(SRAND_SEED);
  double v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_rrdf(v0, 8*INNER_LOOP, -1., 1.);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(double));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_dbl_avx(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}

//
void BM_NSort_8DBL_AVX_SEQ(benchmark::State& state) {
  std::srand(SRAND_SEED);
  double v0[8*INNER_LOOP], v1[8*INNER_LOOP];
  vec_seq(v0, 8*INNER_LOOP, 0.);

  for (auto _ : state)
  {
    state.PauseTiming();
    memcpy(v1, v0, 8*INNER_LOOP*sizeof(double));
    state.ResumeTiming();
    for (size_t i=0; i<INNER_LOOP; ++i) {
      netsort_8_dbl_avx(v1 + i*8);
    }
  }
  benchmark::DoNotOptimize(v1);
}
#endif


//
BENCHMARK(BM_NSort_8DBL_QSORT_RND);
BENCHMARK(BM_NSort_8DBL_QSORT_SEQ);
BENCHMARK(BM_NSort_8DBL_STDSORT_RND);
BENCHMARK(BM_NSort_8DBL_STDSORT_SEQ);
#ifdef HAS_AVX_
  BENCHMARK(BM_NSort_8DBL_AVX_RND);
  BENCHMARK(BM_NSort_8DBL_AVX_SEQ);
#endif
