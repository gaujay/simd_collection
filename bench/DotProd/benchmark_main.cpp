/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

// Google benchmark
#include <benchmark/benchmark.h>

// Options
#define INNER_LOOP 200
//#define SRAND_SEED 55150

#define BM_MULT 2     // Sparse range multiplier
#ifndef BM_MULT       // or
  #define BM_INC 8<<6 // Dense range increment
#endif
#define BM_MIN 8<<3   // 64
#define BM_MAX 8<<12  // 32768

// Helper
#ifdef BM_MULT
  constexpr int bm_mult = BM_MULT;
  #define BM_PARAM bm_mult
  #define BM_RANGE(mult, min, max) RangeMultiplier(mult)->Range(min, max)
#else
  constexpr int bm_inc = BM_INC;
  #define BM_PARAM bm_inc
  #define BM_RANGE(inc, min, max) DenseRange(min, max, inc)
#endif


// Benchmarks
#include "benchmark_dotp_i8.h"
#include "benchmark_dotp_i8ui8.h"
#include "benchmark_dotp_i16i8.h"
#include "benchmark_dotp_i16.h"
#include "benchmark_dotp_i32i16.h"
#include "benchmark_dotp_i32.h"
#include "benchmark_dotp_flt.h"
#include "benchmark_dotp_dbl.h"


//
BENCHMARK_MAIN();
