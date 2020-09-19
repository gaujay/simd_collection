/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

// Benchmark
#include <benchmark/benchmark.h>

#define INNER_LOOP 100
//#define SRAND_SEED 55150

#include "benchmark_nsort_8_i8.h"
#include "benchmark_nsort_8_i16.h"
#include "benchmark_nsort_8_i32.h"
#include "benchmark_nsort_8_flt.h"
#include "benchmark_nsort_8_dbl.h"


//
BENCHMARK_MAIN();
