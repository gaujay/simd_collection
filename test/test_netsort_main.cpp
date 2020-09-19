/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#include "gtest/gtest.h"

#include "Utils/compiler_utils.h"
#include "Utils/generators.h"

#include "NetSort/nsort_8_i8.h"
#include "NetSort/nsort_8_i16.h"
#include "NetSort/nsort_8_i32.h"
#include "NetSort/nsort_8_flt.h"
#include "NetSort/nsort_8_dbl.h"

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>

#ifndef HAS_AVX_
  #warning "Testing non-optimal version (SSSE3/SSE4.1/AVX recommended)"
#endif

static unsigned int _seed = static_cast<unsigned int>(std::time(nullptr));


// Test NetSort for 8 x int8
TEST(NetSortTest, NetSort_8_i8) {
  std::srand(_seed);
  std::vector<int8_t> v0(8);
  vec_rrd(v0, (int8_t)-127, (int8_t)127);
  auto v1 = v0, v2 = v0;

  netsort_8_i8_qsort(v0.data());

#ifdef HAS_SSSE3_
  netsort_8_i8_sse(v1.data());  EXPECT_EQ(v0, v1);
#endif
}

// Test NetSort for 8 x int16
TEST(NetSortTest, NetSort_8_i16) {
  std::srand(_seed);
  std::vector<int16_t> v0(8);
  vec_rrd(v0, (int16_t)-5000, (int16_t)5000);
  auto v1 = v0, v2 = v0;

  netsort_8_i16_qsort(v0.data());

#ifdef HAS_SSSE3_
  netsort_8_i16_sse(v1.data());  EXPECT_EQ(v0, v1);
#endif
}

// Test NetSort for 8 x int32
TEST(NetSortTest, NetSort_8_i32) {
  std::srand(_seed);
  std::vector<int32_t> v0(8);
  vec_rrd(v0, -5000, 5000);
  auto v1 = v0, v2 = v0;

  netsort_8_i32_qsort(v0.data());

#ifdef HAS_SSE4_1_
  netsort_8_i32_sse(v1.data());  EXPECT_EQ(v0, v1);
#endif
#ifdef HAS_AVX2_
  netsort_8_i32_avx2(v2.data()); EXPECT_EQ(v0, v2);
#endif
}

// Test NetSort for 8 x float
TEST(NetSortTest, NetSort_8_flt) {
  std::srand(_seed);
  std::vector<float> v0(8);
  vec_rrdf(v0, -1.f, 1.f);
  auto v1 = v0, v2 = v0;

  netsort_8_flt_qsort(v0.data());

  netsort_8_flt_sse(v1.data()); EXPECT_EQ(v0, v1);
#ifdef HAS_AVX_
  netsort_8_flt_avx(v2.data()); EXPECT_EQ(v0, v2);
#endif
}

// Test NetSort for 8 x double
TEST(NetSortTest, NetSort_8_dbl) {
  std::srand(_seed);
  std::vector<double> v0(8);
  vec_rrdf(v0, -1., 1.);
  auto v1 = v0, v2 = v0;

  netsort_8_dbl_qsort(v0.data());

#ifdef HAS_AVX_
  netsort_8_dbl_avx(v2.data()); EXPECT_EQ(v0, v2);
#endif
}
