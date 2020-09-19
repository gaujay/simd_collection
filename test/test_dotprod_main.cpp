/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#include "gtest/gtest.h"

#include "Utils/compiler_utils.h"
#include "Utils/generators.h"

#include "DotProd/dotp_i8.h"
#include "DotProd/dotp_i8ui8.h"
#include "DotProd/dotp_i16i8.h"
#include "DotProd/dotp_i16.h"
#include "DotProd/dotp_i32i16.h"
#include "DotProd/dotp_i32.h"
#include "DotProd/dotp_flt.h"
#include "DotProd/dotp_dbl.h"

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>

static unsigned int _seed = static_cast<unsigned int>(std::time(nullptr));


// Test DotProd for int8
TEST(DotProdTest, DotProd_i8) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrd<int8_t, int8_t>(1, count, -50, 50);

  int32_t expected = dotProduct_i8_scalar(dv[0].u.data(), dv[0].v.data(), count);

#ifdef HAS_SSSE3_
  EXPECT_EQ(expected, dotProduct_i8_sse_naive(dv[0].u.data(), dv[0].v.data(), count));
#endif
  EXPECT_EQ(expected, dotProduct_i8_sse(dv[0].u.data(), dv[0].v.data(), count));
#ifdef HAS_AVX2_
  EXPECT_EQ(expected, dotProduct_i8_avx2(dv[0].u.data(), dv[0].v.data(), count));
#endif
}

// Test DotProd for int8 x uint8
TEST(DotProdTest, DotProd_i8ui8) {
  std::srand(_seed);
  size_t count = 1023;
  std::vector< int8_t> u(count);
  std::vector<uint8_t> v(count);
  // /!\ Saturate if not: -2^15 <= u[N]*v[N] + u[N+1]*v[N+1] <= 2^15 - 1 (with N even)
  vec_rrd(u, ( int8_t)-128,  (int8_t)127);
  vec_rrd(v, (uint8_t)   0, (uint8_t)127);

  int32_t expected = dotProduct_i8ui8_scalar(u.data(), v.data(), count);
  
#ifdef HAS_SSSE3_
  EXPECT_EQ(expected, dotProduct_i8ui8_sse_naive(u.data(), v.data(), count));
  EXPECT_EQ(expected, dotProduct_i8ui8_sse(u.data(), v.data(), count));
#endif
#ifdef HAS_AVX2_
  EXPECT_EQ(expected, dotProduct_i8ui8_avx2(u.data(), v.data(), count));
#endif
}

// Test DotProd for int16 x int8
TEST(DotProdTest, DotProd_i16i8) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrd<int16_t, int8_t>(1, count, -50, 50);

  int32_t expected = dotProduct_i16i8_scalar(dv[0].u.data(), dv[0].v.data(), count);
  
#ifdef HAS_SSSE3_
  EXPECT_EQ(expected, dotProduct_i16i8_sse_naive(dv[0].u.data(), dv[0].v.data(), count));
#endif
  EXPECT_EQ(expected, dotProduct_i16i8_sse(dv[0].u.data(), dv[0].v.data(), count));
#ifdef HAS_AVX2_
  EXPECT_EQ(expected, dotProduct_i16i8_avx2(dv[0].u.data(), dv[0].v.data(), count));
#endif
}

// Test DotProd for int16
TEST(DotProdTest, DotProd_i16) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrd<int16_t, int16_t>(1, count, -50, 50);

  int32_t expected = dotProduct_i16_scalar(dv[0].u.data(), dv[0].v.data(), count);
  
#ifdef HAS_SSSE3_
  EXPECT_EQ(expected, dotProduct_i16_sse_naive(dv[0].u.data(), dv[0].v.data(), count));
#endif
  EXPECT_EQ(expected, dotProduct_i16_sse(dv[0].u.data(), dv[0].v.data(), count));
#ifdef HAS_AVX2_
  EXPECT_EQ(expected, dotProduct_i16_avx2(dv[0].u.data(), dv[0].v.data(), count));
#endif
}

// Test DotProd for int32 x int16
TEST(DotProdTest, DotProd_i32i16) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrd<int32_t, int16_t>(1, count, -50, 50);

  int32_t expected = dotProduct_i32i16_scalar(dv[0].u.data(), dv[0].v.data(), count);
  
#ifdef HAS_SSSE3_
  EXPECT_EQ(expected, dotProduct_i32i16_sse_naive(dv[0].u.data(), dv[0].v.data(), count));
#endif
  EXPECT_EQ(expected, dotProduct_i32i16_sse(dv[0].u.data(), dv[0].v.data(), count));
#ifdef HAS_AVX2_
  EXPECT_EQ(expected, dotProduct_i32i16_avx2(dv[0].u.data(), dv[0].v.data(), count));
#endif
}

// Test DotProd for int32
TEST(DotProdTest, DotProd_i32) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrd<int32_t, int32_t>(1, count, -50, 50);

  int32_t expected = dotProduct_i32_scalar(dv[0].u.data(), dv[0].v.data(), count);
  
#ifdef HAS_SSSE3_
  EXPECT_EQ(expected, dotProduct_i32_sse_naive(dv[0].u.data(), dv[0].v.data(), count));
#endif
  EXPECT_EQ(expected, dotProduct_i32_sse(dv[0].u.data(), dv[0].v.data(), count));
#ifdef HAS_AVX2_
  EXPECT_EQ(expected, dotProduct_i32_avx2(dv[0].u.data(), dv[0].v.data(), count));
#endif
}

// Test DotProd for float
TEST(DotProdTest, DotProd_flt) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrdf<float>(1, count, -1.f, 1.f);
  
  double expected = (double)dotProduct_flt_scalar(dv[0].u.data(), dv[0].v.data(), count);
  
#ifdef HAS_SSE3_
  EXPECT_NEAR(expected, (double)dotProduct_flt_sse_naive(dv[0].u.data(), dv[0].v.data(), count), 0.0015);
#endif
  EXPECT_NEAR(expected, (double)dotProduct_flt_sse(dv[0].u.data(), dv[0].v.data(), count), 0.0015);
#ifdef HAS_AVX_
  EXPECT_NEAR(expected, (double)dotProduct_flt_avx(dv[0].u.data(), dv[0].v.data(), count), 0.0015);
#endif
#ifdef HAS_FMA_
  EXPECT_NEAR(expected, (double)dotProduct_flt_fma(dv[0].u.data(), dv[0].v.data(), count), 0.0015);
#endif
}

// Test DotProd for double
TEST(DotProdTest, DotProd_dbl) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrdf<double>(1, count, -1., 1.);
  
  double expected = dotProduct_dbl_scalar(dv[0].u.data(), dv[0].v.data(), count);
  
#ifdef HAS_SSE3_
  EXPECT_NEAR(expected, dotProduct_dbl_sse_naive(dv[0].u.data(), dv[0].v.data(), count), 0.0000015);
#endif
  EXPECT_NEAR(expected, dotProduct_dbl_sse(dv[0].u.data(), dv[0].v.data(), count), 0.0000015);
#ifdef HAS_AVX_
  EXPECT_NEAR(expected, dotProduct_dbl_avx(dv[0].u.data(), dv[0].v.data(), count), 0.0000015);
#endif
#ifdef HAS_FMA_
  EXPECT_NEAR(expected, (double)dotProduct_dbl_fma(dv[0].u.data(), dv[0].v.data(), count), 0.0000015);
#endif
}
