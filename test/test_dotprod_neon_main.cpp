/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#include "gtest/gtest.h"

#include "Utils/compiler_utils.h"
#include "Utils/generators.h"

#ifndef HAS_NEON_
  #error "Minimum SIMD support for ARM is NEON"
#endif

#include "DotProd_neon/dotp_i8_neon.h"
#include "DotProd_neon/dotp_i16i8_neon.h"
#include "DotProd_neon/dotp_i16_neon.h"
#include "DotProd_neon/dotp_i32i16_neon.h"
#include "DotProd_neon/dotp_i32_neon.h"
#include "DotProd_neon/dotp_flt_neon.h"

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>

static unsigned int _seed = static_cast<unsigned int>(std::time(nullptr));


// Test DotProd for int8
TEST(DotProdTest, DotProd_i8_neon) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrd<int8_t, int8_t>(1, count, -50, 50);

  int32_t expected = dotProduct_i8_neon_scalar(dv[0].u.data(), dv[0].v.data(), count);

  EXPECT_EQ(expected, dotProduct_i8_neon_naive(dv[0].u.data(), dv[0].v.data(), count));
  EXPECT_EQ(expected, dotProduct_i8_neon(dv[0].u.data(), dv[0].v.data(), count));
}

// Test DotProd for int16 x int8
TEST(DotProdTest, DotProd_i16i8_neon) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrd<int16_t, int8_t>(1, count, -50, 50);

  int32_t expected = dotProduct_i16i8_neon_scalar(dv[0].u.data(), dv[0].v.data(), count);
  
  EXPECT_EQ(expected, dotProduct_i16i8_neon_naive(dv[0].u.data(), dv[0].v.data(), count));
  EXPECT_EQ(expected, dotProduct_i16i8_neon(dv[0].u.data(), dv[0].v.data(), count));
}

// Test DotProd for int16
TEST(DotProdTest, DotProd_i16_neon) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrd<int16_t, int16_t>(1, count, -50, 50);

  int32_t expected = dotProduct_i16_neon_scalar(dv[0].u.data(), dv[0].v.data(), count);
  
  EXPECT_EQ(expected, dotProduct_i16_neon_naive(dv[0].u.data(), dv[0].v.data(), count));
  EXPECT_EQ(expected, dotProduct_i16_neon(dv[0].u.data(), dv[0].v.data(), count));
}

// Test DotProd for int32 x int16
TEST(DotProdTest, DotProd_i32i16_neon) {
  std::srand(_seed);
  size_t count = 1024;
  auto dv = dual_vec_rrd<int32_t, int16_t>(1, count, -50, 50);

  int32_t expected = dotProduct_i32i16_neon_scalar(dv[0].u.data(), dv[0].v.data(), count);
  
  EXPECT_EQ(expected, dotProduct_i32i16_neon_naive(dv[0].u.data(), dv[0].v.data(), count));
  EXPECT_EQ(expected, dotProduct_i32i16_neon(dv[0].u.data(), dv[0].v.data(), count));
}

// Test DotProd for int32
TEST(DotProdTest, DotProd_i32_neon) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrd<int32_t, int32_t>(1, count, -50, 50);

  int32_t expected = dotProduct_i32_neon_scalar(dv[0].u.data(), dv[0].v.data(), count);
 
  EXPECT_EQ(expected, dotProduct_i32_neon_naive(dv[0].u.data(), dv[0].v.data(), count));
  EXPECT_EQ(expected, dotProduct_i32_neon(dv[0].u.data(), dv[0].v.data(), count));
}

// Test DotProd for float
TEST(DotProdTest, DotProd_flt_neon) {
  std::srand(_seed);
  size_t count = 1023;
  auto dv = dual_vec_rrdf<float>(1, count, -1.f, 1.f);
  
  double expected = (double)dotProduct_flt_neon_scalar(dv[0].u.data(), dv[0].v.data(), count);
  
  EXPECT_NEAR(expected, (double)dotProduct_flt_neon_naive(dv[0].u.data(), dv[0].v.data(), count), 0.0015);
  EXPECT_NEAR(expected, (double)dotProduct_flt_neon(dv[0].u.data(), dv[0].v.data(), count), 0.0015);
}
