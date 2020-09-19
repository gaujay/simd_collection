/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef DOTP_I32I16_NEON_H
#define DOTP_I32I16_NEON_H

#include <stdint.h>
#include <arm_neon.h>   // NEON

// SIMD optimization options
#ifndef DOTP3216_NEON_SIZE_MULTIPLE
  #define DOTP3216_NEON_SIZE_MULTIPLE 0     // 16, 8, 4 (0: no optim)
#endif
//#define DOTP3216_NEON_ACCU_2              // Use 2 accumulators (depend on HW/vectors size)


//
static inline int32_t dotProduct_i32i16_neon_scalar(int32_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
  int32_t res = 0;
  for (size_t i=0; i<n; ++i)
  res += u[i] * v[i];
  
  return res;
}

// 
static inline int32_t dotProduct_i32i16_neon_naive(int32_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
  int32_t result;
  size_t count = n >> 2;
  
  // Accumulator
  int32x4_t result_4 = vdupq_n_s32(0);

  // Loop
  while (count--)
  {
    int32x4_t u_4, v_4;

    u_4 = vld1q_s32(u);
    v_4 = vmovl_s16(vld1_s16(v));

    result_4 = vmlaq_s32(result_4, u_4, v_4);
    
    // Next
    u += 4;
    v += 4;
  }
  
  // Horizontal sum
  int64x2_t tmp0 = vpaddlq_s32(result_4);
  int32x2_t tmp1 = vmovn_s64(tmp0);
  result = (int32_t)vget_lane_s64(vpaddl_s32(tmp1), 0);

#if DOTP3216_NEON_SIZE_MULTIPLE < 4
  n &= 3;
  while (n--)
    result += u[n] * v[n];
#endif
  return result;
}

// 
static inline int32_t dotProduct_i32i16_neon(int32_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
  int32_t result;
  size_t count = n >> 4;
  
  // Accumulators
  int32x4_t result0_4 = vdupq_n_s32(0);
#ifdef DOTP3216_NEON_ACCU_2
  int32x4_t result1_4 = vdupq_n_s32(0);
#else
  #define result1_4 result0_4
#endif

  // Unroll x4
#if DOTP32_NEON_SIZE_MULTIPLE >= 16
  do
#else
  while (count--)
#endif
  {
    int32x4_t u0_4, u1_4, u2_4, u3_4;
    int32x4_t v0_4, v1_4, v2_4, v3_4;
    int16x8_t v0_8, v1_8;

    // 0
    u0_4 = vld1q_s32(u);
    v0_8 = vld1q_s16(v);
    v0_4 = vmovl_s16( vget_low_s16(v0_8) );

    result0_4 = vmlaq_s32(result0_4, u0_4, v0_4);

    // 1
    u1_4 = vld1q_s32(u+4);
    v1_4 = vmovl_s16( vget_high_s16(v0_8) );

    result1_4 = vmlaq_s32(result1_4, u1_4, v1_4);

    // 2
    u2_4 = vld1q_s32(u+8);
    v1_8 = vld1q_s16(v+8);
    v2_4 = vmovl_s16( vget_low_s16(v1_8) );

    result0_4 = vmlaq_s32(result0_4, u2_4, v2_4);

    // 3
    u3_4 = vld1q_s32(u+12);
    v3_4 = vmovl_s16( vget_high_s16(v1_8) );

    result1_4 = vmlaq_s32(result1_4, u3_4, v3_4);

    // Next
    u += 16;
    v += 16;
  }
#if DOTP32_NEON_SIZE_MULTIPLE >= 16
  while (--count);
#endif

#if DOTP32_NEON_SIZE_MULTIPLE < 16
  // Unroll remaining x2
  if (n & 8)
  {
    int32x4_t u0_4, u1_4;
    int32x4_t v0_4, v1_4;
    int16x8_t v_8;

    // 0
    u0_4 = vld1q_s32(u);
    v_8  = vld1q_s16(v);
    v0_4 = vmovl_s16( vget_low_s16(v_8) );

    result0_4 = vmlaq_s32(result0_4, u0_4, v0_4);

    // 1
    u1_4 = vld1q_s32(u+4);
    v1_4 = vmovl_s16( vget_high_s16(v_8) );

    result1_4 = vmlaq_s32(result1_4, u1_4, v1_4);

    // Next
    u += 8;
    v += 8;
  }
#endif // DOTP3216_NEON_SIZE_MULTIPLE < 16

#if DOTP3216_NEON_SIZE_MULTIPLE < 8
  // Remaining > 4
  if (n & 4)
  {
    n &= 3;
    int32x4_t u_4, v_4;

    // 0    
    u_4 = vld1q_s32(u + n);
    v_4 = vmovl_s16(vld1_s16(v + n));

    result0_4 = vmlaq_s32(result0_4, u_4, v_4);
  }
#endif // DOTP3216_NEON_SIZE_MULTIPLE < 8
#ifdef DOTP3216_NEON_ACCU_2
  // Sum accumulators
  result0_4 = vaddq_s32(result0_4, result1_4);
#endif
  
  // Horizontal sum
  int64x2_t tmp0 = vpaddlq_s32(result0_4);
  int64x1_t tmp1 = vadd_s64(vget_high_s64(tmp0), vget_low_s64(tmp0));
  result = vget_lane_s32((int32x2_t)tmp1, 0);

#if DOTP3216_NEON_SIZE_MULTIPLE < 4
  // Remaining < 4
  switch (n & 3)
  {
    case 3: result += u[2] * v[2];
    case 2: result += u[1] * v[1];
    case 1: result += u[0] * v[0];
    default: break;
  }
#endif // DOTP3216_NEON_SIZE_MULTIPLE < 8
  
  return result;
}

#ifdef result1_4
  #undef result1_4
#endif


#endif // DOTP_I32I16_NEON_H
