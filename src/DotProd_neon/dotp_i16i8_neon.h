/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef DOTP_I16I8_NEON_H
#define DOTP_I16I8_NEON_H

#include <stdint.h>
#include <arm_neon.h>   // NEON

// SIMD optimization options
#ifndef DOTP168_NEON_SIZE_MULTIPLE
  #define DOTP168_NEON_SIZE_MULTIPLE 0  // 16, 8 (0: no optim)
#endif
#define DOTP168_NEON_ACCU_2             // Use 2 accumulators (depend on HW/vectors size)


//
static inline int32_t dotProduct_i16i8_neon_scalar(int16_t const* __restrict u, int8_t const* __restrict v, size_t n)
{
  int32_t res = 0;
  for (size_t i=0; i<n; ++i)
  res += u[i] * v[i];
  
  return res;
}

// 
static inline int32_t dotProduct_i16i8_neon_naive(int16_t const* __restrict u, int8_t const* __restrict v, size_t n)
{
  int32_t result;
  size_t count = n >> 3;
  
  // Accumulator
  int32x4_t result_4 = vdupq_n_s32(0);

  // Loop
  while (count--)
  {
    int16x8_t u_8, v_8;

    u_8 = vld1q_s16(u);
    v_8 = vmovl_s8(vld1_s8(v));

    result_4 = vmlal_s16(result_4, vget_low_s16( u_8), vget_low_s16( v_8));
    result_4 = vmlal_s16(result_4, vget_high_s16(u_8), vget_high_s16(v_8));
    
    // Next
    u += 8;
    v += 8;
  }
  
  // Horizontal sum
  int64x2_t tmp0 = vpaddlq_s32(result_4);
  int32x2_t tmp1 = vmovn_s64(tmp0);
  result = (int32_t)vget_lane_s64(vpaddl_s32(tmp1), 0);

#if DOTP168_NEON_SIZE_MULTIPLE < 8
  n &= 7;
  while (n--)
    result += u[n] * v[n];
#endif
  return result;
}

// 
static inline int32_t dotProduct_i16i8_neon(int16_t const* __restrict u, int8_t const* __restrict v, size_t n)
{
  int32_t result;
  size_t count = n >> 4;
  
  // Accumulators
  int32x4_t result0_4 = vdupq_n_s32(0);
#ifdef DOTP168_NEON_ACCU_2
  int32x4_t result1_4 = vdupq_n_s32(0);
#else
  #define result1_4 result0_4
#endif

  // Unroll x2
#if DOTP168_NEON_SIZE_MULTIPLE >= 16
  do
#else
  while (count--)
#endif
  {
    int16x8_t u0_8, u1_8;
    int16x8_t v0_8, v1_8;
    int8x16_t v_16;

    // 0
    u0_8 = vld1q_s16(u);
    v_16 = vld1q_s8(v);
    v0_8 = vmovl_s8( vget_low_s8(v_16) );

    result0_4 = vmlal_s16(result0_4, vget_low_s16( u0_8), vget_low_s16( v0_8));
    result1_4 = vmlal_s16(result1_4, vget_high_s16(u0_8), vget_high_s16(v0_8));

    // 1
    u1_8 = vld1q_s16(u+8);
    v1_8 = vmovl_s8( vget_high_s8(v_16) );

    result0_4 = vmlal_s16(result0_4, vget_low_s16( u1_8), vget_low_s16( v1_8));
    result1_4 = vmlal_s16(result1_4, vget_high_s16(u1_8), vget_high_s16(v1_8));

    // Next
    u += 16;
    v += 16;
  }
#if DOTP168_NEON_SIZE_MULTIPLE >= 16
  while (--count);
#endif

#if DOTP168_NEON_SIZE_MULTIPLE < 16
  // Remaining > 8
  if (n & 8)
  {
    n &= 7;
    int16x8_t u_8, v_8;
    
    u_8 = vld1q_s16(u + n);
    v_8 = vmovl_s8(vld1_s8(v + n));

    // 0
    result0_4 = vmlal_s16(result0_4, vget_low_s16( u_8), vget_low_s16( v_8));
    result1_4 = vmlal_s16(result1_4, vget_high_s16(u_8), vget_high_s16(v_8));
  }
#endif // DOTP168_NEON_SIZE_MULTIPLE < 16
#ifdef DOTP168_NEON_ACCU_2
  // Sum accumulators
  result0_4 = vaddq_s32(result0_4, result1_4);
#endif
  
  // Horizontal sum
  int64x2_t tmp0 = vpaddlq_s32(result0_4);
  int64x1_t tmp1 = vadd_s64(vget_high_s64(tmp0), vget_low_s64(tmp0));
  result = vget_lane_s32((int32x2_t)tmp1, 0);

#if DOTP168_NEON_SIZE_MULTIPLE < 8
  // Remaining < 8
  switch (n & 7)
  {
    case 7: result += u[6] * v[6];
    case 6: result += u[5] * v[5];
    case 5: result += u[4] * v[4];
    case 4: result += u[3] * v[3];
    case 3: result += u[2] * v[2];
    case 2: result += u[1] * v[1];
    case 1: result += u[0] * v[0];
    default: break;
  }
#endif // DOTP168_NEON_SIZE_MULTIPLE < 8
  
  return result;
}

#ifdef result1_4
  #undef result1_4
#endif


#endif // DOTP_I16I8_NEON_H
