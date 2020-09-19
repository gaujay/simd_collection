/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef DOTP_FLT_NEON_H
#define DOTP_FLT_NEON_H

#include <stdint.h>
#include <arm_neon.h>   // NEON

// SIMD optimization options
#ifndef DOTPFLT_NEON_SIZE_MULTIPLE
  #define DOTPFLT_NEON_SIZE_MULTIPLE 0   // 16, 8, 4 (0: no optim)
#endif
//#define DOTPFLT_NEON_ACCU_2            // Use 2 accumulators (depend on HW/vectors size)


//
static inline float dotProduct_flt_neon_scalar(float const* __restrict u, float const* __restrict v, size_t n)
{
  float res = 0;
  for (size_t i=0; i<n; ++i)
    res += u[i] * v[i];
    
  return res;
}

// 
static inline float dotProduct_flt_neon_naive(float const* __restrict u, float const* __restrict v, size_t n)
{
  float result;
  size_t count = n >> 2;
  
  // Accumulators
  float32x4_t result_4 = vdupq_n_f32(0);

  // Loop
  while (count--)
  {
    float32x4_t u_4, v_4;

    u_4 = vld1q_f32(u);
    v_4 = vld1q_f32(v);
    
    result_4 = vmlaq_f32(result_4, u_4, v_4);
    
    // Next
    u += 4;
    v += 4;
  }

  // Horizontal sum
  float32x2_t tmp = vpadd_f32(vget_low_f32(result_4), vget_high_f32(result_4));
  result = vget_lane_f32(vpadd_f32(tmp, tmp), 0);

#if DOTPFLT_NEON_SIZE_MULTIPLE < 4
  n &= 3;
  while (n--)
    result += u[n] * v[n];
#endif
  return result;
}

//
static inline float dotProduct_flt_neon(float const* __restrict u, float const* __restrict v, size_t n)
{
  float result;
  size_t count = n >> 4;
  
  // Accumulators
  float32x4_t result0_4 = vdupq_n_f32(0);
#ifdef DOTPFLT_NEON_ACCU_2
  float32x4_t result1_4 = vdupq_n_f32(0);
#else
  #define result1_4 result0_4
#endif

  // Unroll x4
#if DOTPFLT_NEON_SIZE_MULTIPLE >= 16
  do
#else
  while (count--)
#endif
  {
    float32x4_t u0_4, u1_4, u2_4, u3_4;
    float32x4_t v0_4, v1_4, v2_4, v3_4;

    // 0
    u0_4 = vld1q_f32(u);
    v0_4 = vld1q_f32(v);

    result0_4 = vmlaq_f32(result0_4, u0_4, v0_4);

    // 1
    u1_4 = vld1q_f32(u+4);
    v1_4 = vld1q_f32(v+4);

    result1_4 = vmlaq_f32(result1_4, u1_4, v1_4);

    // 2
    u2_4 = vld1q_f32(u+8);
    v2_4 = vld1q_f32(v+8);

    result0_4 = vmlaq_f32(result0_4, u2_4, v2_4);

    // 3
    u3_4 = vld1q_f32(u+12);
    v3_4 = vld1q_f32(v+12);

    result1_4 = vmlaq_f32(result1_4, u3_4, v3_4);

    // Next
    u += 16;
    v += 16;
  }
#if DOTPFLT_NEON_SIZE_MULTIPLE >= 16
  while (--count);
#endif

#if DOTPFLT_NEON_SIZE_MULTIPLE < 16
  // Unroll remaining x2
  if (n & 8)
  {
    float32x4_t u0_4, u1_4;
    float32x4_t v0_4, v1_4;

    // 0
    u0_4 = vld1q_f32(u);
    v0_4 = vld1q_f32(v);

    result0_4 = vmlaq_f32(result0_4, u0_4, v0_4);

    // 1
    u1_4 = vld1q_f32(u+4);
    v1_4 = vld1q_f32(v+4);

    result1_4 = vmlaq_f32(result1_4, u1_4, v1_4);

    u += 8;
    v += 8;
  }
#endif // DOTPFLT_NEON_SIZE_MULTIPLE < 16

#if DOTPFLT_NEON_SIZE_MULTIPLE < 8
  // Remaining > 4
  if (n & 4)
  {
    n &= 3;
    float32x4_t u_4, v_4;

    // 0
    u_4 = vld1q_f32(u + n);
    v_4 = vld1q_f32(v + n);

    result0_4 = vmlaq_f32(result0_4, u_4, v_4);
  }
#endif // DOTPFLT_NEON_SIZE_MULTIPLE < 8
#ifdef DOTPFLT_NEON_ACCU_2
  // Sum accumulators
  result0_4 = vaddq_f32(result0_4, result1_4);
#endif

  // Horizontal sum
  float32x2_t tmp = vpadd_f32(vget_low_f32(result0_4), vget_high_f32(result0_4));
  result = vget_lane_f32(vpadd_f32(tmp, tmp), 0);

#if DOTPFLT_NEON_SIZE_MULTIPLE < 4
  // Remaining < 4
  switch (n & 3)
  {
    case 3: result += u[2] * v[2];
    case 2: result += u[1] * v[1];
    case 1: result += u[0] * v[0];
    default: break;
  }
#endif // DOTPFLT_NEON_SIZE_MULTIPLE < 4

  return result;
}

#ifdef result1_4
  #undef result1_4
#endif


#endif // DOTP_FLT_NEON_H
