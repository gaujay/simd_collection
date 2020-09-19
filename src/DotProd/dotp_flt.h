/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef DOTP_FLT_H
#define DOTP_FLT_H

#include "Utils/compiler_utils.h"
#include "Utils/simd_utils.h"

#include <stdint.h>
#include <emmintrin.h>    // SSE2
#ifdef HAS_SSSE3_
  #include <pmmintrin.h>  // SSE3
#endif
#ifdef HAS_AVX_
  #include <immintrin.h>  // AVX, FMA
#endif

// SIMD optimization options
#ifndef DOTPFLT_SIZE_MULTIPLE
  #define DOTPFLT_SIZE_MULTIPLE 0   // 32, 16, 8, 4 (0: no optim)
#endif
//#define DOTPFLT_ACCU_3   // Use 3/4 accumulators (depend on HW/vectors size)
//#define DOTPFLT_ACCU_4
#if defined(DOTPFLT_ACCU_4) && !defined(DOTPFLT_ACCU_3)  
  #define DOTPFLT_ACCU_3
#endif
#if defined DOTPFLT_256_ALIGNED
  #define DOTPFLT_LOAD_128(x) _mm_load_ps(x)
  #ifdef HAS_AVX_
    #define DOTPFLT_LOAD_256(x) _mm256_load_ps(x)
  #endif
#elif defined DOTPFLT_128_ALIGNED
  #define DOTPFLT_LOAD_128(x) _mm_load_ps(x)
  #ifdef HAS_AVX_
    #define DOTPFLT_LOAD_256(x) _mm256_loadu_ps(x)
  #endif
#else
  #define DOTPFLT_LOAD_128(x) _mm_loadu_ps(x)
  #ifdef HAS_AVX_
    #define DOTPFLT_LOAD_256(x) _mm256_loadu_ps(x)
  #endif
#endif


//
DISABLE_FUNC_VECTORIZATION_
static inline float dotProduct_flt_scalarforced(float const* __restrict u, float const* __restrict v, size_t n)
{
  float res = 0;
  DISABLE_LOOP_VECTORIZATION_
  for (size_t i=0; i<n; ++i)
    res += u[i] * v[i];
    
  return res;
}

//
static inline float dotProduct_flt_scalar(float const* __restrict u, float const* __restrict v, size_t n)
{
  float res = 0;
  for (size_t i=0; i<n; ++i)
    res += u[i] * v[i];
    
  return res;
}

//
#ifdef HAS_SSE3_
static inline float dotProduct_flt_sse_naive(float const* __restrict u, float const* __restrict v, size_t n)
{
  float res;
  size_t count = n >> 3;
  
  // Accumulator
  __m128 accu = _mm_setzero_ps();

  // Unroll x2
  while (count--)
  {
    __m128 u_4, v_4;
    __m128 mult;

    // 0
    u_4 = DOTPFLT_LOAD_128(u);
    v_4 = DOTPFLT_LOAD_128(v);

    mult = _mm_mul_ps(u_4, v_4);
    accu = _mm_add_ps(accu, mult);

    // 1
    u_4 = DOTPFLT_LOAD_128(u + 4);
    v_4 = DOTPFLT_LOAD_128(v + 4);

    mult = _mm_mul_ps(u_4, v_4);
    accu = _mm_add_ps(accu, mult);
    
    // Next
    u += 8;
    v += 8;
  }
  // Horizontal sum
  accu = _mm_hadd_ps(accu, accu); // SSE3
  accu = _mm_hadd_ps(accu, accu);
  res = _mm_cvtss_f32(accu);

#if DOTPFLT_SIZE_MULTIPLE < 8
  n &= 7;
  while (n--)
    res += u[n] * v[n];
#endif

  return res;
}
#endif // HAS_SSE3_

//
static inline float dotProduct_flt_sse(float const* __restrict u, float const* __restrict v, size_t n)
{
  float res;
  size_t count = n >> 4;
 
  // Accumulators
  __m128 accu0 = _mm_setzero_ps();
  __m128 accu1 = _mm_setzero_ps();
#ifdef DOTPFLT_ACCU_3
  __m128 accu2 = _mm_setzero_ps();
  #ifdef DOTPFLT_ACCU_4
    __m128 accu3 = _mm_setzero_ps();
  #else
    #define accu3 accu0
  #endif
#else
  #define accu2 accu0
  #define accu3 accu1
#endif

  // Unroll x4
  while (count--)
  {
    __m128 u0_4, u1_4, u2_4, u3_4;
    __m128 v0_4, v1_4, v2_4, v3_4;
    __m128 mult0, mult1, mult2, mult3;

    // 0
    u0_4 = DOTPFLT_LOAD_128(u);
    v0_4 = DOTPFLT_LOAD_128(v);

    mult0 = _mm_mul_ps(u0_4, v0_4);

    // 1
    u1_4 = DOTPFLT_LOAD_128(u + 4);
    v1_4 = DOTPFLT_LOAD_128(v + 4);

    mult1 = _mm_mul_ps(u1_4, v1_4);
   
    // 2
    u2_4 = DOTPFLT_LOAD_128(u + 8);
    v2_4 = DOTPFLT_LOAD_128(v + 8);

    mult2 = _mm_mul_ps(u2_4, v2_4);

    // 3
    u3_4 = DOTPFLT_LOAD_128(u + 12);
    v3_4 = DOTPFLT_LOAD_128(v + 12);

    mult3 = _mm_mul_ps(u3_4, v3_4);
   
    // Sum
    accu0 = _mm_add_ps(accu0, mult0);
    accu1 = _mm_add_ps(accu1, mult1);
    accu2 = _mm_add_ps(accu2, mult2);
    accu3 = _mm_add_ps(accu3, mult3);
   
    // Next
    u += 16;
    v += 16;
  }
#ifdef DOTPFLT_ACCU_4
  // Sum accumulators
  accu2 = _mm_add_ps(accu2, accu3);
#else
  #ifdef DOTPFLT_ACCU_3
  accu1 = _mm_add_ps(accu1, accu2);
  #endif
#endif
 
#if DOTPFLT_SIZE_MULTIPLE < 16
  // Unroll remaining x2
  if (n & 8)
  {
    __m128 u0_4, u1_4;
    __m128 v0_4, v1_4;
    __m128 mult0, mult1;

    // 0
    u0_4 = DOTPFLT_LOAD_128(u);
    v0_4 = DOTPFLT_LOAD_128(v);

    mult0 = _mm_mul_ps(u0_4, v0_4);

    // 1
    u1_4 = DOTPFLT_LOAD_128(u + 4);
    v1_4 = DOTPFLT_LOAD_128(v + 4);

    mult1 = _mm_mul_ps(u1_4, v1_4);

    // Sum
    accu0 = _mm_add_ps(accu0, mult0);
    accu1 = _mm_add_ps(accu1, mult1);
   
    // Next
    u += 8;
    v += 8;
  }
#endif // DOTPFLT_SIZE_MULTIPLE < 16
#ifdef DOTPFLT_ACCU_4
  // Sum accumulators
  accu1 = _mm_add_ps(accu1, accu2);
#endif

#if DOTPFLT_SIZE_MULTIPLE < 8
  // Remaining > 4
  if (n & 4)
  {
    n &= 3;
    __m128 u_4, v_4;
    __m128 mult;
 
    u_4 = DOTPFLT_LOAD_128(u + n);
    v_4 = DOTPFLT_LOAD_128(v + n);

    mult  = _mm_mul_ps(u_4, v_4);
    accu0 = _mm_add_ps(accu0, mult);
  }
#endif // DOTPFLT_SIZE_MULTIPLE < 8
  
  // Sum accumulators
  accu0 = _mm_add_ps(accu0, accu1);
  res = horizontal_sum_ps(accu0);
  
#if DOTPFLT_SIZE_MULTIPLE < 4
  // Remaining < 4
  switch (n & 3)
  {
    case 3: res += u[2] * v[2];
    case 2: res += u[1] * v[1];
    case 1: res += u[0] * v[0];
    default: break;
  }
#endif // DOTPFLT_SIZE_MULTIPLE < 4
  
  return res;
}

//
#ifdef HAS_AVX_
static inline float dotProduct_flt_avx(float const* __restrict u, float const* __restrict v, size_t n)
{
  float res;
  size_t count = n >> 5;
 
  // Accumulators
  __m256 accu0 = _mm256_setzero_ps();
  __m256 accu1 = _mm256_setzero_ps();
#ifdef DOTPFLT_ACCU_3
  __m256 accu2 = _mm256_setzero_ps();
  #ifdef DOTPFLT_ACCU_4
    __m256 accu3 = _mm256_setzero_ps();
  #else
    #define accu3 accu0
  #endif
#else
  #define accu2 accu0
  #define accu3 accu1
#endif

  // Unroll x4
  while (count--)
  {
    __m256 u0_8, u1_8, u2_8, u3_8;
    __m256 v0_8, v1_8, v2_8, v3_8;
    __m256 mult0, mult1, mult2, mult3;

    // 0
    u0_8 = DOTPFLT_LOAD_256(u);
    v0_8 = DOTPFLT_LOAD_256(v);

    mult0 = _mm256_mul_ps(u0_8, v0_8);

    // 1
    u1_8 = DOTPFLT_LOAD_256(u + 8);
    v1_8 = DOTPFLT_LOAD_256(v + 8);

    mult1 = _mm256_mul_ps(u1_8, v1_8);
   
    // 2
    u2_8 = DOTPFLT_LOAD_256(u + 16);
    v2_8 = DOTPFLT_LOAD_256(v + 16);

    mult2 = _mm256_mul_ps(u2_8, v2_8);

    // 3
    u3_8 = DOTPFLT_LOAD_256(u + 24);
    v3_8 = DOTPFLT_LOAD_256(v + 24);

    mult3 = _mm256_mul_ps(u3_8, v3_8);
   
    // Sum
    accu0 = _mm256_add_ps(accu0, mult0);
    accu1 = _mm256_add_ps(accu1, mult1);
    accu2 = _mm256_add_ps(accu2, mult2);
    accu3 = _mm256_add_ps(accu3, mult3);
   
    // Next
    u += 32;
    v += 32;
  }
#ifdef DOTPFLT_ACCU_4
  // Sum accumulators
  accu2 = _mm256_add_ps(accu2, accu3);
#else
  #ifdef DOTPFLT_ACCU_3
  accu1 = _mm256_add_ps(accu1, accu2);
  #endif
#endif
 
#if DOTPFLT_SIZE_MULTIPLE < 32
  // Unroll remaining x2
  if (n & 16)
  {
    __m256 u0_8, u1_8;
    __m256 v0_8, v1_8;
    __m256 mult0, mult1;

    // 0
    u0_8 = DOTPFLT_LOAD_256(u);
    v0_8 = DOTPFLT_LOAD_256(v);

    mult0 = _mm256_mul_ps(u0_8, v0_8);

    // 1
    u1_8 = DOTPFLT_LOAD_256(u + 8);
    v1_8 = DOTPFLT_LOAD_256(v + 8);

    mult1 = _mm256_mul_ps(u1_8, v1_8);

    // Sum
    accu0 = _mm256_add_ps(accu0, mult0);
    accu1 = _mm256_add_ps(accu1, mult1);
   
    // Next
    u += 16;
    v += 16;
  }
#endif // DOTPFLT_SIZE_MULTIPLE < 32
#ifdef DOTPFLT_ACCU_4
  // Sum accumulators
  accu1 = _mm256_add_ps(accu1, accu2);
#endif

#if DOTPFLT_SIZE_MULTIPLE < 16
  // Remaining > 8
  if (n & 8)
  {
    n &= 7;
    __m256 u_8, v_8;
    __m256 mult;
 
    u_8 = DOTPFLT_LOAD_256(u + n);
    v_8 = DOTPFLT_LOAD_256(v + n);

    mult  = _mm256_mul_ps(u_8, v_8);
    accu0 = _mm256_add_ps(accu0, mult);
  }
#endif // DOTPFLT_SIZE_MULTIPLE < 16
  
  // Sum accumulators
  accu0 = _mm256_add_ps(accu0, accu1);
  res = horizontal_sum_ps(accu0);
 
#if DOTPFLT_SIZE_MULTIPLE < 8
  // Remaining < 8
  switch (n & 7)
  {
    case  7: res += u[6] * v[6];
    case  6: res += u[5] * v[5];
    case  5: res += u[4] * v[4];
    case  4: res += u[3] * v[3];
    case  3: res += u[2] * v[2];
    case  2: res += u[1] * v[1];
    case  1: res += u[0] * v[0];
    default: break;
  }
#endif // DOTPFLT_SIZE_MULTIPLE < 8
  
  return res;
}
#endif // HAS_AVX_

#ifdef HAS_FMA_
static inline float dotProduct_flt_fma(float const* __restrict u, float const* __restrict v, size_t n)
{
  float res;
  size_t count = n >> 5;
 
  // Accumulators
  __m256 accu0 = _mm256_setzero_ps();
  __m256 accu1 = _mm256_setzero_ps();
#ifdef DOTPFLT_ACCU_3
  __m256 accu2 = _mm256_setzero_ps();
  #ifdef DOTPFLT_ACCU_4
    __m256 accu3 = _mm256_setzero_ps();
  #else
    #define accu3 accu0
  #endif
#else
  #define accu2 accu0
  #define accu3 accu1
#endif

  // Unroll x4
  while (count--)
  {
    __m256 u0_8, u1_8, u2_8, u3_8;
    __m256 v0_8, v1_8, v2_8, v3_8;

    // 0
    u0_8 = DOTPFLT_LOAD_256(u);
    v0_8 = DOTPFLT_LOAD_256(v);

    accu0 = _mm256_fmadd_ps(u0_8, v0_8, accu0);

    // 1
    u1_8 = DOTPFLT_LOAD_256(u + 8);
    v1_8 = DOTPFLT_LOAD_256(v + 8);

    accu1 = _mm256_fmadd_ps(u1_8, v1_8, accu1);
   
    // 2
    u2_8 = DOTPFLT_LOAD_256(u + 16);
    v2_8 = DOTPFLT_LOAD_256(v + 16);

    accu2 = _mm256_fmadd_ps(u2_8, v2_8, accu2);

    // 3
    u3_8 = DOTPFLT_LOAD_256(u + 24);
    v3_8 = DOTPFLT_LOAD_256(v + 24);

    accu3 = _mm256_fmadd_ps(u3_8, v3_8, accu3);
    
    // Next
    u += 32;
    v += 32;
  }
#ifdef DOTPFLT_ACCU_4
  // Sum accumulators
  accu2 = _mm256_add_ps(accu2, accu3);
#else
  #ifdef DOTPFLT_ACCU_3
  accu1 = _mm256_add_ps(accu1, accu2);
  #endif
#endif
 
#if DOTPFLT_SIZE_MULTIPLE < 32
  // Unroll remaining x2
  if (n & 16)
  {
    __m256 u0_8, u1_8;
    __m256 v0_8, v1_8;

    // 0
    u0_8 = DOTPFLT_LOAD_256(u);
    v0_8 = DOTPFLT_LOAD_256(v);

    accu0 = _mm256_fmadd_ps(u0_8, v0_8, accu0);

    // 1
    u1_8 = DOTPFLT_LOAD_256(u + 8);
    v1_8 = DOTPFLT_LOAD_256(v + 8);

    accu1 = _mm256_fmadd_ps(u1_8, v1_8, accu1);
   
    // Next
    u += 16;
    v += 16;
  }
#endif // DOTPFLT_SIZE_MULTIPLE < 32
#ifdef DOTPFLT_ACCU_4
  // Sum accumulators
  accu1 = _mm256_add_ps(accu1, accu2);
#endif

#if DOTPFLT_SIZE_MULTIPLE < 16
  // Remaining > 8
  if (n & 8)
  {
    n &= 7;
    __m256 u_8, v_8;
 
    u_8 = DOTPFLT_LOAD_256(u + n);
    v_8 = DOTPFLT_LOAD_256(v + n);

    accu0 = _mm256_fmadd_ps(u_8, v_8, accu0);
  }
#endif // DOTPFLT_SIZE_MULTIPLE < 16
  
  // Sum accumulators
  accu0 = _mm256_add_ps(accu0, accu1);
  res = horizontal_sum_ps(accu0);
  
#if DOTPFLT_SIZE_MULTIPLE < 8
  // Remaining < 8
  switch (n & 7)
  {
    case  7: res += u[6] * v[6];
    case  6: res += u[5] * v[5];
    case  5: res += u[4] * v[4];
    case  4: res += u[3] * v[3];
    case  3: res += u[2] * v[2];
    case  2: res += u[1] * v[1];
    case  1: res += u[0] * v[0];
    default: break;
  }
#endif // DOTPFLT_SIZE_MULTIPLE < 8
  
  return res;
}
#endif // HAS_FMA_

#ifdef accu2
  #undef accu2
#endif
#ifdef accu3
  #undef accu3
#endif

#endif // DOTP_FLT_H
