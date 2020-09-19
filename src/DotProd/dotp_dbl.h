/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef DOTP_DBL_H
#define DOTP_DBL_H

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
#ifndef DOTPDBL_SIZE_MULTIPLE
  #define DOTPDBL_SIZE_MULTIPLE 0   // 16, 8, 4, 2 (0: no optim)
#endif
//#define DOTPDBL_ACCU_3   // Use 3/4 accumulators (depend on HW/vectors size)
//#define DOTPDBL_ACCU_4
#if defined(DOTPDBL_ACCU_4) && !defined(DOTPDBL_ACCU_3)  
  #define DOTPDBL_ACCU_3
#endif
#if defined DOTPDBL_256_ALIGNED
  #define DOTPDBL_LOAD_128(x) _mm_load_pd(x)
  #ifdef HAS_AVX_
    #define DOTPDBL_LOAD_256(x) _mm256_load_pd(x)
  #endif
#elif defined DOTPDBL_128_ALIGNED
  #define DOTPDBL_LOAD_128(x) _mm_load_pd(x)
  #ifdef HAS_AVX_
    #define DOTPDBL_LOAD_256(x) _mm256_loadu_pd(x)
  #endif
#else
  #define DOTPDBL_LOAD_128(x) _mm_loadu_pd(x)
  #ifdef HAS_AVX_
    #define DOTPDBL_LOAD_256(x) _mm256_loadu_pd(x)
  #endif
#endif


//
DISABLE_FUNC_VECTORIZATION_
static inline double dotProduct_dbl_scalarforced(double const* __restrict u, double const* __restrict v, size_t n)
{
  double res = 0;
  DISABLE_LOOP_VECTORIZATION_
  for (size_t i=0; i<n; ++i)
    res += u[i] * v[i];
    
  return res;
}

//
static inline double dotProduct_dbl_scalar(double const* __restrict u, double const* __restrict v, size_t n)
{
  double res = 0;
  for (size_t i=0; i<n; ++i)
    res += u[i] * v[i];
    
  return res;
}

//
#ifdef HAS_SSE3_
static inline double dotProduct_dbl_sse_naive(double const* __restrict u, double const* __restrict v, size_t n)
{
  double res;
  size_t count = n >> 2;
  
  // Accumulator
  __m128d accu = _mm_setzero_pd();

  // Unroll x2
  while (count--)
  {
    __m128d u_2, v_2;
    __m128d mult;

    // 0
    u_2 = DOTPDBL_LOAD_128(u);
    v_2 = DOTPDBL_LOAD_128(v);

    mult = _mm_mul_pd(u_2, v_2);
    accu = _mm_add_pd(accu, mult);

    // 1
    u_2 = DOTPDBL_LOAD_128(u + 2);
    v_2 = DOTPDBL_LOAD_128(v + 2);

    mult = _mm_mul_pd(u_2, v_2);
    accu = _mm_add_pd(accu, mult);
    
    // Next
    u += 4;
    v += 4;
  }
  // Horizontal sum
  accu = _mm_hadd_pd(accu, accu); // SSE3
  res = _mm_cvtsd_f64(accu);

#if DOTPDBL_SIZE_MULTIPLE < 4
  n &= 3;
  while (n--)
    res += u[n] * v[n];
#endif

  return res;
}
#endif // HAS_SSE3_

//
static inline double dotProduct_dbl_sse(double const* __restrict u, double const* __restrict v, size_t n)
{
  double res;
  size_t count = n >> 3;
 
  // Accumulators
  __m128d accu0 = _mm_setzero_pd();
  __m128d accu1 = _mm_setzero_pd();
#ifdef DOTPDBL_ACCU_3
  __m128d accu2 = _mm_setzero_pd();
  #ifdef DOTPDBL_ACCU_4
    __m128d accu3 = _mm_setzero_pd();
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
    __m128d u0_2, u1_2, u2_2, u3_2;
    __m128d v0_2, v1_2, v2_2, v3_2;
    __m128d mult0, mult1, mult2, mult3;

    // 0
    u0_2 = DOTPDBL_LOAD_128(u);
    v0_2 = DOTPDBL_LOAD_128(v);

    mult0 = _mm_mul_pd(u0_2, v0_2);

    // 1
    u1_2 = DOTPDBL_LOAD_128(u + 2);
    v1_2 = DOTPDBL_LOAD_128(v + 2);

    mult1 = _mm_mul_pd(u1_2, v1_2);
   
    // 2
    u2_2 = DOTPDBL_LOAD_128(u + 4);
    v2_2 = DOTPDBL_LOAD_128(v + 4);

    mult2 = _mm_mul_pd(u2_2, v2_2);

    // 3
    u3_2 = DOTPDBL_LOAD_128(u + 6);
    v3_2 = DOTPDBL_LOAD_128(v + 6);

    mult3 = _mm_mul_pd(u3_2, v3_2);
   
    // Sum
    accu0 = _mm_add_pd(accu0, mult0);
    accu1 = _mm_add_pd(accu1, mult1);
    accu2 = _mm_add_pd(accu2, mult2);
    accu3 = _mm_add_pd(accu3, mult3);
   
    // Next
    u += 8;
    v += 8;
  }
#ifdef DOTPDBL_ACCU_4
  // Sum accumulators
  accu2 = _mm_add_pd(accu2, accu3);
#else
  #ifdef DOTPDBL_ACCU_3
  accu1 = _mm_add_pd(accu1, accu2);
  #endif
#endif
 
#if DOTPDBL_SIZE_MULTIPLE < 8
  // Unroll remaining x2
  if (n & 4)
  {
    __m128d u0_2, u1_2;
    __m128d v0_2, v1_2;
    __m128d mult0, mult1;

    // 0
    u0_2 = DOTPDBL_LOAD_128(u);
    v0_2 = DOTPDBL_LOAD_128(v);

    mult0 = _mm_mul_pd(u0_2, v0_2);

    // 1
    u1_2 = DOTPDBL_LOAD_128(u + 2);
    v1_2 = DOTPDBL_LOAD_128(v + 2);

    mult1 = _mm_mul_pd(u1_2, v1_2);

    // Sum
    accu0 = _mm_add_pd(accu0, mult0);
    accu1 = _mm_add_pd(accu1, mult1);
   
    // Next
    u += 4;
    v += 4;
  }
#endif // DOTPDBL_SIZE_MULTIPLE < 16
#ifdef DOTPDBL_ACCU_4
  // Sum accumulators
  accu1 = _mm_add_pd(accu1, accu2);
#endif

#if DOTPDBL_SIZE_MULTIPLE < 4
  // Remaining > 2
  if (n & 2)
  {
    n &= 1;
    __m128d u_2, v_2;
    __m128d mult;
 
    u_2 = DOTPDBL_LOAD_128(u + n);
    v_2 = DOTPDBL_LOAD_128(v + n);

    mult  = _mm_mul_pd(u_2, v_2);
    accu0 = _mm_add_pd(accu0, mult);
  }
#endif // DOTPDBL_SIZE_MULTIPLE < 4
  
  // Sum accumulators
  accu0 = _mm_add_pd(accu0, accu1);
  res = horizontal_sum_pd(accu0);
  
#if DOTPDBL_SIZE_MULTIPLE < 2
  // Remaining < 2
  if (n & 1)
    res += u[0] * v[0];
#endif // DOTPDBL_SIZE_MULTIPLE < 2  
  
  return res;
}

//
#ifdef HAS_AVX_
static inline double dotProduct_dbl_avx(double const* __restrict u, double const* __restrict v, size_t n)
{
  double res;
  size_t count = n >> 4;
 
  // Accumulators
  __m256d accu0 = _mm256_setzero_pd();
  __m256d accu1 = _mm256_setzero_pd();
#ifdef DOTPDBL_ACCU_3
  __m256d accu2 = _mm256_setzero_pd();
  #ifdef DOTPDBL_ACCU_4
    __m256d accu3 = _mm256_setzero_pd();
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
    __m256d u0_4, u1_4, u2_4, u3_4;
    __m256d v0_4, v1_4, v2_4, v3_4;
    __m256d mult0, mult1, mult2, mult3;

    // 0
    u0_4 = DOTPDBL_LOAD_256(u);
    v0_4 = DOTPDBL_LOAD_256(v);

    mult0 = _mm256_mul_pd(u0_4, v0_4);

    // 1
    u1_4 = DOTPDBL_LOAD_256(u + 4);
    v1_4 = DOTPDBL_LOAD_256(v + 4);

    mult1 = _mm256_mul_pd(u1_4, v1_4);
   
    // 2
    u2_4 = DOTPDBL_LOAD_256(u + 8);
    v2_4 = DOTPDBL_LOAD_256(v + 8);

    mult2 = _mm256_mul_pd(u2_4, v2_4);

    // 3
    u3_4 = DOTPDBL_LOAD_256(u + 12);
    v3_4 = DOTPDBL_LOAD_256(v + 12);

    mult3 = _mm256_mul_pd(u3_4, v3_4);
   
    // Sum
    accu0 = _mm256_add_pd(accu0, mult0);
    accu1 = _mm256_add_pd(accu1, mult1);
    accu2 = _mm256_add_pd(accu2, mult2);
    accu3 = _mm256_add_pd(accu3, mult3);
   
    // Next
    u += 16;
    v += 16;
  }
#ifdef DOTPDBL_ACCU_4
  // Sum accumulators
  accu2 = _mm256_add_pd(accu2, accu3);
#else
  #ifdef DOTPDBL_ACCU_3
  accu1 = _mm256_add_pd(accu1, accu2);
  #endif
#endif
 
#if DOTPDBL_SIZE_MULTIPLE < 16
  // Unroll remaining x2
  if (n & 8)
  {
    __m256d u0_4, u1_4;
    __m256d v0_4, v1_4;
    __m256d mult0, mult1;

    // 0
    u0_4 = DOTPDBL_LOAD_256(u);
    v0_4 = DOTPDBL_LOAD_256(v);

    mult0 = _mm256_mul_pd(u0_4, v0_4);

    // 1
    u1_4 = DOTPDBL_LOAD_256(u + 4);
    v1_4 = DOTPDBL_LOAD_256(v + 4);

    mult1 = _mm256_mul_pd(u1_4, v1_4);

    // Sum
    accu0 = _mm256_add_pd(accu0, mult0);
    accu1 = _mm256_add_pd(accu1, mult1);
   
    // Next
    u += 8;
    v += 8;
  }
#endif // DOTPDBL_SIZE_MULTIPLE < 16
#ifdef DOTPDBL_ACCU_4
  // Sum accumulators
  accu1 = _mm256_add_pd(accu1, accu2);
#endif

#if DOTPDBL_SIZE_MULTIPLE < 8
  // Remaining > 4
  if (n & 4)
  {
    n &= 3;
    __m256d u_4, v_4;
    __m256d mult;
 
    u_4 = DOTPDBL_LOAD_256(u + n);
    v_4 = DOTPDBL_LOAD_256(v + n);

    mult  = _mm256_mul_pd(u_4, v_4);
    accu0 = _mm256_add_pd(accu0, mult);
  }
#endif // DOTPDBL_SIZE_MULTIPLE < 8
  
  // Sum accumulators
  accu0 = _mm256_add_pd(accu0, accu1);
  res = horizontal_sum_pd(accu0);
  
#if DOTPDBL_SIZE_MULTIPLE < 4
  // Remaining < 4
  switch (n & 3)
  {
    case  3: res += u[2] * v[2];
    case  2: res += u[1] * v[1];
    case  1: res += u[0] * v[0];
    default: break;
  }
#endif // DOTPDBL_SIZE_MULTIPLE < 4
  
  return res;
}
#endif // HAS_AVX_

#ifdef HAS_FMA_
static inline double dotProduct_dbl_fma(double const* __restrict u, double const* __restrict v, size_t n)
{
  double res;
  size_t count = n >> 4;
 
  // Accumulators
  __m256d accu0 = _mm256_setzero_pd();
  __m256d accu1 = _mm256_setzero_pd();
#ifdef DOTPDBL_ACCU_3
  __m256d accu2 = _mm256_setzero_pd();
  #ifdef DOTPDBL_ACCU_4
    __m256d accu3 = _mm256_setzero_pd();
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
    __m256d u0_4, u1_4, u2_4, u3_4;
    __m256d v0_4, v1_4, v2_4, v3_4;

    // 0
    u0_4 = DOTPDBL_LOAD_256(u);
    v0_4 = DOTPDBL_LOAD_256(v);

    accu0 = _mm256_fmadd_pd(u0_4, v0_4, accu0);

    // 1
    u1_4 = DOTPDBL_LOAD_256(u + 4);
    v1_4 = DOTPDBL_LOAD_256(v + 4);

    accu1 = _mm256_fmadd_pd(u1_4, v1_4, accu1);
   
    // 2
    u2_4 = DOTPDBL_LOAD_256(u + 8);
    v2_4 = DOTPDBL_LOAD_256(v + 8);

    accu2 = _mm256_fmadd_pd(u2_4, v2_4, accu2);

    // 3
    u3_4 = DOTPDBL_LOAD_256(u + 12);
    v3_4 = DOTPDBL_LOAD_256(v + 12);

    accu3 = _mm256_fmadd_pd(u3_4, v3_4, accu3);
    
    // Next
    u += 16;
    v += 16;
  }
#ifdef DOTPDBL_ACCU_4
  // Sum accumulators
  accu2 = _mm256_add_pd(accu2, accu3);
#else
  #ifdef DOTPDBL_ACCU_3
  accu1 = _mm256_add_pd(accu1, accu2);
  #endif
#endif
 
#if DOTPDBL_SIZE_MULTIPLE < 16
  // Unroll remaining x2
  if (n & 8)
  {
    __m256d u0_4, u1_4;
    __m256d v0_4, v1_4;

    // 0
    u0_4 = DOTPDBL_LOAD_256(u);
    v0_4 = DOTPDBL_LOAD_256(v);

    accu0 = _mm256_fmadd_pd(u0_4, v0_4, accu0);

    // 1
    u1_4 = DOTPDBL_LOAD_256(u + 4);
    v1_4 = DOTPDBL_LOAD_256(v + 4);

    accu1 = _mm256_fmadd_pd(u1_4, v1_4, accu1);
   
    // Next
    u += 8;
    v += 8;
  }
#endif // DOTPDBL_SIZE_MULTIPLE < 16
#ifdef DOTPDBL_ACCU_4
  // Sum accumulators
  accu1 = _mm256_add_pd(accu1, accu2);
#endif

#if DOTPDBL_SIZE_MULTIPLE < 8
  // Remaining > 4
  if (n & 4)
  {
    n &= 3;
    __m256d u_4, v_4;
 
    u_4 = DOTPDBL_LOAD_256(u + n);
    v_4 = DOTPDBL_LOAD_256(v + n);

    accu0 = _mm256_fmadd_pd(u_4, v_4, accu0);
  }
#endif // DOTPDBL_SIZE_MULTIPLE < 8
  
  // Sum accumulators
  accu0 = _mm256_add_pd(accu0, accu1);
  res = horizontal_sum_pd(accu0);
  
#if DOTPDBL_SIZE_MULTIPLE < 4
  // Remaining < 4
  switch (n & 3)
  {
    case  3: res += u[2] * v[2];
    case  2: res += u[1] * v[1];
    case  1: res += u[0] * v[0];
    default: break;
  }
#endif // DOTPDBL_SIZE_MULTIPLE < 4
  
  return res;
}
#endif // HAS_FMA_

#ifdef accu2
  #undef accu2
#endif
#ifdef accu3
  #undef accu3
#endif

#endif // DOTP_DBL_H
