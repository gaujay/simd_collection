/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef DOTP_I32I16_H
#define DOTP_I32I16_H

#include "Utils/compiler_utils.h"
#include "Utils/simd_utils.h"

#include <stdint.h>
#include <emmintrin.h>    // SSE2
#ifdef HAS_SSSE3_
  #include <tmmintrin.h>  // SSSE3
#endif
#ifdef HAS_SSE4_1_
  #include <smmintrin.h>  // SSE4.1
#endif
#ifdef HAS_AVX2_
  #include <immintrin.h>  // AVX2
#endif

// SIMD optimization options
#ifndef DOTP3216_SIZE_MULTIPLE
  #define DOTP3216_SIZE_MULTIPLE 0   // 32, 16, 8, 4 (0: no optim)
#endif
#if defined DOTP3216_256_ALIGNED
  #define DOTP3216_LOAD_64(x)  _mm_loadl_epi64((__m128i const*)(x))
  #define DOTP3216_LOAD_128(x) _mm_load_si128((__m128i const*)(x))
  #ifdef HAS_AVX2_
    #define DOTP3216_LOAD_256(x) _mm256_load_si256((__m256i const*)(x))
  #endif
#elif defined DOTP3216_128_ALIGNED
  #define DOTP3216_LOAD_64(x)  _mm_loadl_epi64((__m128i const*)(x))
  #define DOTP3216_LOAD_128(x) _mm_load_si128((__m128i const*)(x))
  #ifdef HAS_AVX2_
    #define DOTP3216_LOAD_256(x) _mm256_loadu_si256((__m256i const*)(x))
  #endif
#else
  #define DOTP3216_LOAD_64(x)  _mm_loadl_epi64((__m128i const*)(x))
  #define DOTP3216_LOAD_128(x) _mm_loadu_si128((__m128i const*)(x))
  #ifdef HAS_AVX2_
    #define DOTP3216_LOAD_256(x) _mm256_loadu_si256((__m256i const*)(x))
  #endif
#endif


//
DISABLE_FUNC_VECTORIZATION_
static inline int32_t dotProduct_i32i16_scalarforced(int32_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
  int32_t res = 0;
  DISABLE_LOOP_VECTORIZATION_
  for (size_t i=0; i<n; ++i)
    res += u[i] * v[i];
  
  return res;
}

//
static inline int32_t dotProduct_i32i16_scalar(int32_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
  int32_t res = 0;
  for (size_t i=0; i<n; ++i)
    res += u[i] * v[i];
  
  return res;
}

//
#ifdef HAS_SSSE3_
static inline int32_t dotProduct_i32i16_sse_naive(int32_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
  int32_t res;
  size_t count = n >> 3;
  
  // Accumulator
  __m128i accu = _mm_setzero_si128();

  // Loop
  while (count--)
  {
    __m128i u_4, v_4;
    __m128i v_8;
    __m128i mult;

    // 0
    u_4 = DOTP3216_LOAD_128(u);
    v_8 = DOTP3216_LOAD_128(v);

    v_4 = _mm_unpacklo_epi16(v_8, v_8);
    v_4 = _mm_srai_epi32(v_4, 16);
    
    mult = multiply_lo_epi32(u_4, v_4);
    accu = _mm_add_epi32(accu, mult);
    
    // 1
    u_4 = DOTP3216_LOAD_128(u + 4);

    v_4 = _mm_unpackhi_epi16(v_8, v_8);
    v_4 = _mm_srai_epi32(v_4, 16);
    
    mult = multiply_lo_epi32(u_4, v_4);
    accu = _mm_add_epi32(accu, mult);

    // Next
    u += 8;
    v += 8;
  }
  // Horizontal sum
  accu = _mm_hadd_epi32(accu, accu); // SSSE3
  accu = _mm_hadd_epi32(accu, accu);
  res = _mm_cvtsi128_si32(accu);

#if DOTP3216_SIZE_MULTIPLE < 8
  n &= 7;
  while (n--)
    res += u[n] * v[n];
#endif

  return res;
}
#endif // HAS_SSSE3_

//
static inline int32_t dotProduct_i32i16_sse(int32_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
  int32_t res;
  size_t count = n >> 4;
  
  // Accumulators
  __m128i accu0 = _mm_setzero_si128();
  __m128i accu1 = _mm_setzero_si128();

  // Unroll x4
  while (count--)
  {
    __m128i u0_4, u1_4;
    __m128i v0_4, v1_4;
    __m128i v0_8, v1_8;
    __m128i mult0, mult1, mult2, mult3;

    // 0
    u0_4 = DOTP3216_LOAD_128(u);
    v0_8 = DOTP3216_LOAD_128(v);
    v0_4 = extend_lo_epi16(v0_8);
    
    mult0 = multiply_lo_epi32(u0_4, v0_4);

    // 1
    u1_4 = DOTP3216_LOAD_128(u + 4);
    v1_4 = extend_hi_epi16(v0_8);
    
    mult1 = multiply_lo_epi32(u1_4, v1_4);
    
    // 2
    u0_4 = DOTP3216_LOAD_128(u + 8);
    v1_8 = DOTP3216_LOAD_128(v + 8);
    v0_4 = extend_lo_epi16(v1_8);
    
    mult2 = multiply_lo_epi32(u0_4, v0_4);

    // 3
    u1_4 = DOTP3216_LOAD_128(u + 12);
    v1_4 = extend_hi_epi16(v1_8);
    
    mult3 = multiply_lo_epi32(u1_4, v1_4);

    // Sum
    accu0 = _mm_add_epi32(accu0, mult0);
    accu1 = _mm_add_epi32(accu1, mult1);
    accu0 = _mm_add_epi32(accu0, mult2);
    accu1 = _mm_add_epi32(accu1, mult3);

    // Next
    u += 16;
    v += 16;
  }
  
//#if DOTP3216_SIZE_MULTIPLE < 16  // Not sure why but slower when true
  // Unroll remaining x2
  if (n & 8)
  {
    __m128i u_4, v_4;
    __m128i v_8;
    __m128i mult0, mult1;

    // 0
    u_4 = DOTP3216_LOAD_128(u);
    v_8 = DOTP3216_LOAD_128(v);
    v_4 = extend_lo_epi16(v_8);
    
    mult0 = multiply_lo_epi32(u_4, v_4);

    // 1
    u_4 = DOTP3216_LOAD_128(u + 4);
    v_4 = extend_hi_epi16(v_8);
    
    mult1 = multiply_lo_epi32(u_4, v_4);

    // Sum
    accu0 = _mm_add_epi32(accu0, mult0);
    accu1 = _mm_add_epi32(accu1, mult1);

    // Next
    u += 8;
    v += 8;
  }

#if DOTP3216_SIZE_MULTIPLE < 8
  // Remaining > 4
  if (n & 4)
  {
    n &= 3;
    __m128i u_4, v_4;
    __m128i v_8;
    __m128i mult;
    
    u_4 = DOTP3216_LOAD_128(u + n);
    v_8 = DOTP3216_LOAD_64 (v + n);
    v_4 = extend_lo_epi16(v_8);
    
    mult  = multiply_lo_epi32(u_4, v_4);
    accu0 = _mm_add_epi32(accu0, mult);
  }
#endif // DOTP3216_SIZE_MULTIPLE < 8
//#endif // DOTP3216_SIZE_MULTIPLE < 16
  
  // Sum accumulators
  accu0 = _mm_add_epi32(accu0, accu1);
  res = horizontal_sum_epi32(accu0);
  
#if DOTP3216_SIZE_MULTIPLE < 4
  // Remaining < 4
  switch (n & 3)
  {
    case 3: res += u[2] * v[2];
    case 2: res += u[1] * v[1];
    case 1: res += u[0] * v[0];
    default: break;
  }
#endif // DOTP3216_SIZE_MULTIPLE < 4
  
  return res;
}

//
#ifdef HAS_AVX2_
static inline int32_t dotProduct_i32i16_avx2(int32_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
  int32_t res;
  size_t count = n >> 5;
  
  // Accumulators
  __m256i accu0 = _mm256_setzero_si256();
  __m256i accu1 = _mm256_setzero_si256();

  // Unroll x4
  while (count--)
  {
    __m256i u0_8, u1_8;
    __m256i v0_8, v1_8;
    __m256i mult0, mult1, mult2, mult3;

    // 0
    u0_8  = DOTP3216_LOAD_256(u);
    v0_8 = _mm256_cvtepi16_epi32(DOTP3216_LOAD_128(v)); //TODO: load 256 + extract ?
    
    mult0 = multiply_lo_epi32(u0_8, v0_8);

    // 1
    u1_8  = DOTP3216_LOAD_256(u + 8);
    v1_8 = _mm256_cvtepi16_epi32(DOTP3216_LOAD_128(v + 8));
    
    mult1 = multiply_lo_epi32(u1_8, v1_8);
    
    // 2
    u0_8  = DOTP3216_LOAD_256(u + 16);
    v0_8 = _mm256_cvtepi16_epi32(DOTP3216_LOAD_128(v + 16));
    
    mult2 = multiply_lo_epi32(u0_8, v0_8);

    // 3
    u1_8  = DOTP3216_LOAD_256(u + 24);
    v1_8 = _mm256_cvtepi16_epi32(DOTP3216_LOAD_128(v + 24));
    
    mult3 = multiply_lo_epi32(u1_8, v1_8);

    // Sum
    accu0 = _mm256_add_epi32(accu0, mult0);
    accu1 = _mm256_add_epi32(accu1, mult1);
    accu0 = _mm256_add_epi32(accu0, mult2);
    accu1 = _mm256_add_epi32(accu1, mult3);

    // Next
    u += 32;
    v += 32;
  }
  
// #if DOTP3216_SIZE_MULTIPLE < 32  // Not sure why but slower when true
  // Unroll remaining x2
  if (n & 16)
  {
    __m256i u_8, v_8;
    __m256i mult0, mult1;

    // 0
    u_8 = DOTP3216_LOAD_256(u);
    v_8 = _mm256_cvtepi16_epi32(DOTP3216_LOAD_128(v));
    
    mult0 = multiply_lo_epi32(u_8, v_8);

    // 1
    u_8 = DOTP3216_LOAD_256(u + 8);
    v_8 = _mm256_cvtepi16_epi32(DOTP3216_LOAD_128(v + 8));
    
    mult1 = multiply_lo_epi32(u_8, v_8);

    // Sum
    accu0 = _mm256_add_epi32(accu0, mult0);
    accu1 = _mm256_add_epi32(accu1, mult1);

    // Next
    u += 16;
    v += 16;
  }

#if DOTP3216_SIZE_MULTIPLE < 16
  // Remaining > 8
  if (n & 8)
  {
    n &= 7;
    __m256i u_8, v_8;
    __m256i mult;
    
    u_8 = DOTP3216_LOAD_256(u + n);
    v_8 = _mm256_cvtepi16_epi32(DOTP3216_LOAD_128(v + n));
    
    mult  = multiply_lo_epi32(u_8, v_8);
    accu0 = _mm256_add_epi32(accu0, mult);
  }
#endif // DOTP3216_SIZE_MULTIPLE < 16
// #endif // DOTP3216_SIZE_MULTIPLE < 32
  
  // Sum accumulators
  accu0 = _mm256_add_epi32(accu0, accu1);
  res = horizontal_sum_epi32(accu0);
  
#if DOTP3216_SIZE_MULTIPLE < 8
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
#endif // DOTP3216_SIZE_MULTIPLE < 8
  
  return res;
}
#endif // HAS_AVX2_


#endif // DOTP_I32I16_H
