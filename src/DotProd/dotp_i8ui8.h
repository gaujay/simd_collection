/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef DOTP_I8UI8_H
#define DOTP_I8UI8_H

#include "Utils/compiler_utils.h"
#include "Utils/simd_utils.h"

#include <stdint.h>
#include <emmintrin.h>    // SSE2
#ifdef HAS_SSSE3_
  #include <tmmintrin.h>  // SSSE3
#endif
#ifdef HAS_AVX2_
  #include <immintrin.h>  // AVX2
#endif

// SIMD optimization options
#ifndef DOTP88_SIZE_MULTIPLE
  #define DOTP88_SIZE_MULTIPLE 0   // 64, 32, 16, 8 (0: no optim)
#endif
//#define DOTP88_ACCU_3   // Use 3/4 accumulators (depend on HW/vectors size)
//#define DOTP88_ACCU_4
#if defined(DOTP88_ACCU_4) && !defined(DOTP88_ACCU_3)
  #define DOTP88_ACCU_3
#endif
#if defined DOTP88_256_ALIGNED
  #define DOTP88_LOAD_64(x)  _mm_loadl_epi64((__m128i const*)(x))
  #define DOTP88_LOAD_128(x) _mm_load_si128((__m128i const*)(x))
  #ifdef HAS_AVX2_
    #define DOTP88_LOAD_256(x) _mm256_load_si256((__m256i const*)(x))
  #endif
#elif defined(DOTP88_128_ALIGNED)
  #define DOTP88_LOAD_64(x)  _mm_loadl_epi64((__m128i const*)(x))
  #define DOTP88_LOAD_128(x) _mm_load_si128((__m128i const*)(x))
  #ifdef HAS_AVX2_
    #define DOTP88_LOAD_256(x) _mm256_loadu_si256((__m256i const*)(x))
  #endif
#else
  #define DOTP88_LOAD_64(x)  _mm_loadl_epi64((__m128i const*)(x))
  #define DOTP88_LOAD_128(x) _mm_loadu_si128((__m128i const*)(x))
  #ifdef HAS_AVX2_
    #define DOTP88_LOAD_256(x) _mm256_loadu_si256((__m256i const*)(x))
  #endif
#endif


//
DISABLE_FUNC_VECTORIZATION_
static inline int32_t dotProduct_i8ui8_scalarforced(int8_t const* __restrict u, uint8_t const* __restrict v, size_t n)
{
  int32_t res = 0;
  DISABLE_LOOP_VECTORIZATION_
  for (size_t i=0; i<n; ++i)
    res += u[i] * v[i];
    
  return res;
}

//
static inline int32_t dotProduct_i8ui8_scalar(int8_t const* __restrict u, uint8_t const* __restrict v, size_t n)
{
  int32_t res = 0;
  for (size_t i=0; i<n; ++i)
    res += u[i] * v[i];
    
  return res;
}

//
#ifdef HAS_SSSE3_
static inline int32_t dotProduct_i8ui8_sse_naive(int8_t const* __restrict u, uint8_t const* __restrict v, size_t n)
{
  int32_t res;
  size_t count = n >> 4;
  
  // Accumulator
  __m128i accu = _mm_setzero_si128();

  // Loop
  while (count--)
  {
    __m128i u_16, v_16;
    __m128i madd;

    // 0
    u_16 = DOTP88_LOAD_128(u);
    v_16 = DOTP88_LOAD_128(v);
    
    madd = _mm_maddubs_epi16(v_16, u_16); // SSSE3
    accu = _mm_add_epi32(accu, extend_lo_epi16(madd));
    accu = _mm_add_epi32(accu, extend_hi_epi16(madd));
    
    // Next
    u += 16;
    v += 16;
  }
  // Horizontal sum
  accu = _mm_hadd_epi32(accu, accu); // SSSE3
  accu = _mm_hadd_epi32(accu, accu);
  res = _mm_cvtsi128_si32(accu);

#if DOTP88_SIZE_MULTIPLE < 16
  n &= 15;
  while (n--)
    res += u[n] * v[n];
#endif

  return res;
}
#endif // HAS_SSSE3_

//
#ifdef HAS_SSSE3_
static inline int32_t dotProduct_i8ui8_sse(int8_t const* __restrict u, uint8_t const* __restrict v, size_t n)
{
  int32_t res;
  size_t count = n >> 5;
  
  // Accumulator
  __m128i accu0 = _mm_setzero_si128();
  __m128i accu1 = _mm_setzero_si128();

  // Unroll x2
  while (count--)
  {
    __m128i u0_16, u1_16;
    __m128i v0_16, v1_16;
    __m128i madd0, madd1;

    // 0
    u0_16 = DOTP88_LOAD_128(u);
    v0_16 = DOTP88_LOAD_128(v);
    
    madd0 = _mm_maddubs_epi16(v0_16, u0_16); // SSSE3 (saturate product pairs into int16)
    
    // 1
    u1_16 = DOTP88_LOAD_128(u + 16);
    v1_16 = DOTP88_LOAD_128(v + 16);
    
    madd1 = _mm_maddubs_epi16(v1_16, u1_16);
    
    // Sum
    accu0 = _mm_add_epi32(accu0, extend_lo_epi16(madd0));
    accu1 = _mm_add_epi32(accu1, extend_hi_epi16(madd0));
    accu0 = _mm_add_epi32(accu0, extend_lo_epi16(madd1));
    accu1 = _mm_add_epi32(accu1, extend_hi_epi16(madd1));
    
    // Next
    u += 32;
    v += 32;
  }
#if DOTP88_SIZE_MULTIPLE < 32
  // Remaining > 16
  if (n & 16)
  {
    __m128i u_16, v_16;
    __m128i madd;

    // 0
    u_16 = DOTP88_LOAD_128(u);
    v_16 = DOTP88_LOAD_128(v);
    
    madd  = _mm_maddubs_epi16(v_16, u_16);
    accu0 = _mm_add_epi32(accu0, extend_lo_epi16(madd));
    accu1 = _mm_add_epi32(accu1, extend_hi_epi16(madd));
    
    // Next
    u += 16;
    v += 16;
  }
#if DOTP88_SIZE_MULTIPLE < 16
  // Remaining > 8
  if (n & 8)
  {
    n &= 7;
    __m128i u_8, v_8;
    __m128i madd;
  
    u_8 = DOTP88_LOAD_64(u + n);
    v_8 = DOTP88_LOAD_64(v + n);

    madd  = _mm_maddubs_epi16(v_8, u_8);
    accu0 = _mm_add_epi32(accu0, extend_lo_epi16(madd));
  }
#endif // DOTP88_SIZE_MULTIPLE < 16
#endif // DOTP88_SIZE_MULTIPLE < 32
  
  // Sum accumulators
  accu0 = _mm_add_epi32(accu0, accu1);
  res = horizontal_sum_epi32(accu0);
  
#if DOTP88_SIZE_MULTIPLE < 8
  // Remaining < 8
  switch (n & 7)
  {
    case 7: res += u[6] * v[6];
    case 6: res += u[5] * v[5];
    case 5: res += u[4] * v[4];
    case 4: res += u[3] * v[3];
    case 3: res += u[2] * v[2];
    case 2: res += u[1] * v[1];
    case 1: res += u[0] * v[0];
    default: break;
  }
#endif
  
  return res;
}
#endif // HAS_SSSE3_

//
#ifdef HAS_AVX2_
static inline int32_t dotProduct_i8ui8_avx2(int8_t const* __restrict u, uint8_t const* __restrict v, size_t n)
{
  int32_t res;
  size_t count = n >> 7;
  
  // Accumulators
  __m256i accu0 = _mm256_setzero_si256();
  __m256i accu1 = _mm256_setzero_si256();
#ifdef DOTP88_ACCU_3
  __m256i accu2 = _mm256_setzero_si256();
  #ifdef DOTP88_ACCU_4
    __m256i accu3 = _mm256_setzero_si256();
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
    __m256i u0_32, u1_32, u2_32, u3_32;
    __m256i v0_32, v1_32, v2_32, v3_32;
    __m256i madd0, madd1, madd2, madd3;

    // 0
    u0_32 = DOTP88_LOAD_256(u);
    v0_32 = DOTP88_LOAD_256(v);

    madd0 = _mm256_maddubs_epi16(v0_32, u0_32);

    // 1
    u1_32 = DOTP88_LOAD_256(u + 32);
    v1_32 = DOTP88_LOAD_256(v + 32);

    madd1 = _mm256_maddubs_epi16(v1_32, u1_32);
    
    // 2
    u2_32 = DOTP88_LOAD_256(u + 64);
    v2_32 = DOTP88_LOAD_256(v + 64);

    madd2 = _mm256_maddubs_epi16(v2_32, u2_32);

    // 3
    u3_32 = DOTP88_LOAD_256(u + 96);
    v3_32 = DOTP88_LOAD_256(v + 96);

    madd3 = _mm256_maddubs_epi16(v3_32, u3_32);
    
    // Sum
    accu0 = _mm256_add_epi32(accu0, extend_lo_epi16(madd0));
    accu1 = _mm256_add_epi32(accu1, extend_hi_epi16(madd0));
    accu2 = _mm256_add_epi32(accu2, extend_lo_epi16(madd1));
    accu3 = _mm256_add_epi32(accu3, extend_hi_epi16(madd1));
    accu0 = _mm256_add_epi32(accu0, extend_lo_epi16(madd2));
    accu1 = _mm256_add_epi32(accu1, extend_hi_epi16(madd2));
    accu2 = _mm256_add_epi32(accu2, extend_lo_epi16(madd3));
    accu3 = _mm256_add_epi32(accu3, extend_hi_epi16(madd3));
    
    // Next
    u += 128;
    v += 128;
  }
#ifdef DOTP88_ACCU_4
  // Sum accumulators
  accu2 = _mm256_add_epi32(accu2, accu3);
#else
  #ifdef DOTP88_ACCU_3
  accu1 = _mm256_add_epi32(accu1, accu2);
  #endif
#endif
  
#if DOTP88_SIZE_MULTIPLE < 128
  // Unroll remaining x2
  if (n & 64)
  {
    __m256i u0_32, u1_32;
    __m256i v0_32, v1_32;
    __m256i madd0, madd1;

    // 0
    u0_32 = DOTP88_LOAD_256(u);
    v0_32 = DOTP88_LOAD_256(v);

    madd0 = _mm256_maddubs_epi16(v0_32, u0_32);

    // 1
    u1_32 = DOTP88_LOAD_256(u + 32);
    v1_32 = DOTP88_LOAD_256(v + 32);

    madd1 = _mm256_maddubs_epi16(v1_32, u1_32);

    // Sum
    accu0 = _mm256_add_epi32(accu0, extend_lo_epi16(madd0));
    accu1 = _mm256_add_epi32(accu1, extend_hi_epi16(madd0));
    accu2 = _mm256_add_epi32(accu2, extend_lo_epi16(madd1));
    accu3 = _mm256_add_epi32(accu3, extend_hi_epi16(madd1));
    
    // Next
    u += 64;
    v += 64;
  }
#endif // DOTP88_SIZE_MULTIPLE < 128
#ifdef DOTP88_ACCU_4
  // Sum accumulators
  accu1 = _mm256_add_epi32(accu1, accu2);
#endif

#if DOTP88_SIZE_MULTIPLE < 64
  // Remaining > 32
  if (n & 32)
  {
    __m256i u_32,  v_32;
    __m256i madd;
  
    u_32 = DOTP88_LOAD_256(u);
    v_32 = DOTP88_LOAD_256(v);

    madd = _mm256_maddubs_epi16(v_32, u_32);
    
    // Sum
    accu0 = _mm256_add_epi32(accu0, extend_lo_epi16(madd));
    accu1 = _mm256_add_epi32(accu1, extend_hi_epi16(madd));
    
    // Next
    u += 32;
    v += 32;
  }
#if DOTP88_SIZE_MULTIPLE < 32
  // Remaining > 16
  if (n & 16)
  {
    n &= 15;
    __m128i u_16, v_16;
    __m256i madd;

    // 0
    u_16 = DOTP88_LOAD_128(u + n);
    v_16 = DOTP88_LOAD_128(v + n);
    
    madd = _mm256_castsi128_si256(_mm_maddubs_epi16(v_16, u_16));
    
    // Sum
    accu0 = _mm256_add_epi32(accu0, extend_lo_epi16(madd));
  }
#endif // DOTP88_SIZE_MULTIPLE < 32
#endif // DOTP88_SIZE_MULTIPLE < 64
  
  // Sum accumulators
  accu0 = _mm256_add_epi32(accu0, accu1);
  res = horizontal_sum_epi32(accu0);
  
#if DOTP88_SIZE_MULTIPLE < 16
  // Remaining < 16
  switch (n & 15)
  {
    case 15: res += u[14] * v[14];
    case 14: res += u[13] * v[13];
    case 13: res += u[12] * v[12];
    case 12: res += u[11] * v[11];
    case 11: res += u[10] * v[10];
    case 10: res += u[ 9] * v[ 9];
    case  9: res += u[ 8] * v[ 8];
    case  8: res += u[ 7] * v[ 7];
    case  7: res += u[ 6] * v[ 6];
    case  6: res += u[ 5] * v[ 5];
    case  5: res += u[ 4] * v[ 4];
    case  4: res += u[ 3] * v[ 3];
    case  3: res += u[ 2] * v[ 2];
    case  2: res += u[ 1] * v[ 1];
    case  1: res += u[ 0] * v[ 0];
    default: break;
  }
#endif // DOTP88_SIZE_MULTIPLE < 16
  
  return res;
}
#endif // HAS_AVX2_

#ifdef accu2
  #undef accu2
#endif
#ifdef accu3
  #undef accu3
#endif

#endif // DOTP_I8UI8_H
