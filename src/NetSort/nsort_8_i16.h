/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef NSORT_8_I16_H
#define NSORT_8_I16_H

#include "Utils/compiler_utils.h"
#include "Utils/simd_utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <emmintrin.h>    // SSE2
#ifdef HAS_SSSE3_
  #include <tmmintrin.h>  // SSSE3
#endif
//#define DEBUG_NS8I16
#ifdef DEBUG_NS8I16
  #include "Utils/generators.h"
#endif

// SIMD optimization options
//#define NSORT_8_I16_EARLY_EXIT  // Enable if array rarely need inter-lanes sorting (first 4 vs last 4)
                                  // Early exit gain is usually about twice the otherwise check penalty (e.g. +14% vs -7%)
#if defined NSORT_8_I16_128_ALIGNED
  #define NSORT_8_I16_LOAD_128(x) _mm_load_si128((__m128i const*)(x))
#else
  #define NSORT_8_I16_LOAD_128(x) _mm_loadu_si128((__m128i const*)(x))
#endif


//
static inline int cmpfunc_i16(const void* __restrict a, const void* __restrict b) {
  return ( *(const int16_t*)a > *(const int16_t*)b );
}

//
static inline void netsort_8_i16_qsort(int16_t* __restrict v)
{
  qsort(v, 8, sizeof(int16_t), cmpfunc_i16);
}

//
#ifdef HAS_SSSE3_
static inline __m128i netsort_8_i16_sse(const __m128i in)
{
  //////// [0,1] [2,3] [4,5] [6,7]
  // 
  const __m128i shfA = _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
  __m128i tmp0 = _mm_shuffle_epi8(in, shfA); // SSSE3

#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp0);
  print_vec(v, "tmp1");
#endif
  
  // Sort
  __m128i min = _mm_min_epi16(in, tmp0);
  __m128i max = _mm_max_epi16(in, tmp0);
  tmp0 = blend_epi16_AA(min, max);
  
#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp0);
  print_vec(v, "sort1"); std::cout << "\n";
#endif
  
  //////// [0,3] [1,2] [4,7] [5,6]
  // 
  __m128i shfl = _mm_setr_epi8(6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9);
  __m128i tmp1 = _mm_shuffle_epi8(tmp0, shfl);
  
#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp1);
  print_vec(v, "tmp2");
#endif
  
  // Sort
  min = _mm_min_epi16(tmp0, tmp1);
  max = _mm_max_epi16(tmp0, tmp1);
  tmp0 = blend_epi16_CC(min, max);
  
#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp0);
  print_vec(v, "sort2"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] [4,5] [6,7]
  // 
  tmp1 = _mm_shuffle_epi8(tmp0, shfA);
  
#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp1);
  print_vec(v, "tmp3");
#endif
  
  // Sort
  min = _mm_min_epi16(tmp0, tmp1);
  max = _mm_max_epi16(tmp0, tmp1);
  tmp0 = blend_epi16_AA(min, max);
  
#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp0);
  print_vec(v, "sort3"); std::cout << "\n";
#endif
  
  //////// [0,7] [1,6] [2,5] [3,4]
  //
  shfl = _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
  tmp1 = _mm_shuffle_epi8(tmp0, shfl);
  
#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp1);
  print_vec(v, "tmp4");
#endif
  
  // Sort
  min = _mm_min_epi16(tmp0, tmp1);
  max = _mm_max_epi16(tmp0, tmp1);
  tmp1 = blend_epi16_F0(min, max);
  
#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp1);
  print_vec(v, "sort4"); std::cout << "\n";
#endif
  
#ifdef NSORT_8_I16_EARLY_EXIT
  //////// No inter-lanes sorting needed (early exit)
  tmp0 = _mm_cmpeq_epi16(tmp0, tmp1);
  if (_mm_movemask_epi8(tmp0) == 0xFFFF)
  {
#ifdef DEBUG_NS8I16
  static bool done = false;
  if (!done)
    std::cout << "\nEarly exit 8xI16 SSE\n";
  done = true;
#endif
    _mm_storeu_si128((__m128i*)(v), tmp1);
    return;
  }
#endif
  
  //////// [0,2] [1,3] [4,6] [5,7]
  // 
  shfl = _mm_setr_epi8(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
  tmp0 = _mm_shuffle_epi8(tmp1, shfl);
  
#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp0);
  print_vec(v, "tmp5");
#endif
  
  // Sort
  min = _mm_min_epi16(tmp0, tmp1);
  max = _mm_max_epi16(tmp0, tmp1);
  tmp0 = blend_epi16_CC(min, max);
  
#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp0);
  print_vec(v, "sort5"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] [4,5] [6,7]
  // 
  tmp1 = _mm_shuffle_epi8(tmp0, shfA);
  
#ifdef DEBUG_NS8I16
  _mm_storeu_si128((__m128i*)(v), tmp1);
  print_vec(v, "tmp6");
#endif
  
  // Sort
  min = _mm_min_epi16(tmp0, tmp1);
  max = _mm_max_epi16(tmp0, tmp1);
  tmp0 = blend_epi16_AA(min, max);
  
  return tmp0;
}

//
static inline void netsort_8_i16_sse(int16_t* __restrict v)
{
  // Load -> [ 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 ]
  __m128i in = NSORT_8_I16_LOAD_128(v);
  
#ifdef DEBUG_NS8I16
  print_vec(v, "\nin SSE"); std::cout << "\n";
#endif
  
  in = netsort_8_i16_sse(in);
  
  //
  // Store
  _mm_storeu_si128((__m128i*)(v), in);
  
#ifdef DEBUG_NS8I16
  print_vec(v, "sort6"); std::cout << "\n";
#endif
}
#endif // HAS_SSSE3_


#endif // NSORT_8_I16_H
