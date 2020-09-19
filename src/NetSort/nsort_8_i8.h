/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef NSORT_8_I8_H
#define NSORT_8_I8_H

#include "Utils/compiler_utils.h"
#include "Utils/simd_utils.h"
#include "nsort_8_i16.h"

#include <stdint.h>
#include <stdlib.h>
#include <emmintrin.h>    // SSE2
#ifdef HAS_SSSE3_
  #include <tmmintrin.h>  // SSSE3
#endif
#ifdef HAS_SSE4_1_
  #include <smmintrin.h>  // SSE4.1
#endif
//#define DEBUG_NS8I8
#ifdef DEBUG_NS8I8
  #include "Utils/generators.h"
#endif

// SIMD optimization options
//#define NSORT_8_I8_EARLY_EXIT // Enable if array rarely need inter-lanes sorting (first 4 vs last 4)
                                // Early exit gain is usually about twice the otherwise check penalty (e.g. +14% vs -7%)


//
static inline int cmpfunc_i8(const void* __restrict a, const void* __restrict b) {
  return ( *(const int8_t*)a > *(const int8_t*)b );
}

//
static inline void netsort_8_i8_qsort(int8_t* __restrict v)
{
  qsort(v, 8, sizeof(int8_t), cmpfunc_i8);
}

//
#ifdef HAS_SSSE3_
static inline void netsort_8_i8_sse(int8_t* __restrict v)
{
  // Load -> [ 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 ]
  __m128i in = _mm_loadl_epi64((__m128i const*)(v));
  
#ifndef HAS_SSE4_1_
  __m128i tmp0 = extend_lo_epi8(in);
  tmp0 = netsort_8_i16_sse(tmp0);
  tmp0 = _mm_packs_epi16(tmp0, tmp0);
#else
  
#ifdef DEBUG_NS8I8
  print_vec(v, "\nin SSE"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] [4,5] [6,7]
  // 
  const __m128i shfA = _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0);
  __m128i tmp0 = _mm_shuffle_epi8(in, shfA); // SSSE3

#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp0);
  print_vec(v, "tmp1");
#endif
  
  // Sort
  __m128i min = _mm_min_epi8(in, tmp0); // SSE4.1
  __m128i max = _mm_max_epi8(in, tmp0);
  tmp0 = blend_epi8_AA(min, max);
  
#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp0);
  print_vec(v, "sort1"); std::cout << "\n";
#endif
  
  //////// [0,3] [1,2] [4,7] [5,6]
  // 
  __m128i shfl = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0);
  __m128i tmp1 = _mm_shuffle_epi8(tmp0, shfl);
  
#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp1);
  print_vec(v, "tmp2");
#endif
  
  // Sort
  min = _mm_min_epi8(tmp0, tmp1);
  max = _mm_max_epi8(tmp0, tmp1);
  tmp0 = blend_epi8_CC(min, max);
  
#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp0);
  print_vec(v, "sort2"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] [4,5] [6,7]
  // 
  tmp1 = _mm_shuffle_epi8(tmp0, shfA);
  
#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp1);
  print_vec(v, "tmp3");
#endif
  
  // Sort
  min = _mm_min_epi8(tmp0, tmp1);
  max = _mm_max_epi8(tmp0, tmp1);
  tmp0 = blend_epi8_AA(min, max);
  
#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp0);
  print_vec(v, "sort3"); std::cout << "\n";
#endif
  
  //////// [0,7] [1,6] [2,5] [3,4]
  //
  shfl = _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  tmp1 = _mm_shuffle_epi8(tmp0, shfl);
  
#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp1);
  print_vec(v, "tmp4");
#endif
  
  // Sort
  min = _mm_min_epi8(tmp0, tmp1);
  max = _mm_max_epi8(tmp0, tmp1);
  tmp1 = blend_epi8_F0(min, max);
  
#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp1);
  print_vec(v, "sort4"); std::cout << "\n";
#endif
  
#ifdef NSORT_8_I8_EARLY_EXIT
  //////// No inter-lanes sorting needed (early exit)
  tmp0 = _mm_cmpeq_epi16(tmp0, tmp1);
  if (_mm_movemask_epi8(tmp0) == 0xFFFF)
  {
#ifdef DEBUG_NS8I8
  static bool done = false;
  if (!done)
    std::cout << "\nEarly exit 8xI8 SSE\n";
  done = true;
#endif
    _mm_storel_epi64((__m128i*)(v), tmp1);
    return;
  }
#endif
  
  //////// [0,2] [1,3] [4,6] [5,7]
  // 
  shfl = _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0);
  tmp0 = _mm_shuffle_epi8(tmp1, shfl);
  
#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp0);
  print_vec(v, "tmp5");
#endif
  
  // Sort
  min = _mm_min_epi8(tmp0, tmp1);
  max = _mm_max_epi8(tmp0, tmp1);
  tmp0 = blend_epi8_CC(min, max);
  
#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp0);
  print_vec(v, "sort5"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] [4,5] [6,7]
  // 
  tmp1 = _mm_shuffle_epi8(tmp0, shfA);
  
#ifdef DEBUG_NS8I8
  _mm_storel_epi64((__m128i*)(v), tmp1);
  print_vec(v, "tmp6");
#endif
  
  // Sort
  min = _mm_min_epi8(tmp0, tmp1);
  max = _mm_max_epi8(tmp0, tmp1);
  tmp0 = blend_epi8_AA(min, max);  
#endif // HAS_SSE4_1_
  
  //
  // Store
  _mm_storel_epi64((__m128i*)(v), tmp0);
  
#ifdef DEBUG_NS8I8
  print_vec(v, "sort6"); std::cout << "\n";
#endif
}
#endif // HAS_SSSE3_


#endif // NSORT_8_I8_H
