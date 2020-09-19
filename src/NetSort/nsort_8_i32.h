/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef NSORT_8_I32_H
#define NSORT_8_I32_H

#include "Utils/compiler_utils.h"
#include "Utils/simd_utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <emmintrin.h>    // SSE2
#ifdef HAS_SSE4_1_
  #include <smmintrin.h>  // SSE4.1
#endif
#ifdef HAS_AVX2_
  #include <immintrin.h>  // AVX2
#endif
//#define DEBUG_NS8I32
#ifdef DEBUG_NS8I32
  #include "Utils/generators.h"
#endif

// SIMD optimization options
//#define NSORT_8_I32_EARLY_EXIT  // Enable if array rarely need inter-lanes sorting (first 4 vs last 4)
                                  // Early exit gain is usually about twice the otherwise check penalty (e.g. +14% vs -7%)
#if defined NSORT_8_I32_256_ALIGNED
  #define NSORT_8_I32_LOAD_128(x) _mm_load_si128((__m128i const*)(x))
  #ifdef HAS_AVX2_
    #define NSORT_8_I32_LOAD_256(x) _mm256_load_si256((__m256i const*)(x))
  #endif
#elif defined NSORT_8_I32_128_ALIGNED
  #define NSORT_8_I32_LOAD_128(x) _mm_load_si128((__m128i const*)(x))
  #ifdef HAS_AVX2_
    #define NSORT_8_I32_LOAD_256(x) _mm256_loadu_si256((__m256i const*)(x))
  #endif
#else
  #define NSORT_8_I32_LOAD_128(x) _mm_loadu_si128((__m128i const*)(x))
  #ifdef HAS_AVX2_
    #define NSORT_8_I32_LOAD_256(x) _mm256_loadu_si256((__m256i const*)(x))
  #endif
#endif


//
static inline int cmpfunc_i32(const void* __restrict a, const void* __restrict b) {
  return ( *(const int32_t*)a > *(const int32_t*)b );
}

//
static inline void netsort_8_i32_qsort(int32_t* __restrict v)
{
  qsort(v, 8, sizeof(int32_t), cmpfunc_i32);
}

//
#ifdef HAS_SSE4_1_
static inline void netsort_8_i32_sse(int32_t* __restrict v)
{
  // Load -> [ 7 | 6 | 5 | 4 ] | [ 3 | 2 | 1 | 0 ]
  __m128i in0 = NSORT_8_I32_LOAD_128(v);
  __m128i in1 = NSORT_8_I32_LOAD_128(v+4);
  
#ifdef DEBUG_NS8I32
  print_vec(v, "\nin SSE"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  __m128i tmp0 = _mm_shuffle_epi32(in0, _MM_SHUFFLE(2,3,0,1));
  __m128i tmp1 = _mm_shuffle_epi32(in1, _MM_SHUFFLE(2,3,0,1));

#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp0);
  _mm_storeu_si128((__m128i*)(v+4), tmp1);
  print_vec(v, "tmp1");
#endif
  
  // Sort
  __m128i min0 = _mm_min_epi32(in0, tmp0); //SSE4.1 -> TODO: SSE2?
  __m128i max0 = _mm_max_epi32(in0, tmp0);
  tmp0 = blend_epi32_0A(min0, max0);

#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   min0);
  _mm_storeu_si128((__m128i*)(v+4), max0);
  print_vec(v, "mm11");
#endif
  
  __m128i min1 = _mm_min_epi32(in1, tmp1);
  __m128i max1 = _mm_max_epi32(in1, tmp1);
  tmp1 = blend_epi32_0A(min1, max1);
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   min1);
  _mm_storeu_si128((__m128i*)(v+4), max1);
  print_vec(v, "mm12");
#endif
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp0);
  _mm_storeu_si128((__m128i*)(v+4), tmp1);
  print_vec(v, "sort1"); std::cout << "\n";
#endif
  
  //////// [0,3] [1,2] | [4,7] [5,6]
  // 
  __m128i tmp2 = _mm_shuffle_epi32(tmp0, _MM_SHUFFLE(0,1,2,3));
  __m128i tmp3 = _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,1,2,3));
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp2);
  _mm_storeu_si128((__m128i*)(v+4), tmp3);
  print_vec(v, "tmp2");
#endif
  
  // Sort
  min0 = _mm_min_epi32(tmp0, tmp2);
  max0 = _mm_max_epi32(tmp0, tmp2);
  tmp0 = blend_epi32_0C(min0, max0);
  
  min1 = _mm_min_epi32(tmp1, tmp3);
  max1 = _mm_max_epi32(tmp1, tmp3);
  tmp1 = blend_epi32_0C(min1, max1);
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp0);
  _mm_storeu_si128((__m128i*)(v+4), tmp1);
  print_vec(v, "sort2"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  tmp2 = _mm_shuffle_epi32(tmp0, _MM_SHUFFLE(2,3,0,1));
  tmp3 = _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(2,3,0,1));
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp2);
  _mm_storeu_si128((__m128i*)(v+4), tmp3);
  print_vec(v, "tmp3");
#endif
  
  // Sort
  min0 = _mm_min_epi32(tmp0, tmp2);
  max0 = _mm_max_epi32(tmp0, tmp2);
  tmp0 = blend_epi32_0A(min0, max0);
  
  min1 = _mm_min_epi32(tmp1, tmp3);
  max1 = _mm_max_epi32(tmp1, tmp3);
  tmp1 = blend_epi32_0A(min1, max1);
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp0);
  _mm_storeu_si128((__m128i*)(v+4), tmp1);
  print_vec(v, "sort3"); std::cout << "\n";
#endif
  
  //////// [0,7] [1,6] | [2,5] [3,4]
  //
  tmp2 = _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0,1,2,3));
  tmp3 = _mm_shuffle_epi32(tmp0, _MM_SHUFFLE(0,1,2,3));
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp2);
  _mm_storeu_si128((__m128i*)(v+4), tmp3);
  print_vec(v, "tmp4");
#endif
  
  // Sort
  min0 = _mm_min_epi32(tmp0, tmp2);
  max0 = _mm_max_epi32(tmp0, tmp2);
  tmp2 = blend_epi32_00(min0, max0);
  
  min1 = _mm_min_epi32(tmp1, tmp3);
  max1 = _mm_max_epi32(tmp1, tmp3);
  tmp3 = blend_epi32_0F(min1, max1);
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp2);
  _mm_storeu_si128((__m128i*)(v+4), tmp3);
  print_vec(v, "sort4"); std::cout << "\n";
#endif
  
#ifdef NSORT_8_I32_EARLY_EXIT
  //////// No inter-lanes sorting needed (early exit)
  tmp0 = _mm_cmpeq_epi32(tmp0, tmp2);
  tmp1 = _mm_cmpeq_epi32(tmp1, tmp3);
  if (_mm_movemask_epi8(tmp0) == 0xFFFF && _mm_movemask_epi8(tmp1) == 0xFFFF)
  {
#ifdef DEBUG_NS8I32
  static bool done = false;
  if (!done)
    std::cout << "\nEarly exit 8xI32 SSE\n";
  done = true;
#endif
    _mm_storeu_si128((__m128i*)(v),   tmp2);
    _mm_storeu_si128((__m128i*)(v+4), tmp3);
    return;
  }
#endif
  
  //////// [0,2] [1,3] | [4,6] [5,7]
  // 
  tmp0 = _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(1,0,3,2));
  tmp1 = _mm_shuffle_epi32(tmp3, _MM_SHUFFLE(1,0,3,2));
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp0);
  _mm_storeu_si128((__m128i*)(v+4), tmp1);
  print_vec(v, "tmp5");
#endif
  
  // Sort
  min0 = _mm_min_epi32(tmp0, tmp2);
  max0 = _mm_max_epi32(tmp0, tmp2);
  tmp0 = blend_epi32_0C(min0, max0);
  
  min1 = _mm_min_epi32(tmp1, tmp3);
  max1 = _mm_max_epi32(tmp1, tmp3);
  tmp1 = blend_epi32_0C(min1, max1);
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp0);
  _mm_storeu_si128((__m128i*)(v+4), tmp1);
  print_vec(v, "sort5"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  tmp2 = _mm_shuffle_epi32(tmp0, _MM_SHUFFLE(2,3,0,1));
  tmp3 = _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(2,3,0,1));
  
#ifdef DEBUG_NS8I32
  _mm_storeu_si128((__m128i*)(v),   tmp2);
  _mm_storeu_si128((__m128i*)(v+4), tmp3);
  print_vec(v, "tmp6");
#endif
  
  // Sort
  min0 = _mm_min_epi32(tmp0, tmp2);
  max0 = _mm_max_epi32(tmp0, tmp2);
  tmp0 = blend_epi32_0A(min0, max0);
  
  min1 = _mm_min_epi32(tmp1, tmp3);
  max1 = _mm_max_epi32(tmp1, tmp3);
  tmp1 = blend_epi32_0A(min1, max1);
  
  //
  // Store
  _mm_storeu_si128((__m128i*)(v),   tmp0);
  _mm_storeu_si128((__m128i*)(v+4), tmp1);
  
#ifdef DEBUG_NS8I32
  print_vec(v, "sort6"); std::cout << "\n";
#endif
}
#endif // HAS_SSE4_1_

// TODO: test
#ifdef HAS_AVX2_
static inline void netsort_8_i32_avx2(int32_t* __restrict v)
{
  // Load -> [ 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 ]
  __m256i in = NSORT_8_I32_LOAD_256(v);
  
#ifdef DEBUG_NS8I32
  print_vec(v, "\nin AVX2"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  __m256i tmp0 = _mm256_shuffle_epi32(in, _MM_SHUFFLE(2,3,0,1));
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp0);
  print_vec(v, "tmp0");
#endif
  
  // Sort
  __m256i min = _mm256_min_epi32(in, tmp0);
  __m256i max = _mm256_max_epi32(in, tmp0);
  tmp0 = _mm256_blend_epi32(min, max, 0xAA);
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp0);
  print_vec(v, "sort1"); std::cout << "\n";
#endif
  
  //////// [0,3] [1,2] | [4,7] [5,6]
  // 
  __m256i tmp1 = _mm256_shuffle_epi32(tmp0, _MM_SHUFFLE(0,1,2,3));
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp1);
  print_vec(v, "tmp1");
#endif
  
  // Sort
  min = _mm256_min_epi32(tmp0, tmp1);
  max = _mm256_max_epi32(tmp0, tmp1);
  tmp0 = _mm256_blend_epi32(min, max, 0xCC);
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp0);
  print_vec(v, "sort2"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  tmp1 = _mm256_shuffle_epi32(tmp0, _MM_SHUFFLE(2,3,0,1));
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp1);
  print_vec(v, "tmp3");
#endif
  
  // Sort
  min = _mm256_min_epi32(tmp0, tmp1);
  max = _mm256_max_epi32(tmp0, tmp1);
  tmp0 = _mm256_blend_epi32(min, max, 0xAA);
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp0);
  print_vec(v, "sort3"); std::cout << "\n";
#endif
  
  //////// [0,7] [1,6] [2,5] [3,4]
  //
  tmp1 = _mm256_permute2f128_si256(tmp0, tmp0, 1); // inv hi/lo (inter-lane penalty)
  tmp1 = _mm256_shuffle_epi32(tmp1, _MM_SHUFFLE(0,1,2,3));
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp1);
  print_vec(v, "tmp4");
#endif
  
  // Sort
  min = _mm256_min_epi32(tmp0, tmp1);
  max = _mm256_max_epi32(tmp0, tmp1);
  tmp1 = _mm256_blend_epi32(min, max, 0xF0);
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp1);
  print_vec(v, "sort4"); std::cout << "\n";
#endif
  
#ifdef NSORT_8_I32_EARLY_EXIT
  //////// No inter-lanes sorting needed (early exit)
  // TODO: test
  tmp0 = _mm256_cmpeq_epi32(tmp0, tmp1);
  if (_mm256_movemask_epi8(tmp0) == 0xFFFFFFFF)
  {
#ifdef DEBUG_NS8I32
  static bool done = false;
  if (!done)
    std::cout << "\nEarly exit 8xI32 AVX2\n";
  done = true;
#endif
    _mm256_storeu_si256((__m256i*)v, tmp1);
    return;
  }
#endif
  
  //////// [0,2] [1,3] | [4,6] [5,7]
  // 
  tmp0 = _mm256_shuffle_epi32(tmp1, _MM_SHUFFLE(1,0,3,2));
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp0);
  print_vec(v, "tmp5");
#endif
  
  // Sort
  min = _mm256_min_epi32(tmp0, tmp1);
  max = _mm256_max_epi32(tmp0, tmp1);
  tmp0 = _mm256_blend_epi32(min, max, 0xCC);
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp0);
  print_vec(v, "sort5"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  tmp1 = _mm256_shuffle_epi32(tmp0, _MM_SHUFFLE(2,3,0,1));
  
#ifdef DEBUG_NS8I32
  _mm256_storeu_si256((__m256i*)v, tmp1);
  print_vec(v, "tmp6");
#endif
  
  // Sort
  min = _mm256_min_epi32(tmp0, tmp1);
  max = _mm256_max_epi32(tmp0, tmp1);
  tmp0 = _mm256_blend_epi32(min, max, 0xAA);
  
  //
  // Store
  _mm256_storeu_si256((__m256i*)v, tmp0);
  
#ifdef DEBUG_NS8I32
  print_vec(v, "sort6"); std::cout << "\n";
#endif
}
#endif // HAS_AVX2_


#endif // NSORT_8_I32_H
