/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef NSORT_8_FLT_H
#define NSORT_8_FLT_H

#include "Utils/compiler_utils.h"
#include "Utils/simd_utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <emmintrin.h>    // SSE2
#ifdef HAS_SSE4_1_
  #include <smmintrin.h>  // SSE4.1
#endif
#ifdef HAS_AVX_
  #include <immintrin.h>  // AVX
#endif
//#define DEBUG_NS8FLT
#ifdef DEBUG_NS8FLT
  #include "Utils/generators.h"
#endif

// SIMD optimization options
//#define NSORT_8_FLT_EARLY_EXIT  // Enable if array rarely need inter-lanes sorting (first 4 vs last 4)
                                  // Early exit gain is usually about twice the otherwise check penalty (e.g. +15% vs -7.5%)
#if defined NSORT_8_FLT_256_ALIGNED
  #define NSORT_8_FLT_LOAD_128(x) _mm_load_ps(x)
  #ifdef HAS_AVX_
    #define NSORT_8_FLT_LOAD_256(x) _mm256_load_ps(x)
  #endif
#elif defined NSORT_8_FLT_128_ALIGNED
  #define NSORT_8_FLT_LOAD_128(x) _mm_load_ps(x)
  #ifdef HAS_AVX_
    #define NSORT_8_FLT_LOAD_256(x) _mm256_loadu_ps(x)
  #endif
#else
  #define NSORT_8_FLT_LOAD_128(x) _mm_loadu_ps(x)
  #ifdef HAS_AVX_
    #define NSORT_8_FLT_LOAD_256(x) _mm256_loadu_ps(x)
  #endif
#endif


//
static inline int cmpfunc_flt(const void* __restrict a, const void* __restrict b) {
  return ( *(const float*)a > *(const float*)b );
}

//
static inline void netsort_8_flt_qsort(float* __restrict v)
{
  qsort(v, 8, sizeof(float), cmpfunc_flt);
}

//
static inline void netsort_8_flt_sse(float* __restrict v)
{
  // Load -> [ 7 | 6 | 5 | 4 ] | [ 3 | 2 | 1 | 0 ]
  __m128 in0 = NSORT_8_FLT_LOAD_128(v);
  __m128 in1 = NSORT_8_FLT_LOAD_128(v+4);
  
#ifdef DEBUG_NS8FLT
  print_vec(v, "\nin SSE"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  __m128 tmp0 = _mm_shuffle_ps(in0, in0, _MM_SHUFFLE(2,3,0,1));
  __m128 tmp1 = _mm_shuffle_ps(in1, in1, _MM_SHUFFLE(2,3,0,1));

#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp0);
  _mm_storeu_ps(v+4, tmp1);
  print_vec(v, "tmp1");
#endif
  
  // Sort
  __m128 min0 = _mm_min_ps(in0, tmp0);
  __m128 max0 = _mm_max_ps(in0, tmp0);
  tmp0 = blend_ps_0A(min0, max0);
  
  __m128 min1 = _mm_min_ps(in1, tmp1);
  __m128 max1 = _mm_max_ps(in1, tmp1);
  tmp1 = blend_ps_0A(min1, max1);
  
#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp0);
  _mm_storeu_ps(v+4, tmp1);
  print_vec(v, "sort1"); std::cout << "\n";
#endif
  
  //////// [0,3] [1,2] | [4,7] [5,6]
  // 
  __m128 tmp2 = _mm_shuffle_ps(tmp0, tmp0, _MM_SHUFFLE(0,1,2,3));
  __m128 tmp3 = _mm_shuffle_ps(tmp1, tmp1, _MM_SHUFFLE(0,1,2,3));
  
#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp2);
  _mm_storeu_ps(v+4, tmp3);
  print_vec(v, "tmp2");
#endif
  
  // Sort
  min0 = _mm_min_ps(tmp0, tmp2);
  max0 = _mm_max_ps(tmp0, tmp2);
  tmp0 = blend_ps_0C(min0, max0);
  
  min1 = _mm_min_ps(tmp1, tmp3);
  max1 = _mm_max_ps(tmp1, tmp3);
  tmp1 = blend_ps_0C(min1, max1);
  
#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp0);
  _mm_storeu_ps(v+4, tmp1);
  print_vec(v, "sort2"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  tmp2 = _mm_shuffle_ps(tmp0, tmp0, _MM_SHUFFLE(2,3,0,1));
  tmp3 = _mm_shuffle_ps(tmp1, tmp1, _MM_SHUFFLE(2,3,0,1));
  
#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp2);
  _mm_storeu_ps(v+4, tmp3);
  print_vec(v, "tmp3");
#endif
  
  // Sort
  min0 = _mm_min_ps(tmp0, tmp2);
  max0 = _mm_max_ps(tmp0, tmp2);
  tmp0 = blend_ps_0A(min0, max0);
  
  min1 = _mm_min_ps(tmp1, tmp3);
  max1 = _mm_max_ps(tmp1, tmp3);
  tmp1 = blend_ps_0A(min1, max1);
  
#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp0);
  _mm_storeu_ps(v+4, tmp1);
  print_vec(v, "sort3"); std::cout << "\n";
#endif
  
  //////// [0,7] [1,6] | [2,5] [3,4]
  //
  tmp2 = _mm_shuffle_ps(tmp1, tmp1, _MM_SHUFFLE(0,1,2,3));
  tmp3 = _mm_shuffle_ps(tmp0, tmp0, _MM_SHUFFLE(0,1,2,3));
  
#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp2);
  _mm_storeu_ps(v+4, tmp3);
  print_vec(v, "tmp4");
#endif
  
  // Sort
  min0 = _mm_min_ps(tmp0, tmp2);
  max0 = _mm_max_ps(tmp0, tmp2);
  tmp2 = blend_ps_00(min0, max0);
  
  min1 = _mm_min_ps(tmp1, tmp3);
  max1 = _mm_max_ps(tmp1, tmp3);
  tmp3 = blend_ps_0F(min1, max1);
  
#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp2);
  _mm_storeu_ps(v+4, tmp3);
  print_vec(v, "sort4"); std::cout << "\n";
#endif
  
#ifdef NSORT_8_FLT_EARLY_EXIT
  //////// No inter-lanes sorting needed (early exit)
  tmp0 = _mm_cmpneq_ps(tmp0, tmp2);
  tmp1 = _mm_cmpneq_ps(tmp1, tmp3);
  if (_mm_movemask_ps(tmp0) == 0 && _mm_movemask_ps(tmp1) == 0)
  {
#ifdef DEBUG_NS8FLT
  static bool done = false;
  if (!done)
    std::cout << "\nEarly exit 8xFLT SSE\n";
  done = true;
#endif
    _mm_storeu_ps(v,   tmp2);
    _mm_storeu_ps(v+4, tmp3);
    return;
  }
#endif
  
  //////// [0,2] [1,3] | [4,6] [5,7]
  // 
  tmp0 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(1,0,3,2));
  tmp1 = _mm_shuffle_ps(tmp3, tmp3, _MM_SHUFFLE(1,0,3,2));
  
#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp0);
  _mm_storeu_ps(v+4, tmp1);
  print_vec(v, "tmp5");
#endif
  
  // Sort
  min0 = _mm_min_ps(tmp0, tmp2);
  max0 = _mm_max_ps(tmp0, tmp2);
  tmp0 = blend_ps_0C(min0, max0);
  
  min1 = _mm_min_ps(tmp1, tmp3);
  max1 = _mm_max_ps(tmp1, tmp3);
  tmp1 = blend_ps_0C(min1, max1);
  
#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp0);
  _mm_storeu_ps(v+4, tmp1);
  print_vec(v, "sort5"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  tmp2 = _mm_shuffle_ps(tmp0, tmp0, _MM_SHUFFLE(2,3,0,1));
  tmp3 = _mm_shuffle_ps(tmp1, tmp1, _MM_SHUFFLE(2,3,0,1));
  
#ifdef DEBUG_NS8FLT
  _mm_storeu_ps(v,   tmp2);
  _mm_storeu_ps(v+4, tmp3);
  print_vec(v, "tmp6");
#endif
  
  // Sort
  min0 = _mm_min_ps(tmp0, tmp2);
  max0 = _mm_max_ps(tmp0, tmp2);
  tmp0 = blend_ps_0A(min0, max0);
  
  min1 = _mm_min_ps(tmp1, tmp3);
  max1 = _mm_max_ps(tmp1, tmp3);
  tmp1 = blend_ps_0A(min1, max1);
  
  //
  // Store
  _mm_storeu_ps(v,   tmp0);
  _mm_storeu_ps(v+4, tmp1);
  
#ifdef DEBUG_NS8FLT
  print_vec(v, "sort6"); std::cout << "\n";
#endif
}

//
#ifdef HAS_AVX_
static inline void netsort_8_flt_avx(float* __restrict v)
{
  // Load -> [ 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 ]
  __m256 in = NSORT_8_FLT_LOAD_256(v);
  
#ifdef DEBUG_NS8FLT
  print_vec(v, "\nin AVX"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  __m256 tmp0 = _mm256_permute_ps(in, _MM_SHUFFLE(2,3,0,1));
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp0);
  print_vec(v, "tmp0");
#endif
  
  // Sort
  __m256 min = _mm256_min_ps(in, tmp0);
  __m256 max = _mm256_max_ps(in, tmp0);
  tmp0 = _mm256_blend_ps(min, max, 0xAA);
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp0);
  print_vec(v, "sort1"); std::cout << "\n";
#endif
  
  //////// [0,3] [1,2] | [4,7] [5,6]
  // 
  __m256 tmp1 = _mm256_permute_ps(tmp0, _MM_SHUFFLE(0,1,2,3));
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp1);
  print_vec(v, "tmp1");
#endif
  
  // Sort
  min = _mm256_min_ps(tmp0, tmp1);
  max = _mm256_max_ps(tmp0, tmp1);
  tmp0 = _mm256_blend_ps(min, max, 0xCC);
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp0);
  print_vec(v, "sort2"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  tmp1 = _mm256_permute_ps(tmp0, _MM_SHUFFLE(2,3,0,1));
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp1);
  print_vec(v, "tmp3");
#endif
  
  // Sort
  min = _mm256_min_ps(tmp0, tmp1);
  max = _mm256_max_ps(tmp0, tmp1);
  tmp0 = _mm256_blend_ps(min, max, 0xAA);
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp0);
  print_vec(v, "sort3"); std::cout << "\n";
#endif
  
  //////// [0,7] [1,6] [2,5] [3,4]
  //
  tmp1 = _mm256_permute2f128_ps(tmp0, tmp0, 1); // inv hi/lo (inter-lane penalty)
  tmp1 = _mm256_permute_ps(tmp1, _MM_SHUFFLE(0,1,2,3));
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp1);
  print_vec(v, "tmp4");
#endif
  
  // Sort
  min = _mm256_min_ps(tmp0, tmp1);
  max = _mm256_max_ps(tmp0, tmp1);
  tmp1 = _mm256_blend_ps(min, max, 0xF0);
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp1);
  print_vec(v, "sort4"); std::cout << "\n";
#endif
  
#ifdef NSORT_8_FLT_EARLY_EXIT
  //////// No inter-lanes sorting needed (early exit)
  //
  tmp0 = _mm256_cmp_ps(tmp0, tmp1, _CMP_NEQ_UQ);
  if (_mm256_movemask_ps(tmp0) == 0)
  {
#ifdef DEBUG_NS8FLT
  static bool done = false;
  if (!done)
    std::cout << "\nEarly exit 8xFLT AVX\n";
  done = true;
#endif
    _mm256_storeu_ps(v, tmp1);
    return;
  }
#endif
  
  //////// [0,2] [1,3] | [4,6] [5,7]
  // 
  tmp0 = _mm256_permute_ps(tmp1, _MM_SHUFFLE(1,0,3,2));
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp0);
  print_vec(v, "tmp5");
#endif
  
  // Sort
  min = _mm256_min_ps(tmp0, tmp1);
  max = _mm256_max_ps(tmp0, tmp1);
  tmp0 = _mm256_blend_ps(min, max, 0xCC);
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp0);
  print_vec(v, "sort5"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  tmp1 = _mm256_permute_ps(tmp0, _MM_SHUFFLE(2,3,0,1));
  
#ifdef DEBUG_NS8FLT
  _mm256_storeu_ps(v, tmp1);
  print_vec(v, "tmp6");
#endif
  
  // Sort
  min = _mm256_min_ps(tmp0, tmp1);
  max = _mm256_max_ps(tmp0, tmp1);
  tmp0 = _mm256_blend_ps(min, max, 0xAA);
  
  //
  // Store
  _mm256_storeu_ps(v, tmp0);
  
#ifdef DEBUG_NS8FLT
  print_vec(v, "sort6"); std::cout << "\n";
#endif
}
#endif // HAS_AVX_


#endif // NSORT_8_FLT_H
