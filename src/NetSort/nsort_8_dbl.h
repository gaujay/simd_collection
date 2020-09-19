/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef NSORT_8_DBL_H
#define NSORT_8_DBL_H

#include "Utils/compiler_utils.h"
#include "Utils/simd_utils.h"

#include <stdint.h>
#include <stdlib.h>
#ifdef HAS_AVX_
  #include <immintrin.h>  // AVX
#endif
//#define DEBUG_NS8DBL
#ifdef DEBUG_NS8DBL
  #include "Utils/generators.h"
#endif

// SIMD optimization options
//#define NSORT_8_DBL_EARLY_EXIT  // Enable if array rarely need inter-lanes sorting (first 4 vs last 4)
                                  // Early exit gain is usually about twice the otherwise check penalty (e.g. +15% vs -7.5%)
#if defined NSORT_8_DBL_256_ALIGNED
    #define NSORT_8_DBL_LOAD_256(x) _mm256_load_pd(x)
#else
    #define NSORT_8_DBL_LOAD_256(x) _mm256_loadu_pd(x)
#endif


//
static inline int cmpfunc_dbl(const void* __restrict a, const void* __restrict b) {
  return ( *(const double*)a > *(const double*)b );
}

//
static inline void netsort_8_dbl_qsort(double* __restrict v)
{
  qsort(v, 8, sizeof(double), cmpfunc_dbl);
}

//
#ifdef HAS_AVX_
static inline void netsort_8_dbl_avx(double* __restrict v)
{
  // Load -> [[ 7 | 6] | [5 | 4 ]] | [[ 3 | 2] | [1 | 0 ]]
  __m256d in0 = NSORT_8_DBL_LOAD_256(v);
  __m256d in1 = NSORT_8_DBL_LOAD_256(v+4);
  
#ifdef DEBUG_NS8DBL
  print_vec(v, "\nin AVX"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  __m256d tmp0 = _mm256_permute_pd(in0, 0x05);
  __m256d tmp1 = _mm256_permute_pd(in1, 0x05);

#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp0);
  _mm256_storeu_pd(v+4, tmp1);
  print_vec(v, "tmp1");
#endif
  
  // Sort
  __m256d min0 = _mm256_min_pd(in0, tmp0);
  __m256d max0 = _mm256_max_pd(in0, tmp0);
  tmp0 = _mm256_blend_pd(min0, max0, 0x0A);
  
  __m256d min1 = _mm256_min_pd(in1, tmp1);
  __m256d max1 = _mm256_max_pd(in1, tmp1);
  tmp1 = _mm256_blend_pd(min1, max1, 0x0A);
  
#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp0);
  _mm256_storeu_pd(v+4, tmp1);
  print_vec(v, "sort1"); std::cout << "\n";
#endif
  
  //////// [0,3] [1,2] | [4,7] [5,6]
  // 
#ifdef HAS_AVX2_ //TODO: test
  __m256d tmp2 = _mm256_permute4x64_pd(tmp0, _MM_SHUFFLE(0,1,2,3));
  __m256d tmp3 = _mm256_permute4x64_pd(tmp1, _MM_SHUFFLE(0,1,2,3));
#else
  __m256d tmp2 = _mm256_permute2f128_pd(tmp0, tmp0, 1); // inv hi/lo (inter-lane penalty)
  __m256d tmp3 = _mm256_permute2f128_pd(tmp1, tmp1, 1);
  tmp2 = _mm256_permute_pd(tmp2, 0x05);
  tmp3 = _mm256_permute_pd(tmp3, 0x05);
#endif
  
#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp2);
  _mm256_storeu_pd(v+4, tmp3);
  print_vec(v, "tmp2");
#endif
  
  // Sort
  min0 = _mm256_min_pd(tmp0, tmp2);
  max0 = _mm256_max_pd(tmp0, tmp2);
  tmp0 = _mm256_blend_pd(min0, max0, 0x0C);
  
  min1 = _mm256_min_pd(tmp1, tmp3);
  max1 = _mm256_max_pd(tmp1, tmp3);
  tmp1 = _mm256_blend_pd(min1, max1, 0x0C);
  
#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp0);
  _mm256_storeu_pd(v+4, tmp1);
  print_vec(v, "sort2"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  tmp2 = _mm256_shuffle_pd(tmp0, tmp0, 0x05);
  tmp3 = _mm256_shuffle_pd(tmp1, tmp1, 0x05);
  
#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp2);
  _mm256_storeu_pd(v+4, tmp3);
  print_vec(v, "tmp3");
#endif
  
  // Sort
  min0 = _mm256_min_pd(tmp0, tmp2);
  max0 = _mm256_max_pd(tmp0, tmp2);
  tmp0 = _mm256_blend_pd(min0, max0, 0x0A);
  
  min1 = _mm256_min_pd(tmp1, tmp3);
  max1 = _mm256_max_pd(tmp1, tmp3);
  tmp1 = _mm256_blend_pd(min1, max1, 0x0A);
  
#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp0);
  _mm256_storeu_pd(v+4, tmp1);
  print_vec(v, "sort3"); std::cout << "\n";
#endif
  
  //////// [0,7] [1,6] | [2,5] [3,4]
  //
#ifdef HAS_AVX2_ //TODO: test
  tmp2 = _mm256_permute4x64_pd(tmp1, _MM_SHUFFLE(0,1,2,3));
  tmp3 = _mm256_permute4x64_pd(tmp0, _MM_SHUFFLE(0,1,2,3));
#else
  tmp2 = _mm256_permute2f128_pd(tmp1, tmp1, 1); // inv hi/lo (inter-lane penalty)
  tmp3 = _mm256_permute2f128_pd(tmp0, tmp0, 1);
  tmp2 = _mm256_permute_pd(tmp2, 0x05);
  tmp3 = _mm256_permute_pd(tmp3, 0x05);
#endif
  
#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp2);
  _mm256_storeu_pd(v+4, tmp3);
  print_vec(v, "tmp4");
#endif
  
  // Sort
  min0 = _mm256_min_pd(tmp0, tmp2);
  max0 = _mm256_max_pd(tmp0, tmp2);
  tmp2 = _mm256_blend_pd(min0, max0, 0x00);
  
  min1 = _mm256_min_pd(tmp1, tmp3);
  max1 = _mm256_max_pd(tmp1, tmp3);
  tmp3 = _mm256_blend_pd(min1, max1, 0x0F);
  
#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp2);
  _mm256_storeu_pd(v+4, tmp3);
  print_vec(v, "sort4"); std::cout << "\n";
#endif
  
#ifdef NSORT_8_DBL_EARLY_EXIT
  //////// No inter-lanes sorting needed (early exit)
  tmp0 = _mm256_cmp_pd(tmp0, tmp2, _CMP_NEQ_UQ);
  tmp1 = _mm256_cmp_pd(tmp1, tmp3, _CMP_NEQ_UQ);
  if (_mm256_movemask_pd(tmp0) == 0 && _mm256_movemask_pd(tmp1) == 0)
  {
#ifdef DEBUG_NS8DBL
  static bool done = false;
  if (!done)
    std::cout << "\nEarly exit 8xDBL AVX\n";
  done = true;
#endif
    _mm256_storeu_pd(v,   tmp2);
    _mm256_storeu_pd(v+4, tmp3);
    return;
  }
#endif
  
  //////// [0,2] [1,3] | [4,6] [5,7]
  // 
#ifdef HAS_AVX2_ //TODO: test
  tmp0 = _mm256_permute4x64_pd(tmp2, _MM_SHUFFLE(1,0,3,2));
  tmp1 = _mm256_permute4x64_pd(tmp3, _MM_SHUFFLE(1,0,3,2));
#else
  tmp0 = _mm256_permute2f128_pd(tmp2, tmp2, 1); // inv hi/lo (inter-lane penalty)
  tmp1 = _mm256_permute2f128_pd(tmp3, tmp3, 1);
#endif
  
#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp0);
  _mm256_storeu_pd(v+4, tmp1);
  print_vec(v, "tmp5");
#endif
  
  // Sort
  min0 = _mm256_min_pd(tmp0, tmp2);
  max0 = _mm256_max_pd(tmp0, tmp2);
  tmp0 = _mm256_blend_pd(min0, max0, 0x0C);
  
  min1 = _mm256_min_pd(tmp1, tmp3);
  max1 = _mm256_max_pd(tmp1, tmp3);
  tmp1 = _mm256_blend_pd(min1, max1, 0x0C);
  
#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp0);
  _mm256_storeu_pd(v+4, tmp1);
  print_vec(v, "sort5"); std::cout << "\n";
#endif
  
  //////// [0,1] [2,3] | [4,5] [6,7]
  // 
  tmp2 = _mm256_shuffle_pd(tmp0, tmp0, 0x05);
  tmp3 = _mm256_shuffle_pd(tmp1, tmp1, 0x05);
  
#ifdef DEBUG_NS8DBL
  _mm256_storeu_pd(v,   tmp2);
  _mm256_storeu_pd(v+4, tmp3);
  print_vec(v, "tmp6");
#endif
  
  // Sort
  min0 = _mm256_min_pd(tmp0, tmp2);
  max0 = _mm256_max_pd(tmp0, tmp2);
  tmp0 = _mm256_blend_pd(min0, max0, 0x0A);
  
  min1 = _mm256_min_pd(tmp1, tmp3);
  max1 = _mm256_max_pd(tmp1, tmp3);
  tmp1 = _mm256_blend_pd(min1, max1, 0x0A);
  
  //
  // Store
  _mm256_storeu_pd(v,   tmp0);
  _mm256_storeu_pd(v+4, tmp1);
  
#ifdef DEBUG_NS8DBL
  print_vec(v, "sort6"); std::cout << "\n";
#endif
}
#endif // HAS_AVX_


#endif // NSORT_8_DBL_H
