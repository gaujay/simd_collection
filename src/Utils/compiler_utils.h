/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef COMPILER_UTILS_H
#define COMPILER_UTILS_H


// 
#if defined(__GNUC__)
  #define DISABLE_FUNC_VECTORIZATION_ __attribute__((optimize("no-tree-vectorize")))
  #define DISABLE_LOOP_VECTORIZATION_

  #ifdef __arm__
    #define IS_ARM_
    #if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__ARM_NEON_FP)
      #define HAS_NEON_
    #endif
    
  #else
    #define IS_X86_
    #ifdef __SSE__
      #define HAS_SSE_
    #endif
    #ifdef __SSE2__
      #define HAS_SSE2_
    #endif
    #ifdef __SSE3__
      #define HAS_SSE3_
    #endif
    #ifdef __SSSE3__
      #define HAS_SSSE3_
    #endif
    #ifdef __SSE4_1__
      #define HAS_SSE4_1_
    #endif
    #ifdef __SSE4_2__
      #define HAS_SSE4_2_
    #endif
    #ifdef __AVX__
      #define HAS_AVX_
    #endif
    #ifdef __AVX2__
      #define HAS_AVX2_
    #endif
    #ifdef __FMA__
      #define HAS_FMA_
    #endif
  #endif

#elif defined(_MSC_VER)
  #define DISABLE_FUNC_VECTORIZATION_
  #define DISABLE_LOOP_VECTORIZATION_ _Pragma("loop(no_vector)")

  #ifdef _M_ARM
    #define IS_ARM_
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
      #define HAS_NEON_
    #endif
    
  #else
    #define IS_X86_
    #ifdef __AVX2__
      #define HAS_AVX2_
      #define HAS_AVX_
      #define HAS_FMA_
      #define HAS_SSE4_2_
      #define HAS_SSE4_1_
      #define HAS_SSSE3_
      #define HAS_SSE3_
      #define HAS_SSE2_
      #define HAS_SSE_
    #elif defined(__AVX__)
      #define HAS_AVX_
      #define HAS_SSE4_2_
      #define HAS_SSE4_1_
      #define HAS_SSSE3_
      #define HAS_SSE3_
      #define HAS_SSE2_
      #define HAS_SSE_
    #elif defined(_M_AMD64) || defined(_M_X64)
      #define HAS_SSE2_
      #define HAS_SSE_
    #elif _M_IX86_FP == 2
      #define HAS_SSE2_
      #define HAS_SSE_
    #elif _M_IX86_FP == 1
      #define HAS_SSE_
    #else
      // Nothing
    #endif
  #endif
#endif


#endif // COMPILER_UTILS_H
