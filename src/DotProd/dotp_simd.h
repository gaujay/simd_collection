/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef DOTP_SIMD_H
#define DOTP_SIMD_H

#include "Utils/compiler_utils.h"

#ifndef HAS_SSE2_
  #error "Minimum SIMD support is SSE2"
#endif

// Data alignment optimizations
//#define DOTP8_SIZE_MULTIPLE     64
//#define DOTP8_128_ALIGNED
//#define DOTP8_256_ALIGNED
//#define DOTP88_SIZE_MULTIPLE    64
//#define DOTP88_128_ALIGNED
//#define DOTP88_256_ALIGNED
//#define DOTP168_SIZE_MULTIPLE   64
//#define DOTP168_128_ALIGNED
//#define DOTP168_256_ALIGNED
//#define DOTP16_SIZE_MULTIPLE    64
//#define DOTP16_128_ALIGNED
//#define DOTP16_256_ALIGNED
//#define DOTP3216_SIZE_MULTIPLE  32
//#define DOTP3216_128_ALIGNED
//#define DOTP3216_256_ALIGNED
//#define DOTP32_SIZE_MULTIPLE    32
//#define DOTP32_128_ALIGNED
//#define DOTP32_256_ALIGNED
//#define DOTPFLT_SIZE_MULTIPLE   32
//#define DOTPFLT_128_ALIGNED
//#define DOTPFLT_256_ALIGNED
//#define DOTPDBL_SIZE_MULTIPLE   16
//#define DOTPDBL_128_ALIGNED
//#define DOTPDBL_256_ALIGNED

//
#include "dotp_i8.h"
#include "dotp_i8ui8.h"
#include "dotp_i16i8.h"
#include "dotp_i16.h"
#include "dotp_i32i16.h"
#include "dotp_i32.h"
#include "dotp_flt.h"
#include "dotp_dbl.h"


// int8 x int8
static inline int32_t dotProduct(int8_t const* __restrict u, int8_t const* __restrict v, size_t n)
{
#ifdef HAS_AVX2_
  return dotProduct_i8_avx2(u, v, n);
#else
  return dotProduct_i8_sse(u, v, n);
#endif
}

// int8 x uint8
static inline int32_t dotProduct(int8_t const* __restrict u, uint8_t const* __restrict v, size_t n)
{
#ifdef HAS_AVX2_
  return dotProduct_i8ui8_avx2(u, v, n);
#else
  return dotProduct_i8ui8_sse(u, v, n);
#endif
}

// int16 x int8
static inline int32_t dotProduct(int16_t const* __restrict u, int8_t const* __restrict v, size_t n)
{
#ifdef HAS_AVX2_
  return dotProduct_i16i8_avx2(u, v, n);
#else
  return dotProduct_i16i8_sse(u, v, n);
#endif
}

// int16 x int16
static inline int32_t dotProduct(int16_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
#ifdef HAS_AVX2_
  return dotProduct_i16_avx2(u, v, n);
#else
  return dotProduct_i16_sse(u, v, n);
#endif
}

// int32 x int16
static inline int32_t dotProduct(int32_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
#ifdef HAS_AVX2_
  return dotProduct_i32i16_avx2(u, v, n);
#else
  return dotProduct_i32i16_sse(u, v, n);
#endif
}

// int32 x int32
static inline int32_t dotProduct(int32_t const* __restrict u, int32_t const* __restrict v, size_t n)
{
#ifdef HAS_AVX2_
  return dotProduct_i32_avx2(u, v, n);
#else
  return dotProduct_i32_sse(u, v, n);
#endif
}

// float x float
static inline float dotProduct(float const* __restrict u, float const* __restrict v, size_t n)
{
#ifdef HAS_FMA_
  return dotProduct_flt_fma(u, v, n);
#elif defined HAS_AVX_
  return dotProduct_flt_avx(u, v, n);
#else
  return dotProduct_flt_sse(u, v, n);
#endif
}

// double x double
static inline double dotProduct(double const* __restrict u, double const* __restrict v, size_t n)
{
#ifdef HAS_FMA_
  return dotProduct_dbl_fma(u, v, n);
#elif defined HAS_AVX_
  return dotProduct_dbl_avx(u, v, n);
#else
  return dotProduct_dbl_sse(u, v, n);
#endif
}


#endif // DOTP_SIMD_H
