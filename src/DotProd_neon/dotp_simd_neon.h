/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef DOTP_SIMD_NEON_H
#define DOTP_SIMD_NEON_H

#include "Utils/compiler_utils.h"

#ifndef HAS_NEON_
  #error "Minimum SIMD support for ARM is NEON"
#endif

// Data size optimizations
//#define DOTP8_NEON_SIZE_MULTIPLE    16
//#define DOTP168_NEON_SIZE_MULTIPLE  16
//#define DOTP16_NEON_SIZE_MULTIPLE   32
//#define DotP3216_NEON_SIZE_MULTIPLE 16
//#define DOTP32_NEON_SIZE_MULTIPLE   16
//#define DOTPFLT_NEON_SIZE_MULTIPLE  16

//
#include "dotp_i8_neon.h"
#include "dotp_i16i8_neon.h"
#include "dotp_i16_neon.h"
#include "dotp_i32i16_neon.h"
#include "dotp_i32_neon.h"
#include "dotp_flt_neon.h"


// int8 x int8
static inline int32_t dotProduct(int8_t const* __restrict u, int8_t const* __restrict v, size_t n)
{
  return dotProduct_i8_neon(u, v, n);
}

// int16 x int8
static inline int32_t dotProduct(int16_t const* __restrict u, int8_t const* __restrict v, size_t n)
{
  return dotProduct_i16i8_neon(u, v, n);
}

// int16 x int16
static inline int32_t dotProduct(int16_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
  return dotProduct_i16_neon(u, v, n);
}

// int32 x int16
static inline int32_t dotProduct(int32_t const* __restrict u, int16_t const* __restrict v, size_t n)
{
  return dotProduct_i32i16_neon(u, v, n);
}

// int32 x int32
static inline int32_t dotProduct(int32_t const* __restrict u, int32_t const* __restrict v, size_t n)
{
  return dotProduct_i32_neon(u, v, n);
}

// float x float
static inline float dotProduct(float const* __restrict u, float const* __restrict v, size_t n)
{
  return dotProduct_flt_neon(u, v, n);
}


#endif // DOTP_SIMD_NEON_H
