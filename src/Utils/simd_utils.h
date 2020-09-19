/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#include "compiler_utils.h"

#include <stdint.h>
#include <emmintrin.h>    // SSE2
#ifdef HAS_SSE3_
  #include <pmmintrin.h>  // SSE3
#endif
#ifdef HAS_SSSE3_
  #include <tmmintrin.h>  // SSSE3
#endif
#ifdef HAS_SSE4_1_
  #include <smmintrin.h>  // SSE4.1
#endif
#ifdef HAS_AVX_
  #include <immintrin.h>  // AVX, AVX2, FMA
#endif


//
inline __m128i extend_lo_epi8(const __m128i a)
{
#ifdef HAS_SSE4_1_
  return _mm_cvtepi8_epi16(a);
#else // SSE2
  __m128i tmp = _mm_unpacklo_epi8(a, a);
  return _mm_srai_epi16(tmp, 8);
#endif
}

//
#ifdef HAS_AVX2_
inline __m256i extend_lo_epi8(const __m256i a)
{
  return _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
}
#endif

//
inline __m128i extend_lo_epi16(const __m128i a)
{
#ifdef HAS_SSE4_1_
  return _mm_cvtepi16_epi32(a);
#else // SSE2
  __m128i tmp = _mm_unpacklo_epi16(a, a);
  return _mm_srai_epi32(tmp, 16);
#endif
}

//
#ifdef HAS_AVX2_
inline __m256i extend_lo_epi16(const __m256i a)
{
  return _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a));
}
#endif

//
inline __m128i extend_hi_epi8(const __m128i a)
{
//#ifdef HAS_SSE4_1_  // May be faster on some (older) architecture
//  __m128i tmp = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
//  return _mm_cvtepi8_epi16(tmp);
//#else // SSE2
  __m128i tmp = _mm_unpackhi_epi8(a, a);
  return _mm_srai_epi16(tmp, 8);
//#endif
}

//
#ifdef HAS_AVX2_
inline __m256i extend_hi_epi8(const __m256i a)
{
  return _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
}
#endif

//
inline __m128i extend_hi_epi16(const __m128i a)
{
//#ifdef HAS_SSE4_1_  // May be faster on some (older) architecture
//  __m128i tmp = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
//  return _mm_cvtepi16_epi32(tmp);
//#else // SSE2
  __m128i tmp = _mm_unpackhi_epi16(a, a);
  return _mm_srai_epi32(tmp, 16);
//#endif
}

//
#ifdef HAS_AVX2_
inline __m256i extend_hi_epi16(const __m256i a)
{
  return _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 1));
}
#endif

//
inline __m128i multiply_lo_epi32(const __m128i a, const __m128i b)
{
#ifdef HAS_SSE4_1_
  return _mm_mullo_epi32(a, b);
#else // SSE2
  __m128i tmp1 = _mm_mul_epu32(a, b); // mul(2,0)
  __m128i tmp2 = _mm_mul_epu32( _mm_srli_si128(a, 4), _mm_srli_si128(b, 4)); // mul(3,1)
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)),
                            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0))); // shuffle results to [63..0] and pack
#endif
}

//
#ifdef HAS_AVX2_
inline __m256i multiply_lo_epi32(const __m256i a, const __m256i b)
{
  return _mm256_mullo_epi32(a, b);
}
#endif

//
inline int32_t horizontal_sum_epi32(const __m128i a)
{
#ifdef HAS_AVX_
  // 3-operand non-destructive AVX lets us save a byte without needing a mov
  __m128i hi64  = _mm_unpackhi_epi64(a, a);
#else // SSE2
  __m128i hi64  = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
#endif
  __m128i sum64 = _mm_add_epi32(hi64, a);
  __m128i hi32  = _mm_shufflelo_epi16(sum64, _MM_SHUFFLE(1, 0, 3, 2));
  __m128i sum32 = _mm_add_epi32(sum64, hi32);
  
  return _mm_cvtsi128_si32(sum32);
}

//
#ifdef HAS_AVX2_
inline int32_t horizontal_sum_epi32(const __m256i a)
{
  __m128i sum128 = _mm_add_epi32( _mm256_castsi256_si128(a),
                                  _mm256_extracti128_si256(a, 1) );
  // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
  __m128i hi64  = _mm_unpackhi_epi64(sum128, sum128);
  __m128i sum64 = _mm_add_epi32(hi64, sum128);
  __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i sum32 = _mm_add_epi32(sum64, hi32);
  
  return _mm_cvtsi128_si32(sum32);
}
#endif

//
inline float horizontal_sum_ps(const __m128 a)
{
#ifdef HAS_SSE3_
  __m128 shf = _mm_movehdup_ps(a);      // broadcast (3,1) to (2,0)
#else // SSE
  __m128 shf = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
#endif
  __m128 sum = _mm_add_ps(a, shf);
  shf        = _mm_movehl_ps(shf, sum); // high half -> low half
  sum        = _mm_add_ss(sum, shf);
  
  return _mm_cvtss_f32(sum);
}

//
#ifdef HAS_AVX_
inline float horizontal_sum_ps(const __m256 a)
{
  __m128 vlo = _mm256_castps256_ps128(a);
  __m128 vhi = _mm256_extractf128_ps(a, 1);
         vlo = _mm_add_ps(vlo, vhi);
         
  return horizontal_sum_ps(vlo);
}
#endif

//
inline double horizontal_sum_pd(const __m128d a)
{
  __m128 und  = _mm_undefined_ps();                   // only use addSD
  __m128 tmp  = _mm_movehl_ps(und, _mm_castpd_ps(a)); // no movhlpd
  __m128d shf = _mm_castps_pd(tmp);
  
  return  _mm_cvtsd_f64(_mm_add_sd(a, shf));
}

//
#ifdef HAS_AVX_
inline double horizontal_sum_pd(const __m256d a)
{
  __m128d vlo = _mm256_castpd256_pd128(a);
  __m128d vhi = _mm256_extractf128_pd(a, 1);
          vlo = _mm_add_pd(vlo, vhi);
         
  return horizontal_sum_pd(vlo);
}
#endif

//
inline __m128i blend_epi8(const __m128i min, const __m128i max, const int mask)
{
  const __m128i mmask = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                     mask&0x80 ? (int8_t)0xFF : 0, mask&0x40 ? (int8_t)0xFF : 0,
                                     mask&0x20 ? (int8_t)0xFF : 0, mask&0x10 ? (int8_t)0xFF : 0,
                                     mask&0x08 ? (int8_t)0xFF : 0, mask&0x04 ? (int8_t)0xFF : 0,
                                     mask&0x02 ? (int8_t)0xFF : 0, mask&0x01 ? (int8_t)0xFF : 0);
  return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
}
inline __m128i blend_epi8_AA(const __m128i min, const __m128i max)
{
  const __m128i mmask = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                     (int8_t)0xFF, 0, (int8_t)0xFF, 0,
                                     (int8_t)0xFF, 0, (int8_t)0xFF, 0);
  return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
}
inline __m128i blend_epi8_CC(const __m128i min, const __m128i max)
{
  const __m128i mmask = _mm_set_epi16(0, 0, 0, 0,
                                      (int16_t)0xFFFF, 0, (int16_t)0xFFFF, 0);
  return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
}
inline __m128i blend_epi8_F0(const __m128i min, const __m128i max)
{
  const __m128i mmask = _mm_set_epi32(0, 0,
                                      (int32_t)0xFFFFFFFF, 0);
  return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
}

//
#ifdef HAS_SSE4_1_
  #define blend_epi16(min, max, mask) _mm_blend_epi16(min, max, mask)
  #define blend_epi16_AA(min, max)    _mm_blend_epi16(min, max, 0xAA)
  #define blend_epi16_CC(min, max)    _mm_blend_epi16(min, max, 0xCC)
  #define blend_epi16_F0(min, max)    _mm_blend_epi16(min, max, 0xF0)
#else // SSE2
  inline __m128i blend_epi16(const __m128i min, const __m128i max, const int mask)
  {
    const __m128i mmask = _mm_set_epi16(mask&0x80 ? (int16_t)0xFFFF : 0, mask&0x40 ? (int16_t)0xFFFF : 0,
                                        mask&0x20 ? (int16_t)0xFFFF : 0, mask&0x10 ? (int16_t)0xFFFF : 0,
                                        mask&0x08 ? (int16_t)0xFFFF : 0, mask&0x04 ? (int16_t)0xFFFF : 0,
                                        mask&0x02 ? (int16_t)0xFFFF : 0, mask&0x01 ? (int16_t)0xFFFF : 0);
    return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
  }
  inline __m128i blend_epi16_AA(const __m128i min, const __m128i max)
  {
    const __m128i mmask = _mm_set_epi16((int16_t)0xFFFF, 0, (int16_t)0xFFFF, 0,
                                        (int16_t)0xFFFF, 0, (int16_t)0xFFFF, 0);
    return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
  }
  inline __m128i blend_epi16_CC(const __m128i min, const __m128i max)
  {
    const __m128i mmask = _mm_set_epi32((int)0xFFFFFFFF, 0, (int)0xFFFFFFFF, 0);
    return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
  }
  inline __m128i blend_epi16_F0(const __m128i min, const __m128i max)
  {
    const __m128i mmask = _mm_set_epi64x((int64_t)0xFFFFFFFFFFFFFFFF, 0);
    return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
  }
#endif

//
#ifdef HAS_AVX2_ // TODO: test
  #define blend_epi32(min, max, mask) _mm_blend_epi32(min, max, mask)
  #define blend_epi32_0A(min, max)    _mm_blend_epi32(min, max, 0x0A)
  #define blend_epi32_0C(min, max)    _mm_blend_epi32(min, max, 0x0C)
  #define blend_epi32_00(min, max)    _mm_blend_epi32(min, max, 0x00)
  #define blend_epi32_0F(min, max)    _mm_blend_epi32(min, max, 0x0F)
#else // SSE2
  inline __m128i blend_epi32(const __m128i min, const __m128i max, const int mask)
  {
    const __m128i mmask = _mm_set_epi32(mask&0x08 ? (int)0xFFFFFFFF : 0, mask&0x04 ? (int)0xFFFFFFFF : 0,
                                        mask&0x02 ? (int)0xFFFFFFFF : 0, mask&0x01 ? (int)0xFFFFFFFF : 0);
    return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
  }
  inline __m128i blend_epi32_0A(const __m128i min, const __m128i max)
  {
    const __m128i mmask = _mm_set_epi32((int)0xFFFFFFFF, 0, (int)0xFFFFFFFF, 0);
    return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
  }
  inline __m128i blend_epi32_0C(const __m128i min, const __m128i max)
  {
    const __m128i mmask = _mm_set_epi32((int)0xFFFFFFFF, (int)0xFFFFFFFF, 0, 0);
    return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
  }
  inline __m128i blend_epi32_00(const __m128i min, const __m128i max)
  {
    const __m128i mmask = _mm_set1_epi32(0);
    return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
  }
  inline __m128i blend_epi32_0F(const __m128i min, const __m128i max)
  {
    const __m128i mmask = _mm_set1_epi32((int)0xFFFFFFFF);
    return _mm_or_si128(_mm_andnot_si128(mmask, min), _mm_and_si128(mmask, max));
  }
#endif

//
#ifdef HAS_SSE4_1_
  #define blend_ps(min, max, mask)  _mm_blend_ps(min, max, mask)
  #define blend_ps_0A(min, max)     _mm_blend_ps(min, max, 0x0A)
  #define blend_ps_0C(min, max)     _mm_blend_ps(min, max, 0x0C)
  #define blend_ps_00(min, max)     _mm_blend_ps(min, max, 0x00)
  #define blend_ps_0F(min, max)     _mm_blend_ps(min, max, 0x0F)
#else // SSE2
  inline __m128 blend_ps(const __m128 min, const __m128 max, const int mask)
  {
    const __m128 mmask = _mm_castsi128_ps(
          _mm_set_epi32(-(mask&0x08), -(mask&0x04), -(mask&0x02), -(mask&0x01)) );
    return _mm_or_ps(_mm_andnot_ps(mmask, min), _mm_and_ps(mmask, max));
  }
  inline __m128 blend_ps_0A(const __m128 min, const __m128 max)
  {
    const __m128 mmask = _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));
    return _mm_or_ps(_mm_andnot_ps(mmask, min), _mm_and_ps(mmask, max));
  }
  inline __m128 blend_ps_0C(const __m128 min, const __m128 max)
  {
    const __m128 mmask = _mm_castsi128_ps(_mm_set_epi32(-1, -1, 0, 0));
    return _mm_or_ps(_mm_andnot_ps(mmask, min), _mm_and_ps(mmask, max));
  }
  inline __m128 blend_ps_00(const __m128 min, const __m128 max)
  {
    const __m128 mmask = _mm_castsi128_ps(_mm_set1_epi32(0));
    return _mm_or_ps(_mm_andnot_ps(mmask, min), _mm_and_ps(mmask, max));
  }
  inline __m128 blend_ps_0F(const __m128 min, const __m128 max)
  {
    const __m128 mmask = _mm_castsi128_ps(_mm_set1_epi32(-1));
    return _mm_or_ps(_mm_andnot_ps(mmask, min), _mm_and_ps(mmask, max));
  }
#endif


#endif // SIMD_UTILS_H
