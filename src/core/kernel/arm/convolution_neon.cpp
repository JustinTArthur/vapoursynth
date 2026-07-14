/*
* Copyright (c) 2012-2026 Fredrik Mellbin
*
* This file is part of VapourSynth.
*
* VapourSynth is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* VapourSynth is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with VapourSynth; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

/*
* NEON 1D (horizontal/vertical/separable) and 3x3 square convolution.
*
* Semantics track kernel/generic.cpp exactly:
*  - 1D edges use the half-sample mirror formulas of conv_scanline_h /
*    conv_plane_v (replicated verbatim in the scalar edge paths).
*  - 3x3 uses replicate edges (filter_plane_3x3 row/column selection).
*  - The separable path quantises the vertical pass into a plane-typed tmp
*    scanline before the horizontal pass, like conv_plane_x.
*
* Integer accumulation:
*  - byte: u8 widens to s16, per-tap vmlal_s16 into int32. 25 * 1023 * 255
*    never overflows. Bit-exact with the int32 C reference.
*  - word: u16 biased by INT16_MIN into s16; biased worst case
*    25 * 1023 * 32768 fits int32, and the coefficient*bias sum is subtracted
*    afterwards. int32 wraparound arithmetic makes this exact, so results are
*    bit-exact with the (unbiased) int32 C reference.
*  - float/half: FMA, not bit-exact with C (allowed for float formats).
*/

#include <cstdint>
#include "VSHelper4.h"
#include "../generic.h"
#include "neon_common.h"

namespace {

enum class CvType { Byte, Word, Float, Half };

template <CvType TY>
struct cv_traits;

template <> struct cv_traits<CvType::Byte>  { typedef uint8_t T; typedef int32_t Acc; typedef int16_t Weight; };
template <> struct cv_traits<CvType::Word>  { typedef uint16_t T; typedef int32_t Acc; typedef int16_t Weight; };
template <> struct cv_traits<CvType::Float> { typedef float T; typedef float Acc; typedef float Weight; };
template <> struct cv_traits<CvType::Half>  { typedef uint16_t T; typedef float Acc; typedef float Weight; };

template <CvType TY>
inline const typename cv_traits<TY>::Weight *cv_coeffs(const vs_generic_params &p)
{
    if constexpr (TY == CvType::Byte || TY == CvType::Word)
        return p.matrix;
    else
        return p.matrixf;
}

template <CvType TY>
inline typename cv_traits<TY>::Acc cv_sample(typename cv_traits<TY>::T x)
{
    if constexpr (TY == CvType::Half)
        return nc_half_to_float(x);
    else
        return static_cast<typename cv_traits<TY>::Acc>(x);
}

// Final scalar store; matches limit(xrint<T>(tmp), maxval) in the C reference.
template <CvType TY>
inline typename cv_traits<TY>::T cv_scalar_store(float accum, float div, float bias, bool saturate, uint16_t maxval)
{
    float tmp = accum * div + bias;
    tmp = saturate ? tmp : std::fabs(tmp);
    if constexpr (TY == CvType::Byte) {
        float c = std::min(std::max(tmp, 0.0f), 255.0f);
        long v = std::lrint(c);
        return static_cast<uint8_t>(std::min<long>(v, maxval));
    } else if constexpr (TY == CvType::Word) {
        float c = std::min(std::max(tmp, 0.0f), 65535.0f);
        long v = std::lrint(c);
        return static_cast<uint16_t>(std::min<long>(v, maxval));
    } else if constexpr (TY == CvType::Float) {
        return tmp;
    } else {
        return nc_float_to_half(tmp);
    }
}

// ---- vectorised tap accumulation over one 16-pixel block --------------------
// Each *_tap16 accumulates coefficient k of a row pointer into 4 chains.

struct acc16_i32 {
    int32x4_t a0, a1, a2, a3;
    void clear() { a0 = a1 = a2 = a3 = vdupq_n_s32(0); }
};

struct acc16_f32 {
    float32x4_t a0, a1, a2, a3;
    void clear() { a0 = a1 = a2 = a3 = vdupq_n_f32(0); }
};

inline void byte_tap16(acc16_i32 &a, const uint8_t *p, int16_t w)
{
    nc_byte16 x = nc_load_byte16(p);
    int16x4_t wv = vdup_n_s16(w);
    a.a0 = vmlal_s16(a.a0, vget_low_s16(x.lo), wv);
    a.a1 = vmlal_s16(a.a1, vget_high_s16(x.lo), wv);
    a.a2 = vmlal_s16(a.a2, vget_low_s16(x.hi), wv);
    a.a3 = vmlal_s16(a.a3, vget_high_s16(x.hi), wv);
}

inline void word_tap16(acc16_i32 &a, const uint16_t *p, int16_t w)
{
    int16x8_t x0 = nc_load_word_biased(p);
    int16x8_t x1 = nc_load_word_biased(p + 8);
    int16x4_t wv = vdup_n_s16(w);
    a.a0 = vmlal_s16(a.a0, vget_low_s16(x0), wv);
    a.a1 = vmlal_s16(a.a1, vget_high_s16(x0), wv);
    a.a2 = vmlal_s16(a.a2, vget_low_s16(x1), wv);
    a.a3 = vmlal_s16(a.a3, vget_high_s16(x1), wv);
}

inline void float_tap16(acc16_f32 &a, const float *p, float w)
{
    float32x4_t wv = vdupq_n_f32(w);
    a.a0 = vfmaq_f32(a.a0, vld1q_f32(p), wv);
    a.a1 = vfmaq_f32(a.a1, vld1q_f32(p + 4), wv);
    a.a2 = vfmaq_f32(a.a2, vld1q_f32(p + 8), wv);
    a.a3 = vfmaq_f32(a.a3, vld1q_f32(p + 12), wv);
}

inline void half_tap16(acc16_f32 &a, const uint16_t *p, float w)
{
    nc_half8 x0 = nc_load_half8(p);
    nc_half8 x1 = nc_load_half8(p + 8);
    float32x4_t wv = vdupq_n_f32(w);
    a.a0 = vfmaq_f32(a.a0, x0.lo, wv);
    a.a1 = vfmaq_f32(a.a1, x0.hi, wv);
    a.a2 = vfmaq_f32(a.a2, x1.lo, wv);
    a.a3 = vfmaq_f32(a.a3, x1.hi, wv);
}

template <CvType TY>
inline void store16(typename cv_traits<TY>::T *dst,
                    typename std::conditional<TY == CvType::Byte || TY == CvType::Word, acc16_i32, acc16_f32>::type &a,
                    int32_t wb, float32x4_t sc, float32x4_t bi, uint32x4_t sm, uint16x8_t mv)
{
    if constexpr (TY == CvType::Byte) {
        nc_store_u8x8(dst, a.a0, a.a1, sc, bi, sm);
        nc_store_u8x8(dst + 8, a.a2, a.a3, sc, bi, sm);
    } else if constexpr (TY == CvType::Word) {
        int32x4_t wbv = vdupq_n_s32(wb);
        nc_store_u16x8(dst, vsubq_s32(a.a0, wbv), vsubq_s32(a.a1, wbv), sc, bi, sm, mv);
        nc_store_u16x8(dst + 8, vsubq_s32(a.a2, wbv), vsubq_s32(a.a3, wbv), sc, bi, sm, mv);
    } else if constexpr (TY == CvType::Float) {
        nc_store_f32x4(dst, a.a0, sc, bi, sm);
        nc_store_f32x4(dst + 4, a.a1, sc, bi, sm);
        nc_store_f32x4(dst + 8, a.a2, sc, bi, sm);
        nc_store_f32x4(dst + 12, a.a3, sc, bi, sm);
    } else {
        nc_store_h16x8(dst, a.a0, a.a1, sc, bi, sm);
        nc_store_h16x8(dst + 8, a.a2, a.a3, sc, bi, sm);
    }
}

template <CvType TY>
inline void tap16(typename std::conditional<TY == CvType::Byte || TY == CvType::Word, acc16_i32, acc16_f32>::type &a,
                  const typename cv_traits<TY>::T *p, typename cv_traits<TY>::Weight w)
{
    if constexpr (TY == CvType::Byte)
        byte_tap16(a, p, w);
    else if constexpr (TY == CvType::Word)
        word_tap16(a, p, w);
    else if constexpr (TY == CvType::Float)
        float_tap16(a, p, w);
    else
        half_tap16(a, p, w);
}

// Sum of coeff * INT16_MIN over the taps in use; subtracted from biased word
// accumulators before the store. Zero for the other formats.
template <CvType TY>
inline int32_t word_bias_sum(const typename cv_traits<TY>::Weight *coeffs, unsigned n)
{
    if constexpr (TY == CvType::Word) {
        int32_t wb = 0;
        for (unsigned i = 0; i < n; ++i)
            wb += static_cast<int32_t>(INT16_MIN) * coeffs[i];
        return wb;
    } else {
        (void)coeffs; (void)n;
        return 0;
    }
}

// ---- 1D horizontal ----------------------------------------------------------

// Scalar mirror-edge pixel; replicates the conv_scanline_h formulas verbatim.
template <CvType TY>
inline typename cv_traits<TY>::T conv_h_edge_px(const typename cv_traits<TY>::T *srcp, unsigned j, unsigned width,
                                                const vs_generic_params &p)
{
    typedef typename cv_traits<TY>::Acc Acc;
    const auto *coeffs = cv_coeffs<TY>(p);
    unsigned fwidth = p.matrixsize;
    unsigned support = fwidth / 2;
    unsigned dist_from_right = width - 1 - j;

    Acc accum = 0;
    for (unsigned k = 0; k < support; ++k) {
        unsigned idx = j < support - k ? std::min(support - k - j - 1, width - 1) : j - support + k;
        accum += coeffs[k] * cv_sample<TY>(srcp[idx]);
    }
    for (unsigned k = support; k < fwidth; ++k) {
        unsigned idx = dist_from_right < k - support ? width - std::min(k - support - dist_from_right, width) : j - support + k;
        accum += coeffs[k] * cv_sample<TY>(srcp[idx]);
    }
    return cv_scalar_store<TY>(static_cast<float>(accum), p.div, p.bias, p.saturate, p.maxval);
}

template <CvType TY>
void conv_h_scanline(const typename cv_traits<TY>::T *srcp, typename cv_traits<TY>::T *dstp, unsigned width,
                     const vs_generic_params &p, float32x4_t sc, float32x4_t bi, uint32x4_t sm, uint16x8_t mv, int32_t wb)
{
    const auto *coeffs = cv_coeffs<TY>(p);
    unsigned fwidth = p.matrixsize;
    unsigned support = fwidth / 2;

    for (unsigned j = 0; j < std::min(width, support); ++j)
        dstp[j] = conv_h_edge_px<TY>(srcp, j, width, p);

    unsigned end = width - std::min(width, support);
    unsigned j = support;
    auto block = [&](unsigned jj) {
        typename std::conditional<TY == CvType::Byte || TY == CvType::Word, acc16_i32, acc16_f32>::type a;
        a.clear();
        for (unsigned k = 0; k < fwidth; ++k)
            tap16<TY>(a, srcp + jj - support + k, coeffs[k]);
        store16<TY>(dstp + jj, a, wb, sc, bi, sm, mv);
    };
    if (end > support) {
        for (; j + 16 <= end; j += 16) block(j);
        if (j < end && end >= support + 16) { block(end - 16); j = end; }
        for (; j < end; ++j)
            dstp[j] = conv_h_edge_px<TY>(srcp, j, width, p);
    }

    for (unsigned jj = std::max(support, end); jj < width; ++jj)
        dstp[jj] = conv_h_edge_px<TY>(srcp, jj, width, p);
}

// ---- 1D vertical --------------------------------------------------------------

template <CvType TY>
void conv_v_scanline(const void *const *srcs, typename cv_traits<TY>::T *dstp, unsigned width,
                     const vs_generic_params &p, float32x4_t sc, float32x4_t bi, uint32x4_t sm, uint16x8_t mv, int32_t wb)
{
    typedef typename cv_traits<TY>::T T;
    typedef typename cv_traits<TY>::Acc Acc;
    const auto *coeffs = cv_coeffs<TY>(p);
    unsigned fwidth = p.matrixsize;

    unsigned j = 0;
    auto block = [&](unsigned jj) {
        typename std::conditional<TY == CvType::Byte || TY == CvType::Word, acc16_i32, acc16_f32>::type a;
        a.clear();
        for (unsigned k = 0; k < fwidth; ++k)
            tap16<TY>(a, static_cast<const T *>(srcs[k]) + jj, coeffs[k]);
        store16<TY>(dstp + jj, a, wb, sc, bi, sm, mv);
    };
    for (; j + 16 <= width; j += 16) block(j);
    if (j < width && width >= 16) { block(width - 16); j = width; }
    for (; j < width; ++j) {
        Acc accum = 0;
        for (unsigned k = 0; k < fwidth; ++k)
            accum += coeffs[k] * cv_sample<TY>(static_cast<const T *>(srcs[k])[j]);
        dstp[j] = cv_scalar_store<TY>(static_cast<float>(accum), p.div, p.bias, p.saturate, p.maxval);
    }
}

// Mirror row-pointer selection of conv_plane_v / conv_plane_x, verbatim.
inline void conv_v_select_rows(const void *src, ptrdiff_t src_stride, const void *srcp[25],
                               unsigned i, unsigned height, unsigned fwidth)
{
    unsigned support = fwidth / 2;
    unsigned dist_from_bottom = height - 1 - i;

    for (unsigned k = 0; k < support; ++k) {
        unsigned row = i < support - k ? std::min(support - k - i - 1, height - 1) : i - support + k;
        srcp[k] = static_cast<const unsigned char *>(src) + static_cast<ptrdiff_t>(row) * src_stride;
    }
    for (unsigned k = support; k < fwidth; ++k) {
        unsigned row = dist_from_bottom < k - support ? height - std::min(k - support - dist_from_bottom, i) : i - support + k;
        srcp[k] = static_cast<const unsigned char *>(src) + static_cast<ptrdiff_t>(row) * src_stride;
    }
}

// ---- plane drivers ------------------------------------------------------------

template <CvType TY>
void conv_plane_h_neon(const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride,
                       const vs_generic_params &p, unsigned width, unsigned height)
{
    typedef typename cv_traits<TY>::T T;
    float32x4_t sc = vdupq_n_f32(p.div), bi = vdupq_n_f32(p.bias);
    uint32x4_t sm = nc_satmask(p.saturate);
    uint16x8_t mv = vdupq_n_u16(p.maxval);
    int32_t wb = word_bias_sum<TY>(cv_coeffs<TY>(p), p.matrixsize);

    for (unsigned i = 0; i < height; ++i) {
        const T *srcp = reinterpret_cast<const T *>(static_cast<const unsigned char *>(src) + static_cast<ptrdiff_t>(i) * src_stride);
        T *dstp = reinterpret_cast<T *>(static_cast<unsigned char *>(dst) + static_cast<ptrdiff_t>(i) * dst_stride);
        conv_h_scanline<TY>(srcp, dstp, width, p, sc, bi, sm, mv, wb);
    }
}

template <CvType TY>
void conv_plane_v_neon(const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride,
                       const vs_generic_params &p, unsigned width, unsigned height)
{
    typedef typename cv_traits<TY>::T T;
    float32x4_t sc = vdupq_n_f32(p.div), bi = vdupq_n_f32(p.bias);
    uint32x4_t sm = nc_satmask(p.saturate);
    uint16x8_t mv = vdupq_n_u16(p.maxval);
    int32_t wb = word_bias_sum<TY>(cv_coeffs<TY>(p), p.matrixsize);

    for (unsigned i = 0; i < height; ++i) {
        const void *srcp[25];
        conv_v_select_rows(src, src_stride, srcp, i, height, p.matrixsize);
        T *dstp = reinterpret_cast<T *>(static_cast<unsigned char *>(dst) + static_cast<ptrdiff_t>(i) * dst_stride);
        conv_v_scanline<TY>(srcp, dstp, width, p, sc, bi, sm, mv, wb);
    }
}

template <CvType TY>
void conv_plane_x_neon(const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride,
                       const vs_generic_params &p, unsigned width, unsigned height)
{
    typedef typename cv_traits<TY>::T T;
    float32x4_t sc = vdupq_n_f32(p.div), bi = vdupq_n_f32(p.bias);
    uint32x4_t sm = nc_satmask(p.saturate);
    uint16x8_t mv = vdupq_n_u16(p.maxval);
    int32_t wb = word_bias_sum<TY>(cv_coeffs<TY>(p), p.matrixsize);

    T *tmp = static_cast<T *>(vsh::vsh_aligned_malloc(width * sizeof(T), 64));
    if (!tmp)
        return;

    for (unsigned i = 0; i < height; ++i) {
        const void *srcp[25];
        conv_v_select_rows(src, src_stride, srcp, i, height, p.matrixsize);
        T *dstp = reinterpret_cast<T *>(static_cast<unsigned char *>(dst) + static_cast<ptrdiff_t>(i) * dst_stride);
        conv_v_scanline<TY>(srcp, tmp, width, p, sc, bi, sm, mv, wb);
        conv_h_scanline<TY>(tmp, dstp, width, p, sc, bi, sm, mv, wb);
    }

    vsh::vsh_aligned_free(tmp);
}

// ---- 3x3 square (replicate edges, filter_plane_3x3 semantics) ------------------

template <CvType TY>
void conv_plane_3x3_neon(const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride,
                         const vs_generic_params &p, unsigned width, unsigned height)
{
    typedef typename cv_traits<TY>::T T;
    typedef typename cv_traits<TY>::Acc Acc;
    const auto *coeffs = cv_coeffs<TY>(p);
    float32x4_t sc = vdupq_n_f32(p.div), bi = vdupq_n_f32(p.bias);
    uint32x4_t sm = nc_satmask(p.saturate);
    uint16x8_t mv = vdupq_n_u16(p.maxval);
    int32_t wb = word_bias_sum<TY>(coeffs, 9);

    for (unsigned i = 0; i < height; ++i) {
        unsigned above_idx = i == 0 ? 0 : i - 1;
        unsigned below_idx = i == height - 1 ? height - 1 : i + 1;
        const T *rows[3] = {
            reinterpret_cast<const T *>(static_cast<const unsigned char *>(src) + static_cast<ptrdiff_t>(above_idx) * src_stride),
            reinterpret_cast<const T *>(static_cast<const unsigned char *>(src) + static_cast<ptrdiff_t>(i) * src_stride),
            reinterpret_cast<const T *>(static_cast<const unsigned char *>(src) + static_cast<ptrdiff_t>(below_idx) * src_stride),
        };
        T *dstp = reinterpret_cast<T *>(static_cast<unsigned char *>(dst) + static_cast<ptrdiff_t>(i) * dst_stride);

        auto scalar_px = [&](unsigned a, unsigned b, unsigned c) -> T {
            Acc accum = 0;
            for (unsigned r = 0; r < 3; ++r) {
                accum += coeffs[r * 3 + 0] * cv_sample<TY>(rows[r][a]);
                accum += coeffs[r * 3 + 1] * cv_sample<TY>(rows[r][b]);
                accum += coeffs[r * 3 + 2] * cv_sample<TY>(rows[r][c]);
            }
            return cv_scalar_store<TY>(static_cast<float>(accum), p.div, p.bias, p.saturate, p.maxval);
        };

        dstp[0] = scalar_px(0, 0, width > 1 ? 1 : 0);

        unsigned end = width > 1 ? width - 1 : 0;
        unsigned j = 1;
        auto block = [&](unsigned jj) {
            typename std::conditional<TY == CvType::Byte || TY == CvType::Word, acc16_i32, acc16_f32>::type a;
            a.clear();
            for (unsigned r = 0; r < 3; ++r)
                for (unsigned k = 0; k < 3; ++k)
                    tap16<TY>(a, rows[r] + jj - 1 + k, coeffs[r * 3 + k]);
            store16<TY>(dstp + jj, a, wb, sc, bi, sm, mv);
        };
        if (end > 1) {
            for (; j + 16 <= end; j += 16) block(j);
            if (j < end && end >= 1 + 16) { block(end - 16); j = end; }
            for (; j < end; ++j) {
                unsigned a = j - 1, b = j, c = j + 1;
                dstp[j] = scalar_px(a, b, c);
            }
        }

        if (width > 1)
            dstp[width - 1] = scalar_px(width - 2, width - 1, width - 1);
    }
}

} // namespace

#define VS_CONV_NEON_ENTRY(KERNEL, FN, TY, TN) \
    void vs_generic_##KERNEL##_##TN##_neon(const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride, const struct vs_generic_params *params, unsigned width, unsigned height) \
    { FN<CvType::TY>(src, src_stride, dst, dst_stride, *params, width, height); }

VS_CONV_NEON_ENTRY(3x3_conv, conv_plane_3x3_neon, Byte, byte)
VS_CONV_NEON_ENTRY(3x3_conv, conv_plane_3x3_neon, Word, word)
VS_CONV_NEON_ENTRY(3x3_conv, conv_plane_3x3_neon, Float, float)
VS_CONV_NEON_ENTRY(3x3_conv, conv_plane_3x3_neon, Half, half)

VS_CONV_NEON_ENTRY(1d_conv_h, conv_plane_h_neon, Byte, byte)
VS_CONV_NEON_ENTRY(1d_conv_h, conv_plane_h_neon, Word, word)
VS_CONV_NEON_ENTRY(1d_conv_h, conv_plane_h_neon, Float, float)
VS_CONV_NEON_ENTRY(1d_conv_h, conv_plane_h_neon, Half, half)

VS_CONV_NEON_ENTRY(1d_conv_v, conv_plane_v_neon, Byte, byte)
VS_CONV_NEON_ENTRY(1d_conv_v, conv_plane_v_neon, Word, word)
VS_CONV_NEON_ENTRY(1d_conv_v, conv_plane_v_neon, Float, float)
VS_CONV_NEON_ENTRY(1d_conv_v, conv_plane_v_neon, Half, half)

VS_CONV_NEON_ENTRY(2d_conv_sep, conv_plane_x_neon, Byte, byte)
VS_CONV_NEON_ENTRY(2d_conv_sep, conv_plane_x_neon, Word, word)
VS_CONV_NEON_ENTRY(2d_conv_sep, conv_plane_x_neon, Float, float)
VS_CONV_NEON_ENTRY(2d_conv_sep, conv_plane_x_neon, Half, half)
