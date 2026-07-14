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
* FEAT_I8MM byte square NxN convolution, N in {5,7,9,11} -- a faster interior for
* the byte path, dispatched only when every coefficient fits int8 (conv_int8) and
* the CPU reports i8mm.
*
* This is the ARM analog of square_vnni_impl.h: usdot is the vpdpbusd equivalent,
* reducing 4 uint8*int8 products into an int32 lane in one op with no byte->word
* widening, and vqtbl1q_u8 plays the role of vpermb -- it builds the sliding
* window so that 32-bit lane l holds the 4 pixels output j+l needs. The
* accumulator therefore stays linear (lane l = output j+l) and the store needs no
* un-scramble.
*
* Bit-exact with sq_interior_byte: the products and the int32 sum are the same
* integers (integer addition is associative, so the regrouping into 4-tap chunks
* cannot change the result), and the scale/bias/round/clamp reuses the same
* nc_store_u8x8. |coeff| <= 127 under conv_int8, so 121 * 127 * 255 cannot
* overflow int32.
*
* Coefficients are zero-padded to a multiple of 4 taps per row, so the window can
* read up to 4G-N taps past the kernel; those pixels are multiplied by zero, but
* they are still *read*, so the SIMD interior is bounded to keep every load inside
* the row and the driver's scalar edge covers whatever is left.
*
* Compiled in its own TU with -march=...+i8mm (see meson.build). NEON baseline
* code must not be built with +i8mm, hence the separate TU rather than a target
* attribute.
*/

#include <cstdint>
#include <cstring>
#include "../generic.h"
#include "conv_scalar.h"
#include "neon_common.h"

namespace {

// Window indices: 32-bit lane l gets pixels {l, l+1, l+2, l+3} of the 8-byte load.
alignas(16) const uint8_t NC_DOT_WIN[16] = {
    0, 1, 2, 3,
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,
};

template <unsigned N>
unsigned sq_interior_byte_dot(const uint8_t *const *rows, uint8_t *dst, unsigned S, unsigned W,
                              const int32_t *cg, float32x4_t sc, float32x4_t bi, uint32x4_t sm)
{
    constexpr unsigned G = (N + 3) / 4;                 // 4-tap groups per row
    const uint8x16_t widx = vld1q_u8(NC_DOT_WIN);

    auto block = [&](unsigned jj) {
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
        for (unsigned r = 0; r < N; ++r) {
            const uint8_t *row = rows[r] + (jj - S);
            for (unsigned g = 0; g < G; ++g) {
                // The group's 4 coefficients, replicated into every 32-bit lane.
                const int8x16_t w = vreinterpretq_s8_s32(vdupq_n_s32(cg[r * G + g]));
                const uint8_t *q = row + 4 * g;
                const uint8x16_t z = vdupq_n_u8(0);
                uint8x16_t t0 = vqtbl1q_u8(vcombine_u8(vld1_u8(q + 0), vget_low_u8(z)), widx);
                uint8x16_t t1 = vqtbl1q_u8(vcombine_u8(vld1_u8(q + 4), vget_low_u8(z)), widx);
                uint8x16_t t2 = vqtbl1q_u8(vcombine_u8(vld1_u8(q + 8), vget_low_u8(z)), widx);
                uint8x16_t t3 = vqtbl1q_u8(vcombine_u8(vld1_u8(q + 12), vget_low_u8(z)), widx);
                a0 = vusdotq_s32(a0, t0, w);
                a1 = vusdotq_s32(a1, t1, w);
                a2 = vusdotq_s32(a2, t2, w);
                a3 = vusdotq_s32(a3, t3, w);
            }
        }
        nc_store_u8x8(dst + jj, a0, a1, sc, bi, sm);
        nc_store_u8x8(dst + jj + 8, a2, a3, sc, bi, sm);
    };

    const unsigned end = W > S ? W - S : 0;
    if (end < S + 16)
        return S;

    // Highest block start whose furthest load, row[(jj - S) + 4(G-1) + 12 + 7],
    // still lands inside the row.
    const long maxjj = static_cast<long>(W) + static_cast<long>(S) - 4L * static_cast<long>(G) - 16L;
    if (maxjj < static_cast<long>(S))
        return S;

    unsigned j = S;
    for (; j + 16 <= end && static_cast<long>(j) <= maxjj; j += 16)
        block(j);

    // Overlapping final block for the sub-vector remainder; re-storing already
    // written columns is safe because this path is bit-exact with the scalar one.
    if (j < end) {
        long c = static_cast<long>(end) - 16;
        if (c > maxjj)
            c = maxjj;
        if (c >= static_cast<long>(S) && static_cast<unsigned>(c) + 16 > j) {
            block(static_cast<unsigned>(c));
            j = static_cast<unsigned>(c) + 16;
        }
    }
    return j;
}

template <unsigned N>
void sq_plane_byte_dot(const void *src, ptrdiff_t ss, void *dst, ptrdiff_t ds,
                       const vs_generic_params &p, unsigned W, unsigned H)
{
    constexpr unsigned S = N / 2;
    constexpr unsigned G = (N + 3) / 4;

    // Pack each row's taps into 4-tap groups of int8, zero-filling the tail.
    int32_t cg[N * G];
    for (unsigned r = 0; r < N; ++r) {
        for (unsigned g = 0; g < G; ++g) {
            int8_t b[4];
            for (unsigned t = 0; t < 4; ++t) {
                const unsigned k = 4 * g + t;
                b[t] = k < N ? static_cast<int8_t>(p.matrix[r * N + k]) : 0;
            }
            std::memcpy(&cg[r * G + g], b, 4);
        }
    }

    const float32x4_t sc = vdupq_n_f32(p.div), bi = vdupq_n_f32(p.bias);
    const uint32x4_t sm = nc_satmask(p.saturate);

    for (unsigned i = 0; i < H; ++i) {
        const uint8_t *rows[N];
        for (unsigned r = 0; r < N; ++r)
            rows[r] = static_cast<const uint8_t *>(src) +
                static_cast<ptrdiff_t>(nc_mirror(static_cast<int>(i) + static_cast<int>(r) - static_cast<int>(S), static_cast<int>(H))) * ss;
        uint8_t *d = static_cast<uint8_t *>(dst) + static_cast<ptrdiff_t>(i) * ds;

        const unsigned aend = sq_interior_byte_dot<N>(rows, d, S, W, cg, sc, bi, sm);

        for (unsigned j = 0; j < S && j < W; ++j)
            d[j] = nc_sq_scalar_px<uint8_t, int32_t, int16_t, N>(rows, j, S, W, p.matrix, p.div, p.bias, p.saturate, p.maxval);
        for (unsigned j = (aend > S ? aend : S); j < W; ++j)
            d[j] = nc_sq_scalar_px<uint8_t, int32_t, int16_t, N>(rows, j, S, W, p.matrix, p.div, p.bias, p.saturate, p.maxval);
    }
}

} // namespace

#define VS_SQUARE_DOT_ENTRY(SZ, N) \
    void vs_generic_##SZ##_conv_byte_neon_dot(const void *src, ptrdiff_t src_stride, void *dst, ptrdiff_t dst_stride, const struct vs_generic_params *params, unsigned width, unsigned height) \
    { sq_plane_byte_dot<N>(src, src_stride, dst, dst_stride, *params, width, height); }

VS_SQUARE_DOT_ENTRY(5x5, 5)
VS_SQUARE_DOT_ENTRY(7x7, 7)
VS_SQUARE_DOT_ENTRY(9x9, 9)
VS_SQUARE_DOT_ENTRY(11x11, 11)
