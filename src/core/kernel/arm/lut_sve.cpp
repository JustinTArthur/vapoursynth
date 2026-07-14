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
* SVE gather Lut1 for 16-bit sources.
*
* A 16-bit source indexes a 65536-entry table, which is far too large for the
* byte path's tbl/vpermb trick, so every ISA runs this scalar -- x86 included,
* where the only Lut1 SIMD kernel is the AVX-512 VBMI byte->byte one. SVE can do
* it with a real gather: widen the 16-bit pixels into 32-bit lanes and use them
* as indices, so one gather retires svcntw() lookups.
*
* Bit-exact by construction: it is the same table read, just several at a time.
*
* Whether it beats the scalar loop depends on how many lookups a gather retires,
* which is svcntw() and therefore vector-length dependent (measured, 65536-entry
* table, random indices):
*
*             word->word   word->byte
*   256-bit      +55%         +75%      (Graviton3, 8 lookups/gather)
*   128-bit       -7%         +36%      (Graviton4, 4 lookups/gather)
*
* so word->word is gated on a vector length above 128 bits and word->byte is not.
* The narrower destination wins at both because the store narrows 4:1 rather than
* 2:1, which is where the scalar loop spends its time.
*/

#include <arm_sve.h>
#include <cstdint>

void vs_lut1_w_w_sve(const uint16_t *src, uint16_t *dst, int w, const uint16_t *lut)
{
    const unsigned n = static_cast<unsigned>(w);
    const unsigned step = static_cast<unsigned>(svcntw());
    for (unsigned i = 0; i < n; i += step) {
        svbool_t pg = svwhilelt_b32_u32(i, n);
        svuint32_t idx = svld1uh_u32(pg, src + i);
        svuint32_t v = svld1uh_gather_u32index_u32(pg, lut, idx);
        svst1h_u32(pg, dst + i, v);
    }
}

void vs_lut1_w_b_sve(const uint16_t *src, uint8_t *dst, int w, const uint8_t *lut)
{
    const unsigned n = static_cast<unsigned>(w);
    const unsigned step = static_cast<unsigned>(svcntw());
    for (unsigned i = 0; i < n; i += step) {
        svbool_t pg = svwhilelt_b32_u32(i, n);
        svuint32_t idx = svld1uh_u32(pg, src + i);
        // byte elements need no index scaling, so this is the offset form.
        svuint32_t v = svld1ub_gather_u32offset_u32(pg, lut, idx);
        svst1b_u32(pg, dst + i, v);
    }
}
