/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef NNZ_HELPER_H_INCLUDED
#define NNZ_HELPER_H_INCLUDED

#include <utility>

#include "nnue_common.h"
#include "nnue_misc.h"
#include "simd.h"
#include "simd.h"

namespace Stockfish {
    template <IndexType Dimensions>
    struct NNZInfo {

#if defined(USE_VNNI) && defined(USE_AVX512)
        uint16_t nnz[Dimensions / 4];
        size_t count = 0;

        struct NNZCursor {
            NNZIndex* out;
            size_t& count;

            __m512i indices;

            NNZCursor(NNZInfo& info, IndexType offset) {
                out = info.nnz;
                count = info.count;

                indices = _mm512_set_epi16(  // Same permute order as _mm512_packus_epi32()
                  31, 30, 29, 28, 15, 14, 13, 12, 27, 26, 25, 24, 11, 10, 9, 8, 23, 22, 21, 20, 7, 6, 5, 4,
                  19, 18, 17, 16, 3, 2, 1, 0);
                indices = _mm512_add_epi16(indices, _mm512_set1_epi16(offset / 4));
            }

            void record2(vec_t neurons1, vec_t neurons2) {
#if defined(USE_AVX512ICL)
                const __m512i       increment    = _mm512_set1_epi16(64);

                // Get a bitmask and gather non zero indices
                const __m512i   inputV01 = _mm512_packs_epi32(neurons1, neurons2);
                const __mmask32 nnzMask  = _mm512_test_epi16_mask(inputV01, inputV01);

                // Avoid _mm512_mask_compressstoreu_epi16() as it's 256 uOps on Zen4
                __m512i nnz = _mm512_maskz_compress_epi16(nnzMask, base);
                _mm512_storeu_si512(out + count, nnz);

                count += popcount(nnzMask);
                base = _mm512_add_epi16(base, increment);
#else
                for (auto neurons : { neurons1, neurons2 }) {
                    const __m512i       increment  = _mm512_set1_epi32(16);
                    // Get a bitmask and gather non zero indices
                    const __mmask16 nnzMask = _mm512_test_epi32_mask(neurons, neurons);
                    const __m512i   nnzV    = _mm512_maskz_compress_epi32(nnzMask, base);
                    _mm512_mask_cvtepi32_storeu_epi16(out + count, 0xFFFF, nnzV);
                    count += popcount(nnzMask);
                    base = _mm512_add_epi32(base, increment);
                }
#endif
            }
        };
#else
        uint8_t bitset[(Dimensions + 31) / 32];

        struct NNZCursor {
            uint8_t* out;

            NNZCursor(NNZInfo& info, IndexType offset) {
                out = info.bitset + offset / 32;
            }

            void record2(vec_t neurons1, vec_t neurons2) {
                auto m1 = vec_nnz(neurons1);
                auto m2 = vec_nnz(neurons2);

                if (sizeof(vec_t) == 16) {
                    *out++ = m1 + (m2 << 4);
                } else {
                    memcpy(out, &m1, sizeof(m1));
                    out += sizeof(m1);
                    memcpy(out, &m1, sizeof(m2));
                    out += sizeof(m2);
                }
            }
        };
#endif

        NNZCursor make_cursor(IndexType offset) const {
            return { *this, offset };
        }
    };
} // namespace Stockfish

#endif