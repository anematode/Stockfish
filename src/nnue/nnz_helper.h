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
#include "simd.h"

namespace Stockfish::Eval::NNUE {
    template <size_t Dimensions>
    struct NNZInfo {

#if defined(USE_VNNI) && defined(USE_AVX512)
        unsigned count = 0;
        uint16_t nnz[Dimensions / 4];

#ifdef USE_AVX512ICL
        alignas(64) static constexpr auto Indices = [] () {
             std::array<std::array<uint16_t, 32>, 2> indices{};
             for (int i = 0; i < 2; ++i) {
                 indices[i] = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23, 8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
                 for (uint16_t& m : indices[i])
                      m += i * Dimensions / 8;
             }
            return indices;
        } ();
#else
        alignas(64) static constexpr auto Indices = [] () {
            std::array<std::array<uint32_t, 16>, 2> indices{};
            for (int i = 0; i < 2; ++i) {
                indices[i] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
                for (uint32_t& m : indices[i])
                    m += i * Dimensions / 8;
            }
            return indices;
        } ();
#endif

        struct NNZCursor {
            NNZInfo& info;
            __m512i indices;
            unsigned count;

            NNZCursor(NNZInfo& info_, bool half, unsigned count_) : info(info_), count(count_) {
                indices = _mm512_load_si512(&Indices[half]);
            }

            void record2(SIMD::vec_t neurons1, SIMD::vec_t neurons2) {
#if defined(USE_AVX512ICL)
                const __m512i       increment    = _mm512_set1_epi16(32);

                // Get a bitmask and gather non zero indices
                const __m512i   inputV01 = _mm512_packs_epi32(neurons1, neurons2);
                const __mmask32 nnzMask  = _mm512_test_epi16_mask(inputV01, inputV01);

                // Avoid _mm512_mask_compressstoreu_epi16() as it's 256 uOps on Zen4
                __m512i nnz = _mm512_maskz_compress_epi16(nnzMask, indices);
                _mm512_storeu_si512(info.nnz + count, nnz);

                count += popcount(nnzMask);
                indices = _mm512_add_epi16(indices, increment);
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

            void finalize() const {
                info.count = count;
            }
        };

        NNZCursor make_cursor(bool perspective) {
            return { *this, perspective, count };
        }
#else
        uint8_t bitset[(Dimensions + 31) / 32];

        struct NNZCursor {
            uint8_t* out;

            NNZCursor(NNZInfo& info, bool perspective) {
                out = info.bitset + perspective * Dimensions / 64;
            }

            void record2(SIMD::vec_t neurons1, SIMD::vec_t neurons2) {
                using namespace SIMD;

#ifdef USE_NEON
                alignas(16) static constexpr uint16_t Mask8[8] = {1, 16, 2, 32, 4, 64, 8, 128};

                const uint32x4_t t1 = vtstq_u32((uint32x4_t) neurons1, (uint32x4_t) neurons1);
                const uint32x4_t t2 = vtstq_u32((uint32x4_t) neurons2, (uint32x4_t) neurons2);

                const uint16x8_t packed = vtrn1q_u16((uint16x8_t) t1, (uint16x8_t) t2);
                const uint16x8_t bits   = vandq_u16(packed, vld1q_u16(Mask8));

                *out++ = vaddvq_u16(bits);
#else
                auto m1 = vec_nnz(neurons1);
                auto m2 = vec_nnz(neurons2);

                if (sizeof(neurons1) == 16) {
                    *out++ = m1 + (m2 << 4);
                } else {
                    size_t bytes = sizeof(neurons1) / 32;
                    memcpy(out, &m1, bytes);
                    out += bytes;
                    memcpy(out, &m2, bytes);
                    out += bytes;
                }
#endif
            }

            void finalize() { }
        };

        NNZCursor make_cursor(bool perspective) {
            return { *this, perspective };
        }
#endif
    };
} // namespace Stockfish

#endif