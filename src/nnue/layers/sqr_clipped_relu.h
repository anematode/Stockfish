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

// Definition of layer ClippedReLU of NNUE evaluation function

#ifndef NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Layers {

// Squared Clipped ReLU
template<IndexType InDims, int WeightScaleBitsLocal = WeightScaleBits>
class SqrClippedReLU {
   public:
    // Input/output type
    using InputType  = std::int32_t;
    using OutputType = std::uint8_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = InputDimensions;
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, 32);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
        std::uint32_t hashValue = 0x538D24C7u;
        hashValue += prevHash;
        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream&) { return true; }

    // Write network parameters
    bool write_parameters(std::ostream&) const { return true; }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, get_hash_value(0));
        return h;
    }

    // Forward propagation
    void propagate(const InputType* input, OutputType* output) const {

#if defined(USE_AVX512)
        constexpr IndexType NumChunks = InputDimensions / 64;
        const auto in  = reinterpret_cast<const __m512i*>(input);
        const auto out = reinterpret_cast<__m512i*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            __m512i words0 =
              _mm512_packs_epi32(_mm512_load_si512(&in[i * 4 + 0]), _mm512_load_si512(&in[i * 4 + 1]));
            __m512i words1 =
              _mm512_packs_epi32(_mm512_load_si512(&in[i * 4 + 2]), _mm512_load_si512(&in[i * 4 + 3]));

            if constexpr (WeightScaleBitsLocal == 6)
            {
                words0 = _mm512_srli_epi16(_mm512_mulhi_epi16(words0, words0), 3);
                words1 = _mm512_srli_epi16(_mm512_mulhi_epi16(words1, words1), 3);
            }
            else if constexpr (WeightScaleBitsLocal == 7)
            {
                words0 = _mm512_srli_epi16(_mm512_mulhi_epi16(words0, words0), 5);
                words1 = _mm512_srli_epi16(_mm512_mulhi_epi16(words1, words1), 5);
            }
            else
            {
                static_assert(WeightScaleBitsLocal == 6 || WeightScaleBitsLocal == 7, 
                              "Unsupported WeightScaleBitsLocal for SIMD squared propagate");
            }

            _mm512_store_si512(&out[i], _mm512_packs_epi16(words0, words1));
        }
        constexpr IndexType Start = NumChunks * 64;

#elif defined(USE_AVX2)
        constexpr IndexType NumChunks = InputDimensions / 32;
        const auto in  = reinterpret_cast<const __m256i*>(input);
        const auto out = reinterpret_cast<__m256i*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            __m256i words0 =
              _mm256_packs_epi32(_mm256_load_si256(&in[i * 4 + 0]), _mm256_load_si256(&in[i * 4 + 1]));
            __m256i words1 =
              _mm256_packs_epi32(_mm256_load_si256(&in[i * 4 + 2]), _mm256_load_si256(&in[i * 4 + 3]));

            if constexpr (WeightScaleBitsLocal == 6)
            {
                words0 = _mm256_srli_epi16(_mm256_mulhi_epi16(words0, words0), 3);
                words1 = _mm256_srli_epi16(_mm256_mulhi_epi16(words1, words1), 3);
            }
            else if constexpr (WeightScaleBitsLocal == 7)
            {
                words0 = _mm256_srli_epi16(_mm256_mulhi_epi16(words0, words0), 5);
                words1 = _mm256_srli_epi16(_mm256_mulhi_epi16(words1, words1), 5);
            }
            else
            {
                static_assert(WeightScaleBitsLocal == 6 || WeightScaleBitsLocal == 7, 
                              "Unsupported WeightScaleBitsLocal for SIMD squared propagate");
            }

            _mm256_store_si256(&out[i], _mm256_packs_epi16(words0, words1));
        }
        constexpr IndexType Start = NumChunks * 32;

#elif defined(USE_SSE2)
        constexpr IndexType NumChunks = InputDimensions / 16;
        const auto in  = reinterpret_cast<const __m128i*>(input);
        const auto out = reinterpret_cast<__m128i*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            __m128i words0 =
              _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 0]), _mm_load_si128(&in[i * 4 + 1]));
            __m128i words1 =
              _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 2]), _mm_load_si128(&in[i * 4 + 3]));

            if constexpr (WeightScaleBitsLocal == 6)
            {
                words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 3);
                words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 3);
            }
            else if constexpr (WeightScaleBitsLocal == 7)
            {
                words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 5);
                words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 5);
            }
            else
            {
                static_assert(WeightScaleBitsLocal == 6 || WeightScaleBitsLocal == 7, 
                              "Unsupported WeightScaleBitsLocal for SIMD squared propagate");
            }

            _mm_store_si128(&out[i], _mm_packs_epi16(words0, words1));
        }
        constexpr IndexType Start = NumChunks * 16;

#elif defined(USE_NEON)
        constexpr IndexType NumChunks = InputDimensions / 16;
        const auto in  = reinterpret_cast<const int32x4_t*>(input);
        const auto out = reinterpret_cast<int8x16_t*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            // vqmovn_s32 narrows 32-bit to 16-bit with signed saturation
            int16x8_t words0 = vcombine_s16(vqmovn_s32(in[i * 4 + 0]), vqmovn_s32(in[i * 4 + 1]));
            int16x8_t words1 = vcombine_s16(vqmovn_s32(in[i * 4 + 2]), vqmovn_s32(in[i * 4 + 3]));

            if constexpr (WeightScaleBitsLocal == 6)
            {
                // Net shift: 19. vqdmulhq_s16 removes 15. Remaining: 4.
                words0 = vshrq_n_s16(vqdmulhq_s16(words0, words0), 4);
                words1 = vshrq_n_s16(vqdmulhq_s16(words1, words1), 4);
            }
            else if constexpr (WeightScaleBitsLocal == 7)
            {
                // Net shift: 21. vqdmulhq_s16 removes 15. Remaining: 6.
                words0 = vshrq_n_s16(vqdmulhq_s16(words0, words0), 6);
                words1 = vshrq_n_s16(vqdmulhq_s16(words1, words1), 6);
            }
            else
            {
                static_assert(WeightScaleBitsLocal == 6 || WeightScaleBitsLocal == 7, 
                              "Unsupported WeightScaleBitsLocal for SIMD squared propagate");
            }

            // vqmovn_s16 narrows 16-bit to 8-bit with signed saturation
            out[i] = vcombine_s8(vqmovn_s16(words0), vqmovn_s16(words1));
        }
        constexpr IndexType Start = NumChunks * 16;

#else
        constexpr IndexType Start = 0;
#endif

        for (IndexType i = Start; i < InputDimensions; ++i)
        {
            output[i] = static_cast<OutputType>(
              std::min(127ll, ((long long) (input[i]) * input[i]) >> (2 * WeightScaleBitsLocal + 7)));
        }
    }
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
