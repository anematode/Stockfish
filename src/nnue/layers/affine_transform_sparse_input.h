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

// Definition of layer AffineTransformSparseInput of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "../../bitboard.h"
#include "../../memory.h"
#include "../simd.h"
#include "../nnue_common.h"

/*
  This file contains the definition for a fully connected layer (aka affine transform) with block sparse input.
*/

namespace Stockfish::Eval::NNUE::Layers {

template <int InDims, int SplitBy, typename T>
void for_each_nnz(const uint8_t* nnz32, T&& callback) {
    static_assert(InDims % 32 == 0);

    size_t nnzBytesLeft = size_t(InDims / 32);

    if constexpr (SplitBy == 1) {
        ptrdiff_t base = 0;

        while (nnzBytesLeft) {
            uint64_t set;
            size_t consumedBytes = std::min(nnzBytesLeft, sizeof(set));
            memcpy(&set, nnz32, consumedBytes);

            while (set) {
                ptrdiff_t i = pop_lsb(set);
                callback(i, 0);
            }

            base += consumedBytes * 8;
            nnzBytesLeft -= consumedBytes;
            nnz32 += consumedBytes;
        }
    } else {
        // If there are more than 1024 neurons then we need to address 32-bit blocks with an
        // 8-bit index
        constexpr bool NeedsWideIndex = InDims > 1024;
        using NNZIndex = std::conditional_t<NeedsWideIndex, uint16_t, uint8_t>;
        using Mask = std::conditional_t<NeedsWideIndex, uint32_t, uint64_t>;

        alignas(64) NNZIndex indices[ceil_to_multiple(InDims / 4, 64)];
        NNZIndex* write = indices;

#if defined(USE_AVX512ICL)
        __m512i base = NeedsWideIndex ?
            _mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
              17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0) :
        _mm512_set_epi8(
          63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41,
          40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
          17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        const __m512i increment = NeedsWideIndex ? _mm512_set1_epi16(32) : _mm512_set1_epi8(64);

        while (nnzBytesLeft) {
            Mask set;
            size_t consumedBytes = std::min(nnzBytesLeft, sizeof(set));
            memcpy(&set, nnz32, consumedBytes);

            __m512i compressed = NeedsWideIndex ?
                _mm512_maskz_compress_epi16(set, base)
                : _mm512_maskz_compress_epi8(set, base);
            _mm512_storeu_si512(write, compressed);
            write += popcount(set);

            base = _mm512_add_epi16(base, increment);
            nnzBytesLeft -= consumedBytes;
            nnz32 += consumedBytes;
        }
#endif

        NNZIndex* idx = indices;
        while (idx + (SplitBy - 1) < write)
            for (int i = 0; i < SplitBy; ++i)
                callback(*idx++, i);

        while (idx < write) {
            callback(*idx++, 0);
        }
    }
}

// Sparse input implementation
template<IndexType InDims, IndexType OutDims>
class AffineTransformSparseInput {
   public:
    // Input/output type
    using InputType  = std::uint8_t;
    using OutputType = std::int32_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static_assert(OutputDimensions % 16 == 0,
                  "Only implemented for OutputDimensions divisible by 16.");

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

#if (USE_SSSE3 | (USE_NEON >= 8))
    static constexpr IndexType ChunkSize = 4;
#else
    static constexpr IndexType ChunkSize = 1;
#endif

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
        std::uint32_t hashValue = 0xCC03DAE4u;
        hashValue += OutputDimensions;
        hashValue ^= prevHash >> 1;
        hashValue ^= prevHash << 31;
        return hashValue;
    }

    static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / ChunkSize) % (PaddedInputDimensions / ChunkSize) * OutputDimensions * ChunkSize
             + i / PaddedInputDimensions * ChunkSize + i % ChunkSize;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#if (USE_SSSE3 | (USE_NEON >= 8))
        return get_weight_index_scrambled(i);
#else
        return i;
#endif
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_little_endian<BiasType>(stream, biases, OutputDimensions);
        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            weights[get_weight_index(i)] = read_little_endian<WeightType>(stream);

        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        write_little_endian<BiasType>(stream, biases, OutputDimensions);

        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            write_little_endian<WeightType>(stream, weights[get_weight_index(i)]);

        return !stream.fail();
    }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_hash_value(0));
        return h;
    }

    // Forward propagation
    void propagate(const InputType* input, OutputType* output, const uint8_t* nnz32) const {

#if (USE_SSSE3 | (USE_NEON >= 8))
    #if defined(USE_AVX512)
        using invec_t  = __m512i;
        using outvec_t = __m512i;
        #define vec_add_32 _mm512_add_epi32
        #define vec_set_32 _mm512_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m512_add_dpbusd_epi32
    #elif defined(USE_AVX2)
        using invec_t  = __m256i;
        using outvec_t = __m256i;
        #define vec_add_32 _mm256_add_epi32
        #define vec_set_32 _mm256_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m256_add_dpbusd_epi32
    #elif defined(USE_SSSE3)
        using invec_t  = __m128i;
        using outvec_t = __m128i;
        #define vec_set_32 _mm_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m128_add_dpbusd_epi32
    #elif defined(USE_NEON_DOTPROD)
        using invec_t  = int8x16_t;
        using outvec_t = int32x4_t;
        #define vec_set_32(a) vreinterpretq_s8_u32(vdupq_n_u32(a))
        #define vec_add_dpbusd_32 SIMD::dotprod_m128_add_dpbusd_epi32
    #elif defined(USE_NEON)
        using invec_t  = int8x16_t;
        using outvec_t = int32x4_t;
        #define vec_set_32(a) vreinterpretq_s8_u32(vdupq_n_u32(a))
        #define vec_add_32 vaddq_s32
        #define vec_add_dpbusd_32 SIMD::neon_m128_add_dpbusd_epi32
    #endif

        constexpr IndexType OutputSimdWidth = sizeof(outvec_t) / sizeof(OutputType);
        constexpr IndexType NumAccums = OutputDimensions / OutputSimdWidth;

        // If we're using high-latency dot product instructions, split the accumulators
        // to create 3 separate dependency chains and merge at the end
    #if defined(USE_VNNI) && defined(USE_AVX512)
        constexpr IndexType SplitBy = InDims >= 1024 ? 4 : 1;
    #elif defined(USE_NEON_DOTPROD) || defined(USE_VNNI)
        constexpr IndexType SplitBy = InDims >= 512 ? 2 : 1;
    #else
        constexpr IndexType SplitBy = 1;
    #endif

        const outvec_t* biasvec = reinterpret_cast<const outvec_t*>(biases);
        outvec_t        acc[SplitBy][NumAccums] = {};
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[0][k] = biasvec[k];

        // convince GCC to not do weird pointer arithmetic in the following loop
        const std::int8_t* weights_cp = weights;
        for_each_nnz<InDims, SplitBy>(nnz32, [&acc, weights_cp, input] (ptrdiff_t i, int splitIndex) {
            assert(splitIndex < SplitBy);

            auto i32 = load_as<std::int32_t>(input + i * sizeof(std::int32_t));
            assert(i32 != 0);

            const invec_t in = vec_set_32(i32);
            const auto    col =
              reinterpret_cast<const invec_t*>(&weights_cp[i * OutputDimensions * ChunkSize]);
            for (IndexType k = 0; k < NumAccums; ++k)
                vec_add_dpbusd_32(acc[splitIndex][k], in, col[k]);
        });

        // Fold multiple accumulators
        for (IndexType si = 1; si < SplitBy; ++si)
            for (IndexType k = 0; k < NumAccums; ++k)
                acc[0][k] = vec_add_32(acc[0][k], acc[si][k]);

        outvec_t* outptr = reinterpret_cast<outvec_t*>(output);
        for (IndexType k = 0; k < NumAccums; ++k)
            outptr[k] = acc[0][k];

    #undef vec_set_32
    #undef vec_add_dpbusd_32
    #ifdef vec_add_32
        #undef vec_add_32
    #endif
#else
        // Use dense implementation for the other architectures.
        affine_transform_non_ssse3<InputDimensions, PaddedInputDimensions, OutputDimensions>(
          output, weights, biases, input);
#endif
    }

   private:
    using BiasType   = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
