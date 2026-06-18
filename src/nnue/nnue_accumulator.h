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

// Class for difference calculation of NNUE evaluation function

#ifndef NNUE_ACCUMULATOR_H_INCLUDED
#define NNUE_ACCUMULATOR_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstring>
#include <utility>

#include "../types.h"
#include "../misc.h"
#include "nnue_architecture.h"
#include "nnue_common.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE {

struct alignas(CacheLineSize) Accumulator;

class FeatureTransformer;

// Class that holds the result of affine transformation of input features,
// combined HalfKA + Threats
struct alignas(CacheLineSize) Accumulator {
    std::array<std::array<i16, L1>, COLOR_NB>          accumulation;
    std::array<std::array<i32, PSQTBuckets>, COLOR_NB> psqtAccumulation;
    std::array<bool, COLOR_NB>                         computed = {};
};


// AccumulatorCaches struct provides per-thread accumulator caches, where each
// cache contains multiple entries for each of the possible king squares.
// When the accumulator needs to be refreshed, the cached entry is used to more
// efficiently update the accumulator, instead of rebuilding it from scratch.
// This idea, was first described by Luecx (author of Koivisto) and
// is commonly referred to as "Finny Tables".
struct AccumulatorCaches {
    template<typename Network>
    AccumulatorCaches(const Network& network) {
        clear(network);
    }

    struct alignas(CacheLineSize) Entry {
        std::array<BiasType, L1>                accumulation;
        std::array<PSQTWeightType, PSQTBuckets> psqtAccumulation;
        alignas(64) std::array<Piece, SQUARE_NB>            pieces;

        // To initialize a refresh entry, we set all its bitboards empty,
        // so we put the biases in the accumulation, without any weights on top
        void clear(const std::array<BiasType, L1>& biases) {
            accumulation = biases;
            std::memset(reinterpret_cast<std::byte*>(this) + offsetof(Entry, psqtAccumulation), 0,
                        sizeof(Entry) - offsetof(Entry, psqtAccumulation));
        }

        std::pair<Bitboard, Bitboard> get_added_removed(__m512i n) {
            __m512i e = _mm512_load_si512(pieces.data());

            __mmask64 changed = _mm512_cmpneq_epi8_mask(e, n);
            __mmask64 added = _mm512_mask_test_epi8_mask(changed, n, n);
            __mmask64 removed = _mm512_mask_test_epi8_mask(changed, e, e);

            return { added, removed };
        }
    };

    struct EntryCluster {
        static constexpr usize Associativity = 4;

        std::array<Entry, Associativity> entries;

        struct ClusterQueryResult {
            Entry& overwrite;
            Entry& nearest;

            Bitboard addedBB;
            Bitboard removedBB;
        };

        ClusterQueryResult query(const std::array<Piece, 64>& board) {
            usize replace, best;
            alignas(64) Bitboard added[Associativity], removed[Associativity];

            __m512i n = _mm512_loadu_si512(board.data());

            for (usize i = 0; i < Associativity; i++) {
                auto [a, r] = entries[i].get_added_removed(n);
                added[i] = a;
                removed[i] = r;
            }

            if constexpr (Associativity == 8) {
                __m512i score = _mm512_add_epi64(
                    _mm512_popcnt_epi64(_mm512_loadu_si512(added)),
                    _mm512_popcnt_epi64(_mm512_loadu_si512(removed))
                );

                __m128i narrowed = _mm512_cvtepi64_epi16(score);
                auto min_index = [] (__m128i v) {
                    return _mm_extract_epi16(_mm_minpos_epu16(v), 1);
                };

                replace = min_index(_mm_sub_epi16(_mm_setzero_si128(), narrowed));
                best = min_index(narrowed);
            } else {
                int scores[Associativity];
                for (usize i = 0; i < Associativity; i++) {
                    scores[i] = popcount(added[i]) + popcount(removed[i]);
                }
                best = 0;
                replace = 0;
                for (usize i = 0; i < Associativity; i++) {
                    if (scores[i] > scores[replace]) replace = i;
                    if (scores[i] < scores[best]) best = i;
                }
            }

            return ClusterQueryResult{
                entries[replace], entries[best], added[best], removed[best]
            };
        }

        template <typename T>
        void clear(const T& v) {
            for (auto & e : entries) e.clear(v);
        }
    };

    template<typename Network>
    void clear(const Network& network) {
        for (auto& entries1D : entries)
            for (auto& entry : entries1D)
                entry.clear(network.featureTransformer.biases);
    }

    std::array<EntryCluster, COLOR_NB>& operator[](Square sq) { return entries[sq]; }

    std::array<std::array<EntryCluster, COLOR_NB>, SQUARE_NB> entries;
};


struct AccumulatorState: public Accumulator {
    DirtyPiece   dirtyPiece;
    DirtyThreats dirtyThreats;
};

class AccumulatorStack {
   public:
    static constexpr usize MaxSize = MAX_PLY + 1;

    [[nodiscard]] const AccumulatorState& latest() const noexcept;

    void                                  reset() noexcept;
    std::pair<DirtyPiece&, DirtyThreats&> push() noexcept;
    void                                  pop() noexcept;

    void evaluate(const Position&           pos,
                  const FeatureTransformer& featureTransformer,
                  // Silence spurious warning on GCC 10
                  [[maybe_unused]] AccumulatorCaches& cache) noexcept;

   private:
    [[nodiscard]] AccumulatorState& mut_latest() noexcept;

    void evaluate_side(Color                     perspective,
                       const Position&           pos,
                       const FeatureTransformer& featureTransformer,
                       // Silence spurious warning on GCC 10
                       [[maybe_unused]] AccumulatorCaches& cache) noexcept;

    [[nodiscard]] usize find_last_usable_accumulator(Color perspective) const noexcept;

    void forward_update_incremental(Color                     perspective,
                                    const Position&           pos,
                                    const FeatureTransformer& featureTransformer,
                                    const usize               begin) noexcept;

    void backward_update_incremental(Color                     perspective,
                                     const Position&           pos,
                                     const FeatureTransformer& featureTransformer,
                                     const usize               end) noexcept;

    std::array<AccumulatorState, MaxSize> accumulators;
    usize                                 size = 1;
};

}  // namespace Stockfish::Eval::NNUE

#endif  // NNUE_ACCUMULATOR_H_INCLUDED
