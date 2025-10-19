/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

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

#include "movepick.h"

#include <cassert>
#include <limits>
#include <utility>

#include "bitboard.h"
#include "misc.h"
#include "position.h"

namespace Stockfish {

namespace {

enum Stages {
    // generate main search moves
    MAIN_TT,
    CAPTURE_INIT,
    GOOD_CAPTURE,
    QUIET_INIT,
    GOOD_QUIET,
    BAD_CAPTURE,
    BAD_QUIET,

    // generate evasion moves
    EVASION_TT,
    EVASION_INIT,
    EVASION,

    // generate probcut moves
    PROBCUT_TT,
    PROBCUT_INIT,
    PROBCUT,

    // generate qsearch moves
    QSEARCH_TT,
    QCAPTURE_INIT,
    QCAPTURE
};

void insertion_sort(ExtMove* begin, ExtMove* end) {
    for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p) {
        ExtMove tmp = *p, *q;
        *p          = *++sortedEnd;
        for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
            *q = *(q - 1);
        *q = tmp;
    }
}

struct PartialInsertionSorter {
    // We store the limits as (limit << 8) | <original index>, and mask off the index before doing the compare, so that
    // ties are broken exactly as in the normal implementation
    __m512i sorted_limits;
    int count = 1;

    struct PackedLimit { uint8_t index; uint8_t _limit[3]; };

    explicit PartialInsertionSorter(ExtMove *begin) {
        sorted_limits = _mm512_castsi128_si512(_mm_cvtsi32_si128((begin->value & 0xffffff) << 8));
    }

    static __m512i mask_off_index(const __m512i vec) {
        return _mm512_maskz_mov_epi8(0xeeeeeeeeeeeeeeeeULL, vec);
    }

    bool insert(ExtMove *move, int move_index) {
        if (count > 15) {
            return true;
        }
        assert(move_index < 256);
        // Only the first `count` elements should be considered for sorting.
        const __mmask16 relevant = (1 << count) - 1;
        const __m512i packed_limit = _mm512_set1_epi32((move->value & 0xffffff) << 8 | move_index);
        // Each 1 bit corresponds to an element that should be to the left of the element to be inserted. By
        // construction, this will be one less than a power of two.
        const __mmask16 limits_ge = _mm512_mask_cmpge_epi32_mask(relevant, mask_off_index(sorted_limits), mask_off_index(packed_limit));
        // By adding 1, we get a single bit that is the correct insertion point of the new element.
        // Then, the bitwise negation gives us those bits where the old elements should be expanded.
        sorted_limits = _mm512_mask_expand_epi32(packed_limit, ~(limits_ge + 1), sorted_limits);
        count++;
        return false;
    }

    void apply(ExtMove copy[16], ExtMove *begin) {
        alignas(64) PackedLimit ls[16];
        _mm512_store_si512(ls, sorted_limits);
        for (int i = 0; i < count; ++i) {
            begin[ls[i].index] = copy[i];
        }
    }
};

// Sort moves in descending order up to and including a given limit.
// The order of moves smaller than the limit is left unspecified.
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {
    if (begin == end) {
        return;
    }

    ExtMove *sortedEnd = begin;
    ExtMove *p = begin + 1;

    if (begin->value < limit) {

        for (; p < end; ++p) {
            if (p->value >= limit)
            {
                ExtMove tmp = *p, *q;
                *p          = *++sortedEnd;
                for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                    *q = *(q - 1);
                *q = tmp;
            }
        }
        /*
        // sortedEnd and p won't alias, which is required for this to work straightforwardly
        PartialInsertionSorter sorter { begin };

        ExtMove copy[16];
        std::copy_n(begin, std::min(std::ptrdiff_t(16), end - begin), copy);

        for (; p < end; ++p) {
            if (p->value >= limit)
            {
                if (sorter.insert(p, p - begin)) {
                    break;
                }
                *p = *++sortedEnd;
            }
        }
        sorter.apply(copy, begin);*/
    }

    for (; p < end; ++p) {
        if (p->value >= limit)
        {
            ExtMove tmp = *p, *q;
            *p          = *++sortedEnd;
            for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                *q = *(q - 1);
            *q = tmp;
        }
    }
}

}  // namespace


// Constructors of the MovePicker class. As arguments, we pass information
// to decide which class of moves to emit, to help sorting the (presumably)
// good moves first, and how important move ordering is at the current node.

// MovePicker constructor for the main search and for the quiescence search
MovePicker::MovePicker(const Position&              p,
                       Move                         ttm,
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const LowPlyHistory*         lph,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const PawnHistory*           ph,
                       int                          pl) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    pawnHistory(ph),
    ttMove(ttm),
    depth(d),
    ply(pl) {

    if (pos.checkers())
        stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm));

    else
        stage = (depth > 0 ? MAIN_TT : QSEARCH_TT) + !(ttm && pos.pseudo_legal(ttm));
}

// MovePicker constructor for ProbCut: we generate captures with Static Exchange
// Evaluation (SEE) greater than or equal to the given threshold.
MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph) :
    pos(p),
    captureHistory(cph),
    ttMove(ttm),
    threshold(th) {
    assert(!pos.checkers());

    stage = PROBCUT_TT + !(ttm && pos.capture_stage(ttm) && pos.pseudo_legal(ttm));
}

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiets moves are ordered using the history tables.
template<GenType Type>
ExtMove* MovePicker::score(MoveList<Type>& ml) {

    static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

    Color us = pos.side_to_move();

    [[maybe_unused]] Bitboard threatByLesser[KING + 1];
    if constexpr (Type == QUIETS)
    {
        threatByLesser[PAWN]   = 0;
        threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(~us);
        threatByLesser[ROOK] =
          pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatByLesser[KNIGHT];
        threatByLesser[QUEEN] = pos.attacks_by<ROOK>(~us) | threatByLesser[ROOK];
        threatByLesser[KING]  = pos.attacks_by<QUEEN>(~us) | threatByLesser[QUEEN];
    }

    ExtMove* it = cur;
    for (auto move : ml)
    {
        ExtMove& m = *it++;
        m          = move;

        const Square    from          = m.from_sq();
        const Square    to            = m.to_sq();
        const Piece     pc            = pos.moved_piece(m);
        const PieceType pt            = type_of(pc);
        const Piece     capturedPiece = pos.piece_on(to);

        if constexpr (Type == CAPTURES)
            m.value = (*captureHistory)[pc][to][type_of(capturedPiece)]
                    + 7 * int(PieceValue[capturedPiece]) + 1024 * bool(pos.check_squares(pt) & to);

        else if constexpr (Type == QUIETS)
        {
            // histories
            m.value = 2 * (*mainHistory)[us][m.from_to()];
            m.value += 2 * (*pawnHistory)[pawn_history_index(pos)][pc][to];
            m.value += (*continuationHistory[0])[pc][to];
            m.value += (*continuationHistory[1])[pc][to];
            m.value += (*continuationHistory[2])[pc][to];
            m.value += (*continuationHistory[3])[pc][to];
            m.value += (*continuationHistory[5])[pc][to];

            // bonus for checks
            m.value += (bool(pos.check_squares(pt) & to) && pos.see_ge(m, -75)) * 16384;

            // penalty for moving to a square threatened by a lesser piece
            // or bonus for escaping an attack by a lesser piece.
            static constexpr int bonus[KING + 1] = {0, 0, 144, 144, 256, 517, 10000};
            int v = threatByLesser[pt] & to ? -95 : 100 * bool(threatByLesser[pt] & from);
            m.value += bonus[pt] * v;


            if (ply < LOW_PLY_HISTORY_SIZE)
                m.value += 8 * (*lowPlyHistory)[ply][m.from_to()] / (1 + ply);
        }

        else  // Type == EVASIONS
        {
            if (pos.capture_stage(m))
                m.value = PieceValue[capturedPiece] + (1 << 28);
            else
            {
                m.value = (*mainHistory)[us][m.from_to()] + (*continuationHistory[0])[pc][to];
                if (ply < LOW_PLY_HISTORY_SIZE)
                    m.value += (*lowPlyHistory)[ply][m.from_to()];
            }
        }
    }
    return it;
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.
template<typename Pred>
Move MovePicker::select(Pred filter) {

    for (; cur < endCur; ++cur)
        if (*cur != ttMove && filter())
            return *cur++;

    return Move::none();
}

// This is the most important method of the MovePicker class. We emit one
// new pseudo-legal move on every call until there are no more moves left,
// picking the move with the highest score from a list of generated moves.
Move MovePicker::next_move() {

    constexpr int goodQuietThreshold = -14000;
top:
    switch (stage)
    {

    case MAIN_TT :
    case EVASION_TT :
    case QSEARCH_TT :
    case PROBCUT_TT :
        ++stage;
        return ttMove;

    case CAPTURE_INIT :
    case PROBCUT_INIT :
    case QCAPTURE_INIT : {
        MoveList<CAPTURES> ml(pos);

        cur = endBadCaptures = moves;
        endCur = endCaptures = score<CAPTURES>(ml);

        insertion_sort(cur, endCur);
        ++stage;
        goto top;
    }

    case GOOD_CAPTURE :
        if (select([&]() {
                if (pos.see_ge(*cur, -cur->value / 18))
                    return true;
                std::swap(*endBadCaptures++, *cur);
                return false;
            }))
            return *(cur - 1);

        ++stage;
        [[fallthrough]];

    case QUIET_INIT :
        if (!skipQuiets)
        {
            MoveList<QUIETS> ml(pos);

            endCur = endGenerated = score<QUIETS>(ml);

            unsigned _;
            auto start = __rdtscp(&_);
            partial_insertion_sort(cur, endCur, -3560 * depth);
            dbg_mean_of(__rdtsc() - start, 2);
        }

        ++stage;
        [[fallthrough]];

    case GOOD_QUIET :
        if (!skipQuiets && select([&]() { return cur->value > goodQuietThreshold; }))
            return *(cur - 1);

        // Prepare the pointers to loop over the bad captures
        cur    = moves;
        endCur = endBadCaptures;

        ++stage;
        [[fallthrough]];

    case BAD_CAPTURE :
        if (select([]() { return true; }))
            return *(cur - 1);

        // Prepare the pointers to loop over quiets again
        cur    = endCaptures;
        endCur = endGenerated;

        ++stage;
        [[fallthrough]];

    case BAD_QUIET :
        if (!skipQuiets)
            return select([&]() { return cur->value <= goodQuietThreshold; });

        return Move::none();

    case EVASION_INIT : {
        MoveList<EVASIONS> ml(pos);

        cur    = moves;
        endCur = endGenerated = score<EVASIONS>(ml);

        insertion_sort(cur, endCur);
        ++stage;
        [[fallthrough]];
    }

    case EVASION :
    case QCAPTURE :
        return select([]() { return true; });

    case PROBCUT :
        return select([&]() { return pos.see_ge(*cur, threshold); });
    }

    assert(false);
    return Move::none();  // Silence warning
}

void MovePicker::skip_quiet_moves() { skipQuiets = true; }

}  // namespace Stockfish
