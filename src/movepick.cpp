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

template <int... Is>
struct SortingNetwork {
	static void sort(int64_t *begin) { }
};

template <>
struct SortingNetwork <> {
	static void sort(int64_t *begin) { }
};

typedef int64_t __attribute__((may_alias)) i64;
	
template <int A, int B, int... Is>
struct SortingNetwork <A, B, Is...> {
	static_assert(sizeof...(Is) % 2 == 0);
	static_assert(A != B);

	static __attribute__((always_inline)) void sort_impl(i64 *begin) {
		i64 a = begin[A], b = begin[B], c;
		asm (
			"cmpq %[b], %[a]\n"
			"mov %[b], %[c]\n"
			"cmovl %[a], %[b]\n"
			"cmovl %[c], %[a]\n"
		: [a]"+r"(a), [b]"+r"(b), [c]"=r"(c)
		);
		begin[A] = a;
		begin[B] = b;

		SortingNetwork<Is...>::sort(begin);
	}

	static void sort(i64 *begin) {
		sort_impl(begin);
	}
};

constexpr int NUM_HANDLERS = 15;
static void (*handlers[NUM_HANDLERS])(i64*) = {
	SortingNetwork<>::sort,
	SortingNetwork<>::sort,
	SortingNetwork<0, 1>::sort,
	SortingNetwork<0, 2, 0, 1, 1, 2>::sort,
	SortingNetwork<0, 2, 1, 3, 0, 1, 2, 3, 1, 2>::sort,
	SortingNetwork<0, 3, 1, 4, 0, 2, 1, 3, 0, 1, 2, 4, 1, 2, 3, 4, 2, 3>::sort,
	SortingNetwork<0, 5, 1, 3, 2, 4, 1, 2, 3, 4, 0, 3, 2, 5, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4>::sort,
	SortingNetwork<0, 6, 2, 3, 4, 5, 0, 2, 1, 4, 3, 6, 0, 1, 2, 5, 3, 4, 1, 2, 4, 6, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6>::sort,
	SortingNetwork<0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7, 0, 1, 2, 3, 4, 5, 6, 7, 2, 4, 3, 5, 1, 4, 3, 6, 1, 2, 3, 4, 5, 6>::sort,
	SortingNetwork<0, 3, 1, 7, 2, 5, 4, 8, 0, 7, 2, 4, 3, 8, 5, 6, 0, 2, 1, 3, 4, 5, 7, 8, 1, 4, 3, 6, 5, 7, 0, 1, 2, 4, 3, 5, 6, 8, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6>::sort,
	SortingNetwork<0,8,1,9,2,7,3,5,4,6,0,2,1,4,5,8,7,9,0,3,2,4,5,7,6,9,0,1,3,6,8,9,1,5,2,3,4,8,6,7,1,2,3,5,4,6,7,8,2,3,4,5,6,7,3,4,5,6>::sort,
	SortingNetwork<0,9,1,6,2,4,3,7,5,8,0,1,3,5,4,10,6,9,7,8,1,3,2,5,4,7,8,10,0,4,1,2,3,7,5,9,6,8,0,1,2,6,4,5,7,8,9,10,2,4,3,6,5,7,8,9,1,2,3,4,5,6,7,8,2,3,4,5,6,7>::sort,
	SortingNetwork<0,8,1,7,2,6,3,11,4,10,5,9,0,1,2,5,3,4,6,9,7,8,10,11,0,2,1,6,5,10,9,11,0,3,1,2,4,6,5,7,8,11,9,10,1,4,3,5,6,8,7,10,1,3,2,5,6,9,8,10,2,3,4,5,6,7,8,9,4,6,5,7,3,4,5,6,7,8>::sort,
	SortingNetwork<0,12,1,10,2,9,3,7,5,11,6,8,1,6,2,3,4,11,7,9,8,10,0,4,1,2,3,6,7,8,9,10,11,12,4,6,5,9,8,11,10,12,0,5,3,8,4,7,6,11,9,10,0,1,2,5,6,9,7,8,10,11,1,3,2,4,5,6,9,10,1,2,3,4,5,7,6,8,2,3,4,5,6,7,8,9,3,4,5,6>::sort,
	SortingNetwork<0,1,2,3,4,5,6,7,8,9,10,11,12,13,0,2,1,3,4,8,5,9,10,12,11,13,0,4,1,2,3,7,5,8,6,10,9,13,11,12,0,6,1,5,3,9,4,10,7,13,8,12,2,10,3,11,4,6,7,9,1,3,2,8,5,11,6,7,10,12,1,4,2,6,3,5,7,11,8,10,9,12,2,4,3,6,5,8,7,10,9,11,3,4,5,6,7,8,9,10,6,7>::sort
};

// Sort moves in descending order up to and including a given limit.
// The order of moves smaller than the limit is left unspecified.
void insertion_sort(ExtMove* begin, ExtMove* end) {
	if (end - begin < NUM_HANDLERS) {
		handlers[end - begin](reinterpret_cast<i64*>(begin));
		return;
	}

	handlers[NUM_HANDLERS - 1](reinterpret_cast<i64*>(begin));

    for (ExtMove *sortedEnd = begin + NUM_HANDLERS - 2, *p = sortedEnd + 1; p < end; ++p) {
		ExtMove tmp = *p, *q;
		*p          = *++sortedEnd;
		for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
			*q = *(q - 1);
		*q = tmp;
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
ExtMove* MovePicker::score(MoveList<Type>& ml, int partition_quiets) {

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
    ExtMove* end = cur + ml.size() - 1;
    for (auto move : ml)
    {
		ExtMove m;
		m = move;

        const Square    from          = m.from_sq();
        const Square    to            = m.to_sq();
        const Piece     pc            = pos.moved_piece(m);
        const PieceType pt            = type_of(pc);
        const Piece     capturedPiece = pos.piece_on(to);

        if constexpr (Type == CAPTURES)
            m.value = (*captureHistory)[pc][to][type_of(capturedPiece)]
                    + 7 * int(PieceValue[capturedPiece]);

        else if constexpr (Type == QUIETS)
        {
            // histories
            m.value = 2 * (*mainHistory)[us][m.raw()];
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
            int v = threatByLesser[pt] & to ? -19 : 20 * bool(threatByLesser[pt] & from);
            m.value += PieceValue[pt] * v;


            if (ply < LOW_PLY_HISTORY_SIZE)
                m.value += 8 * (*lowPlyHistory)[ply][m.raw()] / (1 + ply);

			int should_sort = m.value >= partition_quiets;
			auto *write = should_sort ? it : end;
			it += should_sort;
			end -= !should_sort;
			*write = m;
			continue;
        }

        else  // Type == EVASIONS
        {
            if (pos.capture_stage(m))
                m.value = PieceValue[capturedPiece] + (1 << 28);
            else
            {
                m.value = (*mainHistory)[us][m.raw()] + (*continuationHistory[0])[pc][to];
                if (ply < LOW_PLY_HISTORY_SIZE)
                    m.value += (*lowPlyHistory)[ply][m.raw()];
            }
        }

		*it++ = m;
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

            endCur = endGenerated = cur + ml.size();	
			auto partitioned_end = score<QUIETS>(ml, -3560 * depth);

			insertion_sort(cur, partitioned_end);
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
