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


// Sort moves in descending order up to and including a given limit.
// The order of moves smaller than the limit is left unspecified.
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {

    for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
        if (p->value >= limit)
        {
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
                       const SharedHistories*       sh,
                       int                          pl) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    sharedHistory(sh),
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

#ifdef USE_AVX512ICL

// RankToSection[us][pt][r] tells us that a piece of color us and type pt, present on rank r,
// may move to sections given by the bitset. A section (in [0,3]) is a consecutive pair of rows.
alignas(64) static constexpr auto RankToSection = [] () {
    std::array<std::array<std::array<uint8_t, RANK_NB>, 8 /* >= PIECE_TYPE_NB */>, COLOR_NB> arr{};
    
    for (Color c = WHITE; c < 2; c = Color(c + 1)) {
        for (Rank r = RANK_1; r < 8; r = Rank(r + 1)) {
            for (PieceType pt = PAWN; pt < PIECE_TYPE_NB; pt = PieceType(pt + 1)) {
                uint8_t& result = arr[c][pt][r];
                int section = r / 2;
                result = 1 << section;

                switch (pt) {
                // sliders can go to any section
                case QUEEN: case ROOK: case BISHOP:
                    result = 0b1111;
                    break;
                // Pawns can sometimes cross a section in the direction of a pawn push
                case PAWN:
                    if (r % 2 == (c == WHITE))
                        result |= c == WHITE ? result << 1 : result >> 1;
                    break;
                // Kings can go down a section if on an even row, or else up a section
                case KING:
                    result |= r % 2 == 0 ? result >> 1 : result << 1;
                    break;
                // Knights can go up or down a section
                case KNIGHT:
                    result |= result >> 1 | result << 1;
                    break;
                default:;
                    result = 0;
                }
            }
        }
    }

    return arr;
} ();

unsigned pc_and_index_info(Color us, const Bitboard* bbs, Bitboard our_pieces) {
    __mmask64 rank_nonempty = _mm512_test_epi8_mask( _mm512_loadu_si512(bbs), _mm512_set1_epi64(our_pieces));
    __m512i bits = _mm512_maskz_loadu_epi8(rank_nonempty, &RankToSection[us]);

    // Sequentially test bits, then interleave them
    unsigned mask = 0;
    for (int sect = 0; sect < 4; ++sect) {
        __mmask8 m = _mm512_test_epi64_mask(bits, _mm512_set1_epi8(1 << sect));
        mask += _pdep_u32(m, 0x01111111) << sect;
    }

    return mask;
}


#endif

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiets moves are ordered using the history tables.
template<GenType Type>
ExtMove* MovePicker::score(MoveList<Type>& ml) {

    static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

    Color us = pos.side_to_move();

    [[maybe_unused]] Bitboard threatByLesser[KING + 1];
    #if defined(USE_AVX512)
    alignas(64) int histBuffer[KING][SQUARE_NB];
    #endif // defined

    if constexpr (Type == QUIETS)
    {
        threatByLesser[PAWN]   = 0;
        threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(~us);
        threatByLesser[ROOK] =
          pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatByLesser[KNIGHT];
        threatByLesser[QUEEN] = pos.attacks_by<ROOK>(~us) | threatByLesser[ROOK];
        threatByLesser[KING]  = 0;

        #if defined(USE_AVX512)
        // Each set bit is i + pt * 4
        unsigned mask = pc_and_index_info(us, pos.piece_type_bbs(), pos.pieces(us));
        sf_assume(mask != 0);

        __m512i* hist = (__m512i*)histBuffer;
        const __m256i* conthistBase[5];

        size_t p = (us == BLACK) * PIECE_TYPE_NB * SQUARE_NB * 2 / sizeof(__m256i);
        int idx = 0;
        for (int j: {0,1,2,3,5}) {
            conthistBase[idx++] = (const __m256i*)continuationHistory[j] + p;
        }

        const __m256i* pawn_base = (const __m256i*)&sharedHistory->pawn_entry(pos) + p;

        for (; mask != 0; mask &= mask - 1) {
            unsigned j = lsb(mask);

            __m512i* buff = hist + j - 4;
            __m512i  curHist =  _mm512_cvtepi16_epi32(_mm256_slli_epi16(_mm256_load_si256(pawn_base + j), 1));
            for (auto base : conthistBase)
            {
                curHist = _mm512_add_epi32(curHist,_mm512_cvtepi16_epi32(_mm256_load_si256(base + j)));
            }
            _mm512_store_epi32(buff,curHist);
        }

#if 0
        for (PieceType pt = PAWN; pt <= KING; ++pt)
        {
            Piece pc = make_piece(us,pt);
            for (int i=0; i<64; i+=16)
            {
                __m512i* buff = reinterpret_cast<__m512i*>(&(histBuffer[pt-1][i]));
                __m512i  curHist =  _mm512_cvtepi16_epi32(_mm256_slli_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(&(sharedHistory->pawn_entry(pos)[pc][i]))),1));
                for (int j: {0,1,2,3,5})
                {
                    const __m256i* curConthist = reinterpret_cast<const __m256i*>(&(*continuationHistory[j])[pc][i]);
                    curHist = _mm512_add_epi32(curHist,_mm512_cvtepi16_epi32(_mm256_load_si256(curConthist)));
                }
                _mm512_store_epi32(buff,curHist);

            }
        }
        for (auto& c : histBuffer) for (auto& d : c) dbg_extremes_of(d);
#endif
        #endif // defined
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
                    + 7 * int(PieceValue[capturedPiece]);

        else if constexpr (Type == QUIETS)
        {
            // histories
            m.value = 2 * (*mainHistory)[us][m.raw()];

            #if defined(USE_AVX512)
            m.value += histBuffer[pt-1][to];
            #else

            m.value += 2 * sharedHistory->pawn_entry(pos)[pc][to];
            m.value += (*continuationHistory[0])[pc][to];
            m.value += (*continuationHistory[1])[pc][to];
            m.value += (*continuationHistory[2])[pc][to];
            m.value += (*continuationHistory[3])[pc][to];
            m.value += (*continuationHistory[5])[pc][to];
            #endif // defined


            // bonus for checks
            m.value += (bool(pos.check_squares(pt) & to) && pos.see_ge(m, -75)) * 16384;

            // penalty for moving to a square threatened by a lesser piece
            // or bonus for escaping an attack by a lesser piece.
            int v = 20 * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to));
            m.value += PieceValue[pt] * v;


            if (ply < LOW_PLY_HISTORY_SIZE)
                m.value += 8 * (*lowPlyHistory)[ply][m.raw()] / (1 + ply);
        }

        else  // Type == EVASIONS
        {
            if (pos.capture_stage(m))
                m.value = PieceValue[capturedPiece] + (1 << 28);
            else
                m.value = (*mainHistory)[us][m.raw()] + (*continuationHistory[0])[pc][to];
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

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
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

            partial_insertion_sort(cur, endCur, -3560 * depth);
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

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
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
