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

#include "movegen.h"

#include <cassert>
#include <initializer_list>

#include "attacks.h"
#include "bitboard.h"
#include "position.h"

#if defined(USE_AVX512ICL)
    #include <array>
    #include <algorithm>
    #include <immintrin.h>
#elif defined(USE_NEON)
    #include <arm_neon.h>
#endif

namespace Stockfish {

namespace {

#if defined(USE_AVX512ICL)

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    assert(popcount(to_bb) <= 8);  // <= 8 pawns per side

    const __m128i toSquares =
      _mm_cvtepi8_epi16(_mm512_castsi512_si128(_mm512_maskz_compress_epi8(to_bb, AllSquares)));
    const __m128i fromSquares = _mm_subs_epi16(toSquares, _mm_set1_epi16(offset));
    const __m128i moves       = _mm_or_si128(_mm_slli_epi16(fromSquares, Move::FromSqShift),
                                             _mm_slli_epi16(toSquares, Move::ToSqShift));

    _mm_storeu_si128(reinterpret_cast<__m128i*>(moveList), moves);
    return moveList + popcount(to_bb);
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    assert(popcount(to_bb) <= 32);  // Q can attack up to 27 squares

    const __m512i fromVec = _mm512_set1_epi16(Move(from, SQUARE_ZERO).raw());
    const __m512i toSquares =
      _mm512_cvtepi8_epi16(_mm512_castsi512_si256(_mm512_maskz_compress_epi8(to_bb, AllSquares)));
    const __m512i moves = _mm512_or_si512(fromVec, _mm512_slli_epi16(toSquares, Move::ToSqShift));

    _mm512_storeu_si512(moveList, moves);
    return moveList + popcount(to_bb);
}

// Rook/bishop, indexed by (Pt - BISHOP) and from sq
// Moves are provided in ascending order of the piece's attacks on an empty board
alignas(64) constexpr auto SliderMoves = []() {
    std::array<std::array<std::array<Move, 16>, SQUARE_NB>, 2> arr{};
    for (PieceType pt : {BISHOP, ROOK})
    {
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
        {
            Bitboard bb = Attacks::PseudoAttacks[pt][s];
            int      i  = 0;
            while (bb)
            {
                arr[pt - BISHOP][s][i++] = Move(s, Square(constexpr_lsb(bb)));
                bb &= bb - 1;
            }
        }
    }
    return arr;
}();

// Knight/king analog of the above
alignas(64) constexpr auto KnightKingMoves = []() {
    std::array<std::array<std::array<Move, 8>, SQUARE_NB>, 2> arr{};
    for (PieceType pt : {KNIGHT, KING})
    {
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
        {
            Bitboard bb = Attacks::PseudoAttacks[pt][s];
            int      i  = 0;
            while (bb)
            {
                arr[pt == KING][s][i++] = Move(s, Square(constexpr_lsb(bb)));
                bb &= bb - 1;
            }
        }
    }
    return arr;
}();

template<PieceType Pt>
inline Move*
splat_precomputed_moves(Move* moveList, Square from, Bitboard occupied, Bitboard target) {
    static_assert(Pt != QUEEN && Pt != PAWN, "Unsupported piece type");

    // The nth bit in the mask corresponds to the nth square in the piece's pseudo-attacks
    u32 mask;
    if constexpr (Pt == BISHOP || Pt == ROOK)
    {
        const Attacks::Magic& magic = Attacks::magic(from, Pt);

        mask = magic.attacks[magic.index(occupied)];
        mask &= pext(target, magic.pseudoAttacks);

        const __m256i moves =
          *reinterpret_cast<const __m256i*>(SliderMoves[Pt - BISHOP][from].data());
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(moveList),
                            _mm256_maskz_compress_epi16(mask, moves));
    }
    else
    {
        mask = pext(target, Attacks::PseudoAttacks[Pt][from]);

        __m128i moves = *reinterpret_cast<const __m128i*>(KnightKingMoves[Pt == KING][from].data());
        _mm_storeu_si128(reinterpret_cast<__m128i*>(moveList),
                         _mm_maskz_compress_epi16(mask, moves));
    }

    return moveList + popcount(mask);
}

#else

static inline uint8x16_t iota16() {
    static const uint8_t k[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    return vld1q_u8(k);
}

alignas(16) std::array<uint8_t, 16> nib_lut = [] () {
    std::array<uint8_t, 16> r{};
    for (int v = 0; v < 16; v++) {
        uint8_t pk = 0;
        int n = 0;
        for (int b = 0; b < 4; b++)
            if (v & (1 << b)) pk |= (uint8_t)(b << (2 * n++));
        r[v] = pk;
    }
    return r;
} ();

std::pair<uint8x16_t, size_t> get_active_indices_and_popcount(uint64_t bitset) {
    assert(popcount(bitset) <= 15);

    const uint8x16_t iota = iota16();
    const uint8x16_t one  = vdupq_n_u8(1);

    uint64_t x = bitset;
    x -= (x >> 1) & 0x5555555555555555ull;
    // Per-nibble popcount
    x  = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    // Each nibble in npref_w[4*n:4*n+3] is the # of set bits in bitset[:4*n+3],
    // via a prefix sum
    uint64_t npref_w = x * 0x1111111111111111ull;

    uint8x16_t whole = vcombine_u8(vcreate_u8(bitset), vcreate_u8(npref_w));
    uint8x16_t lo    = vandq_u8(whole, vdupq_n_u8(0x0F));
    uint8x16_t hi    = vshrq_n_u8(whole, 4);
    uint8x16_t nib   = vzip1q_u8(lo, hi);
    uint8x16_t np    = vzip2q_u8(lo, hi);
    uint8x16_t pack_by_nib = vqtbl1q_u8(vld1q_u8(nib_lut.data()), nib);

    // srcNib[k] = #{ j : np[j] <= k } , binary search
    uint8x16_t t8 = vcleq_u8(vdupq_laneq_u8(np, 7), iota);          // np[7]  <= k ?
    uint8x16_t srcNib = vandq_u8(t8, vdupq_n_u8(8));                // 0 or 8
    uint8x16_t p3  = vcleq_u8(vdupq_laneq_u8(np, 3), iota);
    uint8x16_t p11 = vcleq_u8(vdupq_laneq_u8(np, 11), iota);
    uint8x16_t t4  = vbslq_u8(t8, p11, p3);                         // np[sn+3] <= k ?
    srcNib = vbslq_u8(t4, vaddq_u8(srcNib, vdupq_n_u8(4)), srcNib);
    uint8x16_t pr = vqtbl1q_u8(np, vaddq_u8(srcNib, one));          // step 3, probe np[sn+1]
    srcNib = vbslq_u8(vcleq_u8(pr, iota), vaddq_u8(srcNib, vdupq_n_u8(2)), srcNib);
    pr = vqtbl1q_u8(np, srcNib);                                    // step 4, probe np[sn]
    srcNib = vbslq_u8(vcleq_u8(pr, iota), vaddq_u8(srcNib, one), srcNib);

    // within-nibble rank
    uint8x16_t nstart = vqtbl1q_u8(np, vsubq_u8(srcNib, one));
    uint8x16_t lr     = vsubq_u8(iota, nstart);

    // within-nibble position
    uint8x16_t pack = vqtbl1q_u8(pack_by_nib, srcNib);
    int8x16_t  sh   = vreinterpretq_s8_u8(vshlq_n_u8(lr, 1));
    uint8x16_t pos  = vandq_u8(vshlq_u8(pack, vnegq_s8(sh)), vdupq_n_u8(3));
    // add nibble base
    return { vaddq_u8(vshlq_n_u8(srcNib, 2), pos), npref_w >> 60 };
}

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    while (to_bb)
    {
        Square to   = pop_lsb(to_bb);
        *moveList++ = Move(to - offset, to);
    }
    return moveList;
}

inline Move* splat_moves_15(Move* moveList, Square from, Bitboard to_bb) {
    auto [ indices, count ] = get_active_indices_and_popcount(to_bb);

    uint16x8_t f = vdupq_n_u16(Move(from, SQ_A1).raw());
    Move* p = moveList;
    for (uint8x8_t q : { vget_low_u8(indices), vget_high_u8(indices) }) {
        uint16x8_t m = vmovl_u8(q);
        m = vshlq_n_u16(m, Move::ToSqShift);

        uint16x8_t t = vaddq_u16(f, m);
        vst1q_u16(reinterpret_cast<uint16_t*>(p), t);
        p += 8;
    }

    return moveList + count;
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    while (to_bb)
        *moveList++ = Move(from, pop_lsb(to_bb));
    return moveList;
}

#endif

template<GenType Type, Direction D, bool Enemy>
Move* make_promotions(Move* moveList, [[maybe_unused]] Square to) {

    constexpr bool all = Type == EVASIONS || Type == NON_EVASIONS;

    if constexpr (Type == CAPTURES || all)
        *moveList++ = Move::make<PROMOTION>(to - D, to, QUEEN);

    if constexpr ((Type == CAPTURES && Enemy) || (Type == QUIETS && !Enemy) || all)
    {
        *moveList++ = Move::make<PROMOTION>(to - D, to, ROOK);
        *moveList++ = Move::make<PROMOTION>(to - D, to, BISHOP);
        *moveList++ = Move::make<PROMOTION>(to - D, to, KNIGHT);
    }

    return moveList;
}


template<Color Us, GenType Type>
Move* generate_pawn_moves(const Position& pos, Move* moveList, Bitboard target) {

    constexpr Color     Them     = ~Us;
    constexpr Bitboard  TRank7BB = (Us == WHITE ? Rank7BB : Rank2BB);
    constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
    constexpr Direction Up       = pawn_push(Us);
    constexpr Direction UpRight  = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft   = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

    const Bitboard emptySquares = ~pos.pieces();
    const Bitboard enemies      = Type == EVASIONS ? pos.checkers() : pos.pieces(Them);

    Bitboard pawnsOn7    = pos.pieces(Us, PAWN) & TRank7BB;
    Bitboard pawnsNotOn7 = pos.pieces(Us, PAWN) & ~TRank7BB;

    // Single and double pawn pushes, no promotions
    if constexpr (Type != CAPTURES)
    {
        Bitboard b1 = shift<Up>(pawnsNotOn7) & emptySquares;
        Bitboard b2 = shift<Up>(b1 & TRank3BB) & emptySquares;

        if constexpr (Type == EVASIONS)  // Consider only blocking squares
        {
            b1 &= target;
            b2 &= target;
        }

        moveList = splat_pawn_moves<Up>(moveList, b1);
        moveList = splat_pawn_moves<Up + Up>(moveList, b2);
    }

    // Promotions and underpromotions
    if (pawnsOn7)
    {
        Bitboard b1 = shift<UpRight>(pawnsOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsOn7) & enemies;
        Bitboard b3 = shift<Up>(pawnsOn7) & emptySquares;

        if constexpr (Type == EVASIONS)
            b3 &= target;

        while (b1)
            moveList = make_promotions<Type, UpRight, true>(moveList, pop_lsb(b1));

        while (b2)
            moveList = make_promotions<Type, UpLeft, true>(moveList, pop_lsb(b2));

        while (b3)
            moveList = make_promotions<Type, Up, false>(moveList, pop_lsb(b3));
    }

    // Standard and en passant captures
    if constexpr (Type == CAPTURES || Type == EVASIONS || Type == NON_EVASIONS)
    {
        Bitboard b1 = shift<UpRight>(pawnsNotOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsNotOn7) & enemies;

        moveList = splat_pawn_moves<UpRight>(moveList, b1);
        moveList = splat_pawn_moves<UpLeft>(moveList, b2);

        if (pos.ep_square() != SQ_NONE)
        {
            assert(rank_of(pos.ep_square()) == relative_rank(Us, RANK_6));

            // An en passant capture cannot resolve a discovered check
            if (Type == EVASIONS && (target & (pos.ep_square() + Up)))
                return moveList;

            b1 = pawnsNotOn7 & Attacks::attacks_bb<PAWN>(pos.ep_square(), Them);

            assert(b1);

            while (b1)
                *moveList++ = Move::make<EN_PASSANT>(pop_lsb(b1), pos.ep_square());
        }
    }

    return moveList;
}


template<Color Us, PieceType Pt>
Move* generate_moves(const Position& pos, Move* moveList, Bitboard target) {

    static_assert(Pt != KING && Pt != PAWN, "Unsupported piece type in generate_moves()");

    Bitboard bb = pos.pieces(Us, Pt);

    while (bb)
    {
        Square from = pop_lsb(bb);
        if constexpr (Pt != QUEEN)
        {
#ifdef USE_AVX512ICL
            moveList = splat_precomputed_moves<Pt>(moveList, from, pos.pieces(), target);
            continue;
#endif
        }
        Bitboard b = Attacks::attacks_bb<Pt>(from, pos.pieces()) & target;
#ifdef USE_NEON
        if constexpr (Pt != QUEEN) {
            moveList = splat_moves_15(moveList, from, b);
            continue;
        }
#endif

        moveList = splat_moves(moveList, from, b);
    }

    return moveList;
}


template<Color Us, GenType Type>
Move* generate_all(const Position& pos, Move* moveList) {

    static_assert(Type != LEGAL, "Unsupported type in generate_all()");

    const Square ksq = pos.square<KING>(Us);
    Bitboard     target;

    // Skip generating non-king moves when in double check
    if (Type != EVASIONS || !more_than_one(pos.checkers()))
    {
        target = Type == EVASIONS     ? Attacks::between_bb(ksq, lsb(pos.checkers()))
               : Type == NON_EVASIONS ? ~pos.pieces(Us)
               : Type == CAPTURES     ? pos.pieces(~Us)
                                      : ~pos.pieces();  // QUIETS

        moveList = generate_pawn_moves<Us, Type>(pos, moveList, target);
        moveList = generate_moves<Us, KNIGHT>(pos, moveList, target);
        moveList = generate_moves<Us, BISHOP>(pos, moveList, target);
        moveList = generate_moves<Us, ROOK>(pos, moveList, target);
        moveList = generate_moves<Us, QUEEN>(pos, moveList, target);
    }

    Bitboard b = Type == EVASIONS ? ~pos.pieces(Us) : target;

#ifdef USE_AVX512ICL
    moveList = splat_precomputed_moves<KING>(moveList, ksq, 0ULL, b);
#else
    moveList = splat_moves(moveList, ksq, Attacks::attacks_bb<KING>(ksq) & b);
#endif

    if ((Type == QUIETS || Type == NON_EVASIONS) && pos.can_castle(Us & ANY_CASTLING))
        for (CastlingRights cr : {Us & KING_SIDE, Us & QUEEN_SIDE})
            if (!pos.castling_impeded(cr) && pos.can_castle(cr))
                *moveList++ = Move::make<CASTLING>(ksq, pos.castling_rook_square(cr));

    return moveList;
}

}  // namespace


// <CAPTURES>     Generates all pseudo-legal captures plus queen promotions
// <QUIETS>       Generates all pseudo-legal non-captures and underpromotions
// <EVASIONS>     Generates all pseudo-legal check evasions
// <NON_EVASIONS> Generates all pseudo-legal captures and non-captures
//
// Returns a pointer to the end of the move list.
template<GenType Type>
Move* generate(const Position& pos, Move* moveList) {

    static_assert(Type != LEGAL, "Unsupported type in generate()");
    assert((Type == EVASIONS) == bool(pos.checkers()));

    Color us = pos.side_to_move();

    return us == WHITE ? generate_all<WHITE, Type>(pos, moveList)
                       : generate_all<BLACK, Type>(pos, moveList);
}

// Explicit template instantiations
template Move* generate<CAPTURES>(const Position&, Move*);
template Move* generate<QUIETS>(const Position&, Move*);
template Move* generate<EVASIONS>(const Position&, Move*);
template Move* generate<NON_EVASIONS>(const Position&, Move*);

// generate<LEGAL> generates all the legal moves in the given position

template<>
Move* generate<LEGAL>(const Position& pos, Move* moveList) {

    Color    us     = pos.side_to_move();
    Bitboard pinned = pos.blockers_for_king(us) & pos.pieces(us);
    Square   ksq    = pos.square<KING>(us);
    Move*    cur    = moveList;

    moveList =
      pos.checkers() ? generate<EVASIONS>(pos, moveList) : generate<NON_EVASIONS>(pos, moveList);
    while (cur != moveList)
        if (((pinned & cur->from_sq()) || cur->from_sq() == ksq || cur->type_of() == EN_PASSANT)
            && !pos.legal(*cur))
            *cur = *(--moveList);
        else
            ++cur;

    return moveList;
}

}  // namespace Stockfish
