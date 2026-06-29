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
#elif defined(USE_SVE2)
#include <arm_sve.h>
#include <arm_neon.h>
#include <arm_neon_sve_bridge.h>
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

#elif defined(USE_SVE2)
constexpr uint64_t Planes[] = {
    0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
    0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL, 0xFFFFFFFF00000000ULL
};

// Returns set of active indices, and the number of active indices. Requires
// that the mask has <= 16 active bits.
// Example:
//    m = 0x4000'2020'0010'0121
// -> res.first is  [ 0, 5, 8, 20, 37, 45, 62, xx ... ]
//    res.second is 7
std::pair<uint8x16_t, uint64_t> get_set_bits_sve2(uint64_t mask) {
    assert(popcount(mask) <= 16);
    uint64x2x3_t planes = vld1q_u64_x3(Planes);

    auto neon_to_sve = [] (uint64x2_t v) {
        return svset_neonq_u64(svundef_u64(), v);
    };

    const uint64x2_t mask_v = vdupq_n_u64(mask);
    const svuint64_t mask_sv = neon_to_sve(mask_v);
    const auto bp_01 = svget_neonq_u64(svbext_u64(neon_to_sve(planes.val[0]), mask_sv));
    const auto bp_23 = svget_neonq_u64(svbext_u64(neon_to_sve(planes.val[1]), mask_sv));
    const auto bp_45 = svget_neonq_u64(svbext_u64(neon_to_sve(planes.val[2]), mask_sv));

    size_t count = vaddv_u8(vcnt_u8(vreinterpret_u8_u64(vdup_n_u64(mask))));

    // Pack the six 16-bit planes into the low slots: [m0,m1,m2,m3,m4,m5,0,0]
    const uint8x16_t idx = { 0,1, 8,9, 16,17, 24,25, 32,33, 40,41,
                             0xff,0xff,0xff,0xff };
    uint8x16x3_t tbl = { vreinterpretq_u8_u64(bp_01),
                         vreinterpretq_u8_u64(bp_23),
                         vreinterpretq_u8_u64(bp_45) };
    uint8x16_t uz2 = vqtbl3q_u8(tbl, idx);

    auto interleave = [] (uint8x16_t v) {
        poly64x2_t p = vreinterpretq_p64_u8(v);
        uint64x2_t a = vreinterpretq_u64_p128(
            vmull_p64((poly64_t)vget_low_p64(p), (poly64_t)vget_low_p64(p)) );
        uint64x2_t b = vreinterpretq_u64_p128(vmull_high_p64(p, p));
        a = vrax1q_u64(a, b);
        return vreinterpretq_u8_u64(a);
    };

    uz2 = interleave(uz2);
    uz2 = interleave(uz2);
    uz2 = interleave(uz2);

    return { uz2, count };
}

// Expand the set bits of to_bb into moves originating from `from`
// Small means at most 8 targets (knights and kings), so a
// single 128-bit store suffices
template<bool Small>
inline Move* splat_moves_sve2(Move* moveList, Square from, Bitboard to_bb) {
    const auto [squares, count] = get_set_bits_sve2(to_bb);
    const uint16x8_t fromBits   = vdupq_n_u16(Move(from, SQUARE_ZERO).raw());

    vst1q_u16(reinterpret_cast<uint16_t*>(moveList),
              vorrq_u16(fromBits, vmovl_u8(vget_low_u8(squares))));
    if constexpr (!Small)
        vst1q_u16(reinterpret_cast<uint16_t*>(moveList) + 8,
                  vorrq_u16(fromBits, vmovl_u8(vget_high_u8(squares))));

    return moveList + count;
}

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    assert(popcount(to_bb) <= 8);  // <= 8 pawns per side
    const auto [squares, count] = get_set_bits_sve2(to_bb);

    // to == from + offset, so the from square is recovered by subtraction. The
    // arithmetic is done in 16 bits, which wraps correctly for both signs.
    const uint16x8_t to16   = vmovl_u8(vget_low_u8(squares));
    const uint16x8_t from16 = vsubq_u16(to16, vdupq_n_u16(uint16_t(offset)));
    const uint16x8_t moves  = vorrq_u16(vshlq_n_u16(from16, Move::FromSqShift), to16);

    vst1q_u16(reinterpret_cast<uint16_t*>(moveList), moves);
    return moveList + count;
}

// Fallback for queens, whose attacks can exceed the 16-square limit of
// get_set_bits_sve2().
inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    while (to_bb)
        *moveList++ = Move(from, pop_lsb(to_bb));
    return moveList;
}

#else
template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    while (to_bb)
    {
        Square to   = pop_lsb(to_bb);
        *moveList++ = Move(to - offset, to);
    }
    return moveList;
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
#ifdef USE_AVX512ICL
        if constexpr (Pt != QUEEN)
        {
            moveList = splat_precomputed_moves<Pt>(moveList, from, pos.pieces(), target);
            continue;
        }
#elif defined(USE_SVE2)
        if constexpr (Pt != QUEEN)
        {
            Bitboard b = Attacks::attacks_bb<Pt>(from, pos.pieces()) & target;
            moveList   = splat_moves_sve2<Pt == KNIGHT>(moveList, from, b);
            continue;
        }
#endif
        Bitboard b = Attacks::attacks_bb<Pt>(from, pos.pieces()) & target;

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

#if defined(USE_AVX512ICL)
    moveList = splat_precomputed_moves<KING>(moveList, ksq, 0ULL, b);
#elif defined(USE_SVE2)
    moveList = splat_moves_sve2<true>(moveList, ksq, Attacks::attacks_bb<KING>(ksq) & b);
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
