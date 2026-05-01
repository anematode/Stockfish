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

#include "bitboard.h"

#include <algorithm>
#include <bitset>
#include <initializer_list>

#include "misc.h"

namespace Stockfish {

uint8_t PopCnt16[1 << 16];
uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];

Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard RayPassBB[SQUARE_NB][SQUARE_NB];

alignas(64) Magic Magics[SQUARE_NB][2];

#ifdef USE_PEXT
using MagicMask = uint16_t;
#else
using MagicMask = Bitboard;
#endif

// Returns an ASCII representation of a bitboard suitable
// to be printed to standard output. Useful for debugging.
std::string Bitboards::pretty(Bitboard b) {

    std::string s = "+---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_8;; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
            s += b & make_square(f, r) ? "| X " : "|   ";

        s += "| " + std::to_string(1 + r) + "\n+---+---+---+---+---+---+---+---+\n";

        if (r == RANK_1)
            break;
    }
    s += "  a   b   c   d   e   f   g   h\n";

    return s;
}

namespace {
[[maybe_unused]] constexpr Bitboard constexpr_pext(Bitboard b, Bitboard m) {
    Bitboard result = 0, bit = 0;
    while (m)
    {
        Bitboard last = m & -m;
        result |= bool(b & last) << bit++;
        m ^= last;
    }
    return result;
}

#ifndef USE_PEXT
// Pre-computed magics
// clang-format off
constexpr Bitboard RookMagicsInit[SQUARE_NB] = {
    0x0a80004000801220ULL, 0x8040004010002008ULL, 0x2080200010008008ULL, 0x1100100008210004ULL,
    0xc200209084020008ULL, 0x2100010004000208ULL, 0x0400081000822421ULL, 0x0200010422048844ULL,
    0x0800800080400024ULL, 0x0001402000401000ULL, 0x3000801000802001ULL, 0x4400800800100083ULL,
    0x0904802402480080ULL, 0x4040800400020080ULL, 0x0018808042000100ULL, 0x4040800080004100ULL,
    0x0040048001458024ULL, 0x00a0004000205000ULL, 0x3100808010002000ULL, 0x4825010010000820ULL,
    0x5004808008000401ULL, 0x2024818004000a00ULL, 0x0005808002000100ULL, 0x2100060004806104ULL,
    0x0080400880008421ULL, 0x4062220600410280ULL, 0x010a004a00108022ULL, 0x0000100080080080ULL,
    0x0021000500080010ULL, 0x0044000202001008ULL, 0x0000100400080102ULL, 0xc020128200040545ULL,
    0x0080002000400040ULL, 0x0000804000802004ULL, 0x0000120022004080ULL, 0x010a386103001001ULL,
    0x9010080080800400ULL, 0x8440020080800400ULL, 0x0004228824001001ULL, 0x000000490a000084ULL,
    0x0080002000504000ULL, 0x200020005000c000ULL, 0x0012088020420010ULL, 0x0010010080080800ULL,
    0x0085001008010004ULL, 0x0002000204008080ULL, 0x0040413002040008ULL, 0x0000304081020004ULL,
    0x0080204000800080ULL, 0x3008804000290100ULL, 0x1010100080200080ULL, 0x2008100208028080ULL,
    0x5000850800910100ULL, 0x8402019004680200ULL, 0x0120911028020400ULL, 0x0000008044010200ULL,
    0x0020850200244012ULL, 0x0020850200244012ULL, 0x0000102001040841ULL, 0x140900040a100021ULL,
    0x000200282410a102ULL, 0x000200282410a102ULL, 0x000200282410a102ULL, 0x4048240043802106ULL,
};
constexpr Bitboard BishopMagicsInit[SQUARE_NB] = {
    0x40106000a1160020ULL, 0x0020010250810120ULL, 0x2010010220280081ULL, 0x002806004050c040ULL,
    0x0002021018000000ULL, 0x2001112010000400ULL, 0x0881010120218080ULL, 0x1030820110010500ULL,
    0x0000120222042400ULL, 0x2000020404040044ULL, 0x8000480094208000ULL, 0x0003422a02000001ULL,
    0x000a220210100040ULL, 0x8004820202226000ULL, 0x0018234854100800ULL, 0x0100004042101040ULL,
    0x0004001004082820ULL, 0x0010000810010048ULL, 0x1014004208081300ULL, 0x2080818802044202ULL,
    0x0040880c00a00100ULL, 0x0080400200522010ULL, 0x0001000188180b04ULL, 0x0080249202020204ULL,
    0x1004400004100410ULL, 0x00013100a0022206ULL, 0x2148500001040080ULL, 0x4241080011004300ULL,
    0x4020848004002000ULL, 0x10101380d1004100ULL, 0x0008004422020284ULL, 0x01010a1041008080ULL,
    0x0808080400082121ULL, 0x0808080400082121ULL, 0x0091128200100c00ULL, 0x0202200802010104ULL,
    0x8c0a020200440085ULL, 0x01a0008080b10040ULL, 0x0889520080122800ULL, 0x100902022202010aULL,
    0x04081a0816002000ULL, 0x0000681208005000ULL, 0x8170840041008802ULL, 0x0a00004200810805ULL,
    0x0830404408210100ULL, 0x2602208106006102ULL, 0x1048300680802628ULL, 0x2602208106006102ULL,
    0x0602010120110040ULL, 0x0941010801043000ULL, 0x000040440a210428ULL, 0x0008240020880021ULL,
    0x0400002012048200ULL, 0x00ac102001210220ULL, 0x0220021002009900ULL, 0x84440c080a013080ULL,
    0x0001008044200440ULL, 0x0004c04410841000ULL, 0x2000500104011130ULL, 0x1a0c010011c20229ULL,
    0x0044800112202200ULL, 0x0434804908100424ULL, 0x0300404822c08200ULL, 0x48081010008a2a80ULL,
};
// clang-format on
#endif

// Computes all rook and bishop attacks at compile time. Magic bitboards are
// used to look up attacks of sliding pieces. As a reference see
// https://www.chessprogramming.org/Magic_Bitboards. In particular, here we use
// the so called "fancy" approach.
constexpr void
init_magics(PieceType pt, MagicMask table[], Magic magics[][2], bool tableAlreadyInit) {
#ifndef USE_PEXT
    const Bitboard* hardcodedMagics = (pt == ROOK) ? RookMagicsInit : BishopMagicsInit;
#endif
    int size = 0;

    for (Square s = SQ_A1; s <= SQ_H8; ++s)
    {
        // Board edges are not considered in the relevant occupancies
        Bitboard edges = ((Rank1BB | Rank8BB) & ~rank_bb(s)) | ((FileABB | FileHBB) & ~file_bb(s));

        // Given a square 's', the mask is the bitboard of sliding attacks from
        // 's' computed on an empty board. The index must be big enough to contain
        // all the attacks for each possible subset of the mask and so is 2 power
        // the number of 1s of the mask. Hence we deduce the size of the shift to
        // apply to the 64 or 32 bits word to get the index.
        Magic&   m       = magics[s][pt - BISHOP];
        Bitboard attacks = Bitboards::sliding_attack(pt, s, 0);
        m.mask           = attacks & ~edges;
#ifdef USE_PEXT
        m.pseudoAttacks = attacks;
#else
        m.shift = (Is64Bit ? 64 : 32) - constexpr_popcount(m.mask);
        m.magic = hardcodedMagics[s];
#endif
        // Set the offset for the attacks table of the square. We have individual
        // table sizes for each square with "Fancy Magic Bitboards".
        m.attacks = s == SQ_A1 ? table : magics[s - 1][pt - BISHOP].attacks + size;
        size      = 0;

        // Use Carry-Rippler trick to enumerate all subsets of m.mask and store
        // the corresponding sliding attack bitboard in the lookup table.
        Bitboard b = 0;
#ifdef USE_PEXT
        Bitboard prevSliding = -1;
#endif
        do
        {
            if (!tableAlreadyInit)
            {
                Bitboard sliding = Bitboards::sliding_attack(pt, s, b);
#ifdef USE_PEXT
                m.attacks[size] =
                  sliding != prevSliding ? constexpr_pext(sliding, attacks) : m.attacks[size - 1];
                prevSliding = sliding;
#else
                m.attacks[m.index(b)] = sliding;
#endif
            }
            size++;
            b = (b - m.mask) & m.mask;
        } while (b);
    }
}

constexpr auto RookTable = []() {
    std::array<MagicMask, 0x19000> result{};
    Magic                          magics[SQUARE_NB][2] = {};
    init_magics(ROOK, result.data(), magics, false);
    return result;
}();
constexpr auto BishopTable = []() {
    std::array<MagicMask, 0x1480> result{};
    Magic                         magics[SQUARE_NB][2] = {};
    init_magics(BISHOP, result.data(), magics, false);
    return result;
}();
}


// Initializes various bitboard tables. It is called at
// startup and relies on global objects to be already zero-initialized.
void Bitboards::init() {

    for (unsigned i = 0; i < (1 << 16); ++i)
        PopCnt16[i] = uint8_t(std::bitset<16>(i).count());

    for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
        for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
            SquareDistance[s1][s2] = std::max(distance<File>(s1, s2), distance<Rank>(s1, s2));

    init_magics(ROOK, const_cast<MagicMask*>(RookTable.data()), Magics, true);
    init_magics(BISHOP, const_cast<MagicMask*>(BishopTable.data()), Magics, true);

    for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
    {
        for (PieceType pt : {BISHOP, ROOK})
            for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
            {
                if (PseudoAttacks[pt][s1] & s2)
                {
                    LineBB[s1][s2] = (attacks_bb(pt, s1, 0) & attacks_bb(pt, s2, 0)) | s1 | s2;
                    BetweenBB[s1][s2] =
                      (attacks_bb(pt, s1, square_bb(s2)) & attacks_bb(pt, s2, square_bb(s1)));
                    RayPassBB[s1][s2] =
                      attacks_bb(pt, s1, 0) & (attacks_bb(pt, s2, square_bb(s1)) | s2);
                }
                BetweenBB[s1][s2] |= s2;
            }
    }
}

}  // namespace Stockfish
