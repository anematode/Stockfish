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

#include "qk4.h"

#include <algorithm>
#include <cmath>

#include "../../attacks.h"
#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"

#include <iostream>

namespace Stockfish::Eval::NNUE::Features {

void QK4::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    Square ksq = pos.square<KING>(perspective);
    Color opp_color = ~perspective;
    Bitboard queens = pos.pieces(opp_color, QUEEN);
    if (!queens)
        return;


    // Horizontally mirror if king is on file A-D
    int flip_h = file_of(ksq) < FILE_E ? 0x7 : 0x0;

    // Oriented king square
    Square oriented_ksq = Square(int(ksq) ^ (56 * perspective) ^ flip_h);
    int king_bucket = int(oriented_ksq);

    const int RAY_DIRECTIONS[8][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
    };

    while (queens) {
        Square qsq = pop_lsb(queens);
        // Find check threat squares (where the queen can move and deliver check)
        Bitboard check_squares = Attacks::attacks_bb<QUEEN>(qsq, pos.pieces()) &
                Attacks::attacks_bb<QUEEN>(ksq, pos.pieces());
        // Exclude squares occupied by opponent's own pieces
        check_squares &= ~pos.pieces(opp_color);
        while (check_squares) {
            Square check_sq = pop_lsb(check_squares);
            Square oriented_check_sq = Square(int(check_sq) ^ (56 * perspective) ^ flip_h);

            int ofd = int(file_of(oriented_check_sq)) - int(file_of(oriented_ksq));
            int ord = int(rank_of(oriented_check_sq)) - int(rank_of(oriented_ksq));
            int sf = (ofd == 0) ? 0 : ((ofd > 0) ? 1 : -1);
            int sr = (ord == 0) ? 0 : ((ord > 0) ? 1 : -1);

            int dir_idx = -1;
            for (int i = 0; i < 8; ++i) {
                if (RAY_DIRECTIONS[i][0] == sf && RAY_DIRECTIONS[i][1] == sr) {
                    dir_idx = i;
                    break;
                }
            }

            if (dir_idx >= 0) {
                int dist = std::max(std::abs(ofd), std::abs(ord)) - 1;
                int ray = dir_idx * 3 + std::min(dist, 2);

                Bitboard attackers = pos.attackers_to(check_sq);

                // Contested state logic (King Capture Rule)
                Bitboard opp_attackers = attackers & pos.pieces(opp_color);
                bool protected_by_friendly = (opp_attackers & ~square_bb(qsq));

                Bitboard our_attackers = attackers & pos.pieces(perspective);
                bool others_attack = (our_attackers & ~square_bb(ksq));
                bool king_attacks = our_attackers & ksq;

                bool can_be_taken = others_attack || (king_attacks && !protected_by_friendly);

                int state = 0;
                if (protected_by_friendly && !can_be_taken) state = 1;
                else if (protected_by_friendly && can_be_taken) state = 2;
                else if (!protected_by_friendly && can_be_taken) state = 3;

                IndexType index = IndexBase + king_bucket * 96 + ray * 4 + state;
                active.push_back(index);
            }
        }
    }
}

void QK4::append_changed_indices(
  Color perspective, Square ksq, const DiffType& diff, bool opponent_has_queen, IndexList& removed, IndexList& added) {
    (void) perspective;
    (void) ksq;
    (void) diff;
    (void) opponent_has_queen;
    (void) removed;
    (void) added;
}

bool QK4::requires_refresh(const DiffType& diff, Color perspective) {
    (void) diff;
    (void) perspective;
    return false;
}

}  // namespace Stockfish::Eval::NNUE::Features
