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

//Definition of input features QK4 of NNUE evaluation function

#ifndef NNUE_FEATURES_QK4_H_INCLUDED
#define NNUE_FEATURES_QK4_H_INCLUDED

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

// Feature QK4: 4-State Queen Check Threat features (6,144 features)
class QK4 {
   public:
    // Hash value embedded in the evaluation file
    static constexpr u32 HashValue = 0x41514b34u;

    // Number of feature dimensions
    static constexpr IndexType Dimensions = 6144;

    // Index base in the auxiliary weights array
    static constexpr IndexType IndexBase = 64368;

    // Maximum number of simultaneously active features.
    static constexpr IndexType MaxActiveDimensions = 24;
    using IndexList                                = ValueList<IndexType, 256>;
    using DiffType                                 = DirtyPiece;

    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    static void append_changed_indices(
      Color perspective, Square ksq, const DiffType& diff, bool opponent_has_queen, IndexList& removed, IndexList& added);

    static bool requires_refresh(const DiffType& diff, Color perspective);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_QK4_H_INCLUDED
