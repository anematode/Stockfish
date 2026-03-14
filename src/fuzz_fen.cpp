/*
  AFL++ persistent-mode harness for fuzzing Stockfish's FEN parser.

  Targets Position::set() and shallow move generation/execution
  to find crashes caused by malformed FEN strings.
*/

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <unistd.h>

#include "bitboard.h"
#include "movegen.h"
#include "position.h"
#include "types.h"

using namespace Stockfish;

// AFL++ shared-memory fuzzing macros
__AFL_FUZZ_INIT();

int main() {
    Bitboards::init();
    Position::init();

    // Deferred forkserver: fork happens after expensive init
    __AFL_INIT();

    unsigned char* buf = __AFL_FUZZ_TESTCASE_BUF;

    while (__AFL_LOOP(10000)) {
        int len = __AFL_FUZZ_TESTCASE_LEN;
        if (len <= 0)
            continue;

        // Cap input length and build a clean C string
        if (len > 256)
            len = 256;

        char fen[257];
        std::memcpy(fen, buf, len);
        fen[len] = '\0';

        // Sanitize: replace embedded nulls and newlines with spaces
        for (int i = 0; i < len; i++) {
            if (fen[i] == '\0' || fen[i] == '\n' || fen[i] == '\r')
                fen[i] = ' ';
        }

        // Parse the FEN — this is the primary fuzz target
        StateInfo st;
        Position  pos;
        pos.set(std::string(fen), false, &st);

        // Depth-2 perft: exercises move generation on the (possibly corrupt) position
        StateInfo st1, st2;
        for (const auto& m1 : MoveList<LEGAL>(pos)) {
            pos.do_move(m1, st1, nullptr);
            for (const auto& m2 : MoveList<LEGAL>(pos)) {
                pos.do_move(m2, st2, nullptr);
                pos.undo_move(m2);
            }
            pos.undo_move(m1);
        }
    }

    return 0;
}
