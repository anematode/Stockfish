#!/usr/bin/env bash
#
# Build script for AFL++ fuzzing of Stockfish's FEN parser.
#
# Usage: bash scripts/build_fuzz.sh
#
# Prerequisites: afl++ (apt-get install afl++)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"

echo "=== Step 1: Build Stockfish objects with AFL++ instrumentation ==="
cd "$SRC_DIR"
make clean || true
make build ARCH=x86-64-avx2 \
    CXX=afl-clang-fast++ \
    COMP=clang \
    debug=yes \
    sanitize="address undefined" \
    optimize=no \
    -j"$(nproc)" || true
# The above 'make build' will fail at link time because we haven't changed
# anything about main, but the .o files we need are already compiled.
# Check that the objects exist:
if [ ! -f position.o ]; then
    echo "ERROR: Object files were not created. Check afl-clang-fast++ is installed."
    exit 1
fi

echo "=== Step 2: Compile fuzz harness and link ==="

# Collect all .o files except main.o
OBJS=$(ls *.o | grep -v '^main\.o$' | tr '\n' ' ')

# Architecture flags for x86-64-avx2
ARCH_FLAGS="-DUSE_AVX2 -DUSE_SSE41 -DUSE_SSSE3 -DUSE_SSE2 -DUSE_POPCNT -DIS_64BIT -mavx2 -msse4.1 -mssse3 -msse2 -mpopcnt -mbmi"

# Compile the fuzz harness
afl-clang-fast++ -std=c++17 -g -O0 \
    -fsanitize=address,undefined \
    $ARCH_FLAGS \
    -I"$SRC_DIR" \
    -c fuzz_fen.cpp -o fuzz_fen.o

# Link everything together
afl-clang-fast++ -std=c++17 -g -O0 \
    -fsanitize=address,undefined \
    $ARCH_FLAGS \
    $OBJS fuzz_fen.o \
    -lpthread -o stockfish-fuzz

echo ""
echo "=== Build complete: src/stockfish-fuzz ==="
echo ""
echo "Run with:"
echo "  bash scripts/run_fuzz.sh          # parallel on all $(nproc) cores"
echo "  bash scripts/run_fuzz.sh 16       # or specify thread count"
echo ""
echo "Single-core (manual):"
echo "  export ASAN_OPTIONS=\"abort_on_error=1:symbolize=0:detect_leaks=0\""
echo "  export UBSAN_OPTIONS=\"halt_on_error=1\""
echo "  afl-fuzz -i fuzz/seeds -o fuzz/output -x fuzz/fen.dict -m none -- ./src/stockfish-fuzz"
