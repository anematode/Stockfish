#!/usr/bin/env bash
#
# Run AFL++ fuzzing in parallel across all available cores.
#
# Usage:
#   bash scripts/run_fuzz.sh [NUM_THREADS]
#
# NUM_THREADS defaults to $(nproc) if not specified.
# Requires: stockfish-fuzz binary built via scripts/build_fuzz.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FUZZ_BIN="$ROOT_DIR/src/stockfish-fuzz"
SEEDS_DIR="$ROOT_DIR/fuzz/seeds"
DICT="$ROOT_DIR/fuzz/fen.dict"
OUTPUT_DIR="$ROOT_DIR/fuzz/output"

THREADS="${1:-$(nproc)}"

if [ ! -x "$FUZZ_BIN" ]; then
    echo "ERROR: $FUZZ_BIN not found. Run scripts/build_fuzz.sh first."
    exit 1
fi

if [ "$THREADS" -lt 1 ]; then
    echo "ERROR: Need at least 1 thread."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

export AFL_AUTORESUME=1
export ASAN_OPTIONS="abort_on_error=1:symbolize=0:detect_leaks=0"
export UBSAN_OPTIONS="halt_on_error=1"

# Trap to kill all background fuzzers on exit
cleanup() {
    echo ""
    echo "=== Stopping all fuzzer instances ==="
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo "All fuzzers stopped."
}
trap cleanup EXIT INT TERM

echo "=== Starting $THREADS parallel AFL++ instances ==="
echo "    Output: $OUTPUT_DIR"
echo "    Binary: $FUZZ_BIN"
echo ""

# Instance 0: Main fuzzer (deterministic mutations)
echo "[main] Starting main fuzzer instance..."
afl-fuzz -M main \
    -i "$SEEDS_DIR" \
    -o "$OUTPUT_DIR" \
    -x "$DICT" \
    -m none \
    -- "$FUZZ_BIN" &

# Small delay so the main instance creates the output dir structure
sleep 1

# Instances 1..N-1: Secondary fuzzers (random mutations)
for i in $(seq 1 $((THREADS - 1))); do
    echo "[s$i] Starting secondary fuzzer instance $i/$((THREADS - 1))..."
    afl-fuzz -S "s$i" \
        -i "$SEEDS_DIR" \
        -o "$OUTPUT_DIR" \
        -x "$DICT" \
        -m none \
        -- "$FUZZ_BIN" &
done

echo ""
echo "=== All $THREADS instances launched ==="
echo "    Monitor with: afl-whatsup $OUTPUT_DIR"
echo "    Or:           watch -n5 afl-whatsup $OUTPUT_DIR"
echo ""
echo "Press Ctrl+C to stop all instances."
echo ""

# Wait for all background jobs
wait
