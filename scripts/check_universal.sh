#!/bin/bash

# Verify that the universal binary selects the correct per-arch build under
# Intel SDE emulation for a range of target CPUs
#
# Usage: check_universal.sh STOCKFISH_EXE SDE_EXE EXPECTED_BENCH

set -eu

if [ $# -ne 3 ]; then
    echo "Usage: $0 STOCKFISH_EXE SDE_EXE EXPECTED_BENCH" >&2
    exit 2
fi

STOCKFISH_EXE=$1
SDE_EXE=$2
EXPECTED_BENCH=$3

if [ "$(uname)" = 'Linux' ]; then
# Windows 11 doesn't support these old arches
PAIRS="
p4p:x86-64
nhm:x86-64-sse41-popcnt
"
else
PAIRS=''
fi

PAIRS="$PAIRS
snb:x86-64-sse41-popcnt
ivb:x86-64-sse41-popcnt
hsw:x86-64-bmi2
skl:x86-64-bmi2
skx:x86-64-avx512
clx:x86-64-vnni512
icl:x86-64-avx512icl
adl:x86-64-avxvnni
"

JOBS=$(nproc 2>/dev/null || echo 4)

check_pair() {
    pair=$1
    cpu=${pair%%:*}
    expected_compiler=${pair##*:}
    compiler_out=$("$SDE_EXE" "-$cpu" -- "$STOCKFISH_EXE" compiler 2>&1 || true)
    bench_out=$("$SDE_EXE" "-$cpu" -- "$STOCKFISH_EXE" bench 2>&1 || true)
    actual_compiler=$(printf '%s\n' "$compiler_out" | awk -F: '/Compilation architecture/ {
        sub(/^[[:space:]]+/, "", $2); sub(/[[:space:]]+$/, "", $2); print $2; exit
    }')
    actual_bench=$(printf '%s\n' "$bench_out" | awk -F: '/Nodes searched/ {
        sub(/^[[:space:]]+/, "", $2); sub(/[[:space:]]+$/, "", $2); print $2; exit
    }')
    if [ "$actual_compiler" != "$expected_compiler" ] || [ "$actual_bench" != "$EXPECTED_BENCH" ]; then
        printf '===== CPU %s output (expected %s/%s, got %s/%s) =====\n' \
            "$cpu" "$expected_compiler" "$EXPECTED_BENCH" "${actual_compiler:--}" "$actual_bench" >&2
        return 1
    fi
    printf 'CPU %s ok\n' "$cpu" >&2
    return 0
}

FAIL=0
running=0

reap_one() {
    if ! wait -n; then
        FAIL=1
    fi
    running=$((running - 1))
}

for pair in $PAIRS; do
    while [ "$running" -ge "$JOBS" ]; do
        reap_one
    done
    check_pair "$pair" &
    running=$((running + 1))
done

while [ "$running" -gt 0 ]; do
    reap_one
done

if [ "$FAIL" != 0 ]; then
    echo "check_universal.sh: failed"
    exit 1
fi

echo "check_universal.sh: Good! Universal binary has correct hardware detection."
