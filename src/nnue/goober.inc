

// --- Generated C++ Code Blocks ---
if constexpr (M == 0 && N == 0) {
    ctx->template apply_threats<>();
} else if constexpr (M == 0 && N == 1) {
    ctx->template apply_threats<Sub>(removed[0]);
} else if constexpr (M == 1 && N == 0) {
    ctx->template apply_threats<Add>(added[0]);
} else if constexpr (M == 0 && N == 2) {
    ctx->template apply_threats<Sub, Sub>(removed[0], removed[1]);
} else if constexpr (M == 1 && N == 1) {
    ctx->template apply_threats<Add, Sub>(added[0], removed[0]);
} else if constexpr (M == 2 && N == 0) {
    ctx->template apply_threats<Add, Add>(added[0], added[1]);
} else if constexpr (M == 0 && N == 3) {
    ctx->template apply_threats<Sub, Sub, Sub>(removed[0], removed[1], removed[2]);
} else if constexpr (M == 1 && N == 2) {
    ctx->template apply_threats<Add, Sub, Sub>(added[0], removed[0], removed[1]);
} else if constexpr (M == 2 && N == 1) {
    ctx->template apply_threats<Add, Add, Sub>(added[0], added[1], removed[0]);
} else if constexpr (M == 3 && N == 0) {
    ctx->template apply_threats<Add, Add, Add>(added[0], added[1], added[2]);
} else if constexpr (M == 0 && N == 4) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 1 && N == 3) {
    ctx->template apply_threats<Add, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2]);
} else if constexpr (M == 2 && N == 2) {
    ctx->template apply_threats<Add, Add, Sub, Sub>(added[0], added[1], removed[0], removed[1]);
} else if constexpr (M == 3 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Sub>(added[0], added[1], added[2], removed[0]);
} else if constexpr (M == 4 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add>(added[0], added[1], added[2], added[3]);
} else if constexpr (M == 0 && N == 5) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 1 && N == 4) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 2 && N == 3) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2]);
} else if constexpr (M == 3 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1]);
} else if constexpr (M == 4 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], removed[0]);
} else if constexpr (M == 5 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4]);
} else if constexpr (M == 0 && N == 6) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4], removed[5]);
} else if constexpr (M == 1 && N == 5) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 2 && N == 4) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 3 && N == 3) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1], removed[2]);
} else if constexpr (M == 4 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], added[3], removed[0], removed[1]);
} else if constexpr (M == 5 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], added[4], removed[0]);
} else if constexpr (M == 6 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4], added[5]);
} else if constexpr (M == 0 && N == 7) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6]);
} else if constexpr (M == 1 && N == 6) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5]);
} else if constexpr (M == 2 && N == 5) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 3 && N == 4) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 4 && N == 3) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], removed[0], removed[1], removed[2]);
} else if constexpr (M == 5 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], removed[0], removed[1]);
} else if constexpr (M == 6 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], removed[0]);
} else if constexpr (M == 7 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4], added[5], added[6]);
} else if constexpr (M == 0 && N == 8) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7]);
} else if constexpr (M == 1 && N == 7) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6]);
} else if constexpr (M == 2 && N == 6) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5]);
} else if constexpr (M == 3 && N == 5) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 4 && N == 4) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 5 && N == 3) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], removed[0], removed[1], removed[2]);
} else if constexpr (M == 6 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], removed[0], removed[1]);
} else if constexpr (M == 7 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], removed[0]);
} else if constexpr (M == 8 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7]);
} else if constexpr (M == 0 && N == 9) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8]);
} else if constexpr (M == 1 && N == 8) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7]);
} else if constexpr (M == 2 && N == 7) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6]);
} else if constexpr (M == 3 && N == 6) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5]);
} else if constexpr (M == 4 && N == 5) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 5 && N == 4) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 6 && N == 3) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], removed[0], removed[1], removed[2]);
} else if constexpr (M == 7 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], removed[0], removed[1]);
} else if constexpr (M == 8 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], removed[0]);
} else if constexpr (M == 9 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8]);
} else if constexpr (M == 0 && N == 10) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9]);
} else if constexpr (M == 1 && N == 9) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8]);
} else if constexpr (M == 2 && N == 8) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7]);
} else if constexpr (M == 3 && N == 7) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6]);
} else if constexpr (M == 4 && N == 6) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5]);
} else if constexpr (M == 5 && N == 5) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 6 && N == 4) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 7 && N == 3) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], removed[0], removed[1], removed[2]);
} else if constexpr (M == 8 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], removed[0], removed[1]);
} else if constexpr (M == 9 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], removed[0]);
} else if constexpr (M == 10 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9]);
} else if constexpr (M == 0 && N == 11) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10]);
} else if constexpr (M == 1 && N == 10) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9]);
} else if constexpr (M == 2 && N == 9) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8]);
} else if constexpr (M == 3 && N == 8) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7]);
} else if constexpr (M == 4 && N == 7) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6]);
} else if constexpr (M == 5 && N == 6) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5]);
} else if constexpr (M == 6 && N == 5) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 7 && N == 4) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 8 && N == 3) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], removed[0], removed[1], removed[2]);
} else if constexpr (M == 9 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], removed[0], removed[1]);
} else if constexpr (M == 10 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], removed[0]);
} else if constexpr (M == 11 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10]);
} else if constexpr (M == 0 && N == 12) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10], removed[11]);
} else if constexpr (M == 1 && N == 11) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10]);
} else if constexpr (M == 2 && N == 10) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9]);
} else if constexpr (M == 3 && N == 9) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8]);
} else if constexpr (M == 4 && N == 8) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7]);
} else if constexpr (M == 5 && N == 7) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6]);
} else if constexpr (M == 6 && N == 6) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5]);
} else if constexpr (M == 7 && N == 5) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 8 && N == 4) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 9 && N == 3) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], removed[0], removed[1], removed[2]);
} else if constexpr (M == 10 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], removed[0], removed[1]);
} else if constexpr (M == 11 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], removed[0]);
} else if constexpr (M == 12 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], added[11]);
} else if constexpr (M == 0 && N == 13) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10], removed[11], removed[12]);
} else if constexpr (M == 1 && N == 12) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10], removed[11]);
} else if constexpr (M == 2 && N == 11) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10]);
} else if constexpr (M == 3 && N == 10) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9]);
} else if constexpr (M == 4 && N == 9) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8]);
} else if constexpr (M == 5 && N == 8) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7]);
} else if constexpr (M == 6 && N == 7) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6]);
} else if constexpr (M == 7 && N == 6) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5]);
} else if constexpr (M == 8 && N == 5) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 9 && N == 4) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 10 && N == 3) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], removed[0], removed[1], removed[2]);
} else if constexpr (M == 11 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], removed[0], removed[1]);
} else if constexpr (M == 12 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], added[11], removed[0]);
} else if constexpr (M == 13 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], added[11], added[12]);
} else if constexpr (M == 0 && N == 14) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10], removed[11], removed[12], removed[13]);
} else if constexpr (M == 1 && N == 13) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10], removed[11], removed[12]);
} else if constexpr (M == 2 && N == 12) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10], removed[11]);
} else if constexpr (M == 3 && N == 11) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10]);
} else if constexpr (M == 4 && N == 10) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9]);
} else if constexpr (M == 5 && N == 9) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8]);
} else if constexpr (M == 6 && N == 8) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7]);
} else if constexpr (M == 7 && N == 7) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6]);
} else if constexpr (M == 8 && N == 6) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5]);
} else if constexpr (M == 9 && N == 5) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 10 && N == 4) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 11 && N == 3) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], removed[0], removed[1], removed[2]);
} else if constexpr (M == 12 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], added[11], removed[0], removed[1]);
} else if constexpr (M == 13 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], added[11], added[12], removed[0]);
} else if constexpr (M == 14 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], added[11], added[12], added[13]);
} else if constexpr (M == 0 && N == 15) {
    ctx->template apply_threats<Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10], removed[11], removed[12], removed[13], removed[14]);
} else if constexpr (M == 1 && N == 14) {
    ctx->template apply_threats<Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10], removed[11], removed[12], removed[13]);
} else if constexpr (M == 2 && N == 13) {
    ctx->template apply_threats<Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10], removed[11], removed[12]);
} else if constexpr (M == 3 && N == 12) {
    ctx->template apply_threats<Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10], removed[11]);
} else if constexpr (M == 4 && N == 11) {
    ctx->template apply_threats<Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9], removed[10]);
} else if constexpr (M == 5 && N == 10) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8], removed[9]);
} else if constexpr (M == 6 && N == 9) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7], removed[8]);
} else if constexpr (M == 7 && N == 8) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6], removed[7]);
} else if constexpr (M == 8 && N == 7) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5], removed[6]);
} else if constexpr (M == 9 && N == 6) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], removed[0], removed[1], removed[2], removed[3], removed[4], removed[5]);
} else if constexpr (M == 10 && N == 5) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], removed[0], removed[1], removed[2], removed[3], removed[4]);
} else if constexpr (M == 11 && N == 4) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], removed[0], removed[1], removed[2], removed[3]);
} else if constexpr (M == 12 && N == 3) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], added[11], removed[0], removed[1], removed[2]);
} else if constexpr (M == 13 && N == 2) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], added[11], added[12], removed[0], removed[1]);
} else if constexpr (M == 14 && N == 1) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Sub>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], added[11], added[12], added[13], removed[0]);
} else if constexpr (M == 15 && N == 0) {
    ctx->template apply_threats<Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add, Add>(added[0], added[1], added[2], added[3], added[4], added[5], added[6], added[7], added[8], added[9], added[10], added[11], added[12], added[13], added[14]);
}
 else { static_assert(false); }

// ---------------------------------
