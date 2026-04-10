#!/usr/bin/env python3
"""
Parse binary movepick dump files produced by Stockfish's MovepickSampler.

Binary format per sample:
  32 bytes: packed position (2 squares per byte, 4-bit nibbles, low=even sq)
  1 byte:   side to move (0=white, 1=black)
  1 byte:   ep square (0-63, or 64=none)
  4 bytes:  castling rights (uint8 each: WHITE_OO, WHITE_OOO, BLACK_OO, BLACK_OOO)
  1 byte:   number of moves N
  N * 2 bytes: move raw values (uint16 LE)
  N * 22 bytes: per-move features (11 x int16 LE each):
      mainHistory, pawnHistory,
      contHist0, contHist1, contHist2, contHist3, contHist5,
      goodCheck, threatValue, lowPlyHistory, ply
"""

import json
import queue
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import chess

# PieceValue indexed by python-chess piece type (PAWN=1..KING=6)
PIECE_VALUE = [0, 208, 781, 825, 1276, 2538, 0]

# Stockfish Piece enum -> python-chess piece
# W_PAWN=1..W_KING=6, B_PAWN=9..B_KING=14
SF_PIECE_TO_CHESS = {
    0: None,
    1: chess.Piece(chess.PAWN, chess.WHITE),
    2: chess.Piece(chess.KNIGHT, chess.WHITE),
    3: chess.Piece(chess.BISHOP, chess.WHITE),
    4: chess.Piece(chess.ROOK, chess.WHITE),
    5: chess.Piece(chess.QUEEN, chess.WHITE),
    6: chess.Piece(chess.KING, chess.WHITE),
    9: chess.Piece(chess.PAWN, chess.BLACK),
    10: chess.Piece(chess.KNIGHT, chess.BLACK),
    11: chess.Piece(chess.BISHOP, chess.BLACK),
    12: chess.Piece(chess.ROOK, chess.BLACK),
    13: chess.Piece(chess.QUEEN, chess.BLACK),
    14: chess.Piece(chess.KING, chess.BLACK),
}

FEATURE_NAMES = [
    "mainHistory",
    "pawnHistory",
    "contHist0",
    "contHist1",
    "contHist2",
    "contHist3",
    "contHist5",
    "goodCheck",
    "threatValue",
    "lowPlyHistory",
    "ply",
]


def decode_move(raw: int) -> chess.Move:
    """Decode a Stockfish raw move (uint16) into a python-chess Move."""
    to_sq = raw & 0x3F
    from_sq = (raw >> 6) & 0x3F
    flag = raw & (3 << 14)

    PROMOTION = 1 << 14
    EN_PASSANT = 2 << 14
    CASTLING = 3 << 14

    if flag == PROMOTION:
        promo_pt = ((raw >> 12) & 3) + chess.KNIGHT  # KNIGHT=2 in python-chess
        return chess.Move(from_sq, to_sq, promotion=promo_pt)
    elif flag == EN_PASSANT:
        return chess.Move(from_sq, to_sq)
    elif flag == CASTLING:
        # Stockfish encodes castling as king-to-rook; python-chess wants king-to-dest
        # from_sq is king square, to_sq is rook square
        rank = from_sq // 8
        if to_sq > from_sq:  # kingside
            dest = chess.square(6, rank)  # g1 or g8
        else:  # queenside
            dest = chess.square(2, rank)  # c1 or c8
        return chess.Move(from_sq, dest)
    else:
        return chess.Move(from_sq, to_sq)


@dataclass
class MoveFeatures:
    move: chess.Move
    raw_move: int
    mainHistory: int = 0
    pawnHistory: int = 0
    contHist0: int = 0
    contHist1: int = 0
    contHist2: int = 0
    contHist3: int = 0
    contHist5: int = 0
    goodCheck: int = 0
    threatValue: int = 0
    lowPlyHistory: int = 0
    ply: int = 0


@dataclass
class Sample:
    board: chess.Board
    moves: list[MoveFeatures] = field(default_factory=list)


def parse_dump(filepath: str) -> list[Sample]:
    with open(filepath, "rb") as f:
        data = f.read()

    samples = []
    pos = 0

    while pos < len(data):
        # Need at least the header: 32 + 1 + 1 + 4 + 1 = 39 bytes
        if pos + 39 > len(data):
            break

        # --- Decode position ---
        packed = data[pos : pos + 32]
        pos += 32

        board = chess.Board.empty()

        for sq in range(64):
            byte_idx = sq // 2
            nibble = (packed[byte_idx] >> ((sq % 2) * 4)) & 0xF
            piece = SF_PIECE_TO_CHESS.get(nibble)
            if piece is not None:
                board.set_piece_at(sq, piece)

        stm = data[pos]
        pos += 1
        board.turn = chess.WHITE if stm == 0 else chess.BLACK

        ep_sq = data[pos]
        pos += 1
        board.ep_square = ep_sq if ep_sq < 64 else None

        castling_bytes = data[pos : pos + 4]
        pos += 4
        castling = 0
        if castling_bytes[0]:  # WHITE_OO
            castling |= chess.BB_H1
        if castling_bytes[1]:  # WHITE_OOO
            castling |= chess.BB_A1
        if castling_bytes[2]:  # BLACK_OO
            castling |= chess.BB_H8
        if castling_bytes[3]:  # BLACK_OOO
            castling |= chess.BB_A8
        board.castling_rights = castling

        num_moves = data[pos]
        pos += 1

        # --- Read raw moves ---
        moves_size = num_moves * 2
        if pos + moves_size > len(data):
            break
        raw_moves = struct.unpack_from(f"<{num_moves}H", data, pos)
        pos += moves_size

        # --- Read per-move features ---
        features_size = num_moves * 22  # 11 x int16
        if pos + features_size > len(data):
            break

        move_features = []
        for i in range(num_moves):
            feats = struct.unpack_from("<11h", data, pos)
            pos += 22

            mf = MoveFeatures(
                move=decode_move(raw_moves[i]),
                raw_move=raw_moves[i],
            )
            for name, val in zip(FEATURE_NAMES, feats):
                setattr(mf, name, val)
            mf.goodCheck *= 16384
            pt = board.piece_type_at(mf.move.from_square)
            mf.threatValue *= PIECE_VALUE[pt] if pt else 0
            move_features.append(mf)

        sample = Sample(board=board, moves=move_features)
        samples.append(sample)

    return samples


STOCKFISH_PATH = "bins/stockfish.master.gcc15"


class StockfishEngine:
    def __init__(self, path: str = STOCKFISH_PATH, multipv: int = 5):
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name MultiPV value {multipv}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token: str) -> list[str]:
        lines = []
        while True:
            line = self.proc.stdout.readline().strip()
            lines.append(line)
            if line.startswith(token):
                break
        return lines

    def evaluate(self, fen: str, depth: int = 12) -> list[dict]:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        lines = self._read_until("bestmove")

        # Parse the final depth info lines (one per multipv)
        results = {}
        for line in lines:
            if not line.startswith("info depth"):
                continue
            parts = line.split()
            try:
                d = int(parts[parts.index("depth") + 1])
            except (ValueError, IndexError):
                continue
            if d != depth:
                continue
            try:
                mpv = int(parts[parts.index("multipv") + 1])
            except (ValueError, IndexError):
                continue

            # Parse score
            score_idx = parts.index("score")
            score_type = parts[score_idx + 1]
            score_val = int(parts[score_idx + 2])
            if score_type == "cp":
                score = {"type": "cp", "value": score_val}
            else:  # mate
                score = {"type": "mate", "value": score_val}

            # Parse move (first move of PV)
            pv_idx = parts.index("pv")
            move = parts[pv_idx + 1]

            results[mpv] = {"rank": mpv, "score": score, "move": move}

        return [results[k] for k in sorted(results)]

    def close(self):
        self._send("quit")
        self.proc.wait()


def run_evals(dump_path: str, depth: int = 12, engine_path: str = STOCKFISH_PATH,
              limit: int | None = None, concurrency: int = 1):
    samples = parse_dump(dump_path)
    if limit is not None:
        samples = samples[:limit]
    total = len(samples)
    print(f"Evaluating {total} samples at depth {depth} with {concurrency} engine(s)...")

    engine_pool = queue.Queue()
    for _ in range(concurrency):
        engine_pool.put(StockfishEngine(engine_path))

    out_path = dump_path + ".evals.json"
    results = [None] * total
    done_count = 0
    lock = threading.Lock()

    def eval_sample(i: int, s: Sample):
        nonlocal done_count
        engine = engine_pool.get()
        try:
            fen = s.board.fen()
            evals = engine.evaluate(fen, depth)
            results[i] = {"sample": i, "fen": fen, "evals": evals}
        finally:
            engine_pool.put(engine)
        with lock:
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                print(f"  {done_count}/{total}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for i, s in enumerate(samples):
            pool.submit(eval_sample, i, s)

    while not engine_pool.empty():
        engine_pool.get().close()

    with open(out_path, "w") as f:
        json.dump(results, f)

    print(f"Wrote {out_path}")


def print_dump(dump_path: str):
    samples = parse_dump(dump_path)
    print(f"Parsed {len(samples)} samples\n")

    for i, s in enumerate(samples):
        print(f"=== Sample {i} ===")
        print(s.board)
        print(f"FEN: {s.board.fen()}")
        print(f"Moves ({len(s.moves)}):")
        for mf in s.moves:
            print(
                f"  {s.board.san(mf.move):8s} "
                f"main={mf.mainHistory:6d}  pawn={mf.pawnHistory:6d}  "
                f"ch0={mf.contHist0:6d}  ch1={mf.contHist1:6d}  "
                f"ch2={mf.contHist2:6d}  ch3={mf.contHist3:6d}  "
                f"ch5={mf.contHist5:6d}  "
                f"chk={mf.goodCheck}  threat={mf.threatValue:4d}  "
                f"lowPly={mf.lowPlyHistory:6d}  ply={mf.ply}"
            )
        print()


def score_to_cp(score: dict) -> int | None:
    """Convert an oracle score dict to centipawns. Returns None for mate scores."""
    if score["type"] == "cp":
        return score["value"]
    return None


# Features used for fitting (excludes ply which is context, not a sortable signal)
FIT_FEATURE_NAMES = [
    "mainHistory",
    "pawnHistory",
    "contHist0",
    "contHist1",
    "contHist2",
    "contHist3",
    "contHist5",
    "goodCheck",
    "threatValue",
    "lowPlyHistory",
]


def _load_fit_data(dump_path, depth, engine_path, concurrency, limit):
    """Load samples and evals, filtering unusable positions. Returns (samples, evals, filtered indices)."""
    evals_path = dump_path + ".evals.json"
    try:
        with open(evals_path) as f:
            evals_data = json.load(f)
        print(f"Loaded existing evals from {evals_path}")
    except FileNotFoundError:
        print(f"Evals not found, running eval first...")
        run_evals(dump_path, depth=depth, engine_path=engine_path,
                  concurrency=concurrency, limit=limit)
        with open(evals_path) as f:
            evals_data = json.load(f)

    samples = parse_dump(dump_path)
    if limit is not None:
        samples = samples[:limit]

    skipped_mate = 0
    skipped_non_quiet = 0
    filtered = []  # list of (sample, oracle_evals, oracle_cp_map)

    for sample, ev in zip(samples, evals_data):
        oracle_evals = ev["evals"]
        if not oracle_evals:
            skipped_mate += 1
            continue
        cp_values = [score_to_cp(e["score"]) for e in oracle_evals]
        if any(v is None for v in cp_values):
            skipped_mate += 1
            continue
        quiet_ucis = {mf.move.uci() for mf in sample.moves}
        if oracle_evals[0]["move"] not in quiet_ucis:
            skipped_non_quiet += 1
            continue
        oracle_cp = {e["move"]: score_to_cp(e["score"]) for e in oracle_evals}
        filtered.append((sample, oracle_evals, oracle_cp))

    print(f"  {len(filtered)} usable positions "
          f"(skipped: {skipped_mate} mate, {skipped_non_quiet} non-quiet best move)")
    return filtered


def _get_features(mf: MoveFeatures) -> list[float]:
    return [getattr(mf, name) for name in FIT_FEATURE_NAMES]


def _rank_metrics(filtered, score_fn):
    """Compute ranking quality metrics given a scoring function score_fn(MoveFeatures) -> float."""
    import numpy as np
    top1_match = 0
    top3_overlap = 0
    total = 0

    for sample, oracle_evals, _ in filtered:
        if not sample.moves:
            continue
        scores = [score_fn(mf) for mf in sample.moves]
        ranked_uci = [sample.moves[j].move.uci()
                      for j in np.argsort([-s for s in scores])]
        oracle_ranked = [e["move"] for e in oracle_evals]

        if ranked_uci[0] == oracle_ranked[0]:
            top1_match += 1
        model_top3 = set(ranked_uci[:3])
        oracle_top3 = set(oracle_ranked[:3])
        top3_overlap += len(model_top3 & oracle_top3)
        total += 1

    return top1_match, top3_overlap, total


def _print_metrics(top1_match, top3_overlap, total):
    if total > 0:
        print(f"  Top-1 accuracy: {top1_match}/{total} "
              f"({100 * top1_match / total:.1f}%)")
        print(f"  Top-3 overlap:  {top3_overlap}/{total * 3} "
              f"({100 * top3_overlap / (total * 3):.1f}%)")


def run_fit(dump_path: str, depth: int = 12, engine_path: str = STOCKFISH_PATH,
            concurrency: int = 1, limit: int | None = None):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    filtered = _load_fit_data(dump_path, depth, engine_path, concurrency, limit)

    # === Pointwise regression with standardized features ===
    print("\n--- Pointwise regression (standardized features) ---")

    X_rows = []
    y_rows = []
    for sample, oracle_evals, oracle_cp in filtered:
        cp_values = list(oracle_cp.values())
        worst_oracle = min(cp_values)
        default_target = worst_oracle - 100

        features = []
        targets = []
        for mf in sample.moves:
            uci = mf.move.uci()
            target = oracle_cp.get(uci, default_target)
            features.append(_get_features(mf))
            targets.append(target)

        if not features:
            continue

        mean_target = sum(targets) / len(targets)
        for feat, target in zip(features, targets):
            X_rows.append(feat)
            y_rows.append(target - mean_target)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.float64)
    print(f"  {len(X)} training rows")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_pt = Ridge(alpha=1.0, fit_intercept=False)
    model_pt.fit(X_scaled, y)

    print(f"  R² = {model_pt.score(X_scaled, y):.4f}")
    print(f"\n  Standardized weights:")
    max_name_len = max(len(n) for n in FIT_FEATURE_NAMES)
    for name, w in zip(FIT_FEATURE_NAMES, model_pt.coef_):
        print(f"    {name:{max_name_len}s}  {w:+.4f}")

    # Raw-space weights: w_raw = w_scaled / std
    raw_coef_pt = model_pt.coef_ / scaler.scale_
    print(f"\n  Raw-space weights:")
    for name, w in zip(FIT_FEATURE_NAMES, raw_coef_pt):
        print(f"    {name:{max_name_len}s}  {w:+.6f}")

    print(f"\n  Ranking quality (pointwise):")
    def score_pt(mf):
        feat = np.array(_get_features(mf)).reshape(1, -1)
        return model_pt.predict(scaler.transform(feat))[0]
    _print_metrics(*_rank_metrics(filtered, score_pt))

    # === Pairwise ranking ===
    print("\n--- Pairwise ranking (standardized features) ---")

    X_pair = []
    y_pair = []
    rng = np.random.default_rng(42)

    for sample, oracle_evals, oracle_cp in filtered:
        # Build feature map for this position
        move_feats = {}
        for mf in sample.moves:
            move_feats[mf.move.uci()] = np.array(_get_features(mf))

        oracle_moves = [e["move"] for e in oracle_evals if e["move"] in move_feats]

        # Pairs: oracle-ranked move vs lower-ranked or unranked moves
        oracle_set = set(oracle_moves)
        unranked = [uci for uci in move_feats if uci not in oracle_set]

        # Between oracle moves (higher rank beats lower rank), both directions
        for i in range(len(oracle_moves)):
            for j in range(i + 1, len(oracle_moves)):
                diff = move_feats[oracle_moves[i]] - move_feats[oracle_moves[j]]
                X_pair.append(diff)
                y_pair.append(1.0)
                X_pair.append(-diff)
                y_pair.append(-1.0)

        # Oracle moves vs a sample of unranked moves, both directions
        n_neg = min(len(unranked), 3)
        for om in oracle_moves:
            for um in rng.choice(unranked, size=n_neg, replace=False) if unranked else []:
                diff = move_feats[om] - move_feats[um]
                X_pair.append(diff)
                y_pair.append(1.0)
                X_pair.append(-diff)
                y_pair.append(-1.0)

    X_pair = np.array(X_pair, dtype=np.float64)
    y_pair = np.array(y_pair, dtype=np.float64)
    print(f"  {len(X_pair)} training pairs")

    scaler_pw = StandardScaler()
    X_pair_scaled = scaler_pw.fit_transform(X_pair)

    model_pw = Ridge(alpha=1.0, fit_intercept=False)
    model_pw.fit(X_pair_scaled, y_pair)

    print(f"  R² = {model_pw.score(X_pair_scaled, y_pair):.4f}")
    print(f"\n  Standardized weights:")
    for name, w in zip(FIT_FEATURE_NAMES, model_pw.coef_):
        print(f"    {name:{max_name_len}s}  {w:+.4f}")

    raw_coef_pw = model_pw.coef_ / scaler_pw.scale_
    print(f"\n  Raw-space weights:")
    for name, w in zip(FIT_FEATURE_NAMES, raw_coef_pw):
        print(f"    {name:{max_name_len}s}  {w:+.6f}")

    print(f"\n  Ranking quality (pairwise):")
    def score_pw(mf):
        feat = np.array(_get_features(mf))
        return feat.dot(raw_coef_pw)
    _print_metrics(*_rank_metrics(filtered, score_pw))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse and analyze movepick dump files")
    parser.add_argument("dump_file", help="Path to the binary dump file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("print", help="Print decoded dump contents")

    eval_parser = subparsers.add_parser("eval", help="Run Stockfish MultiPV evals on each position")
    eval_parser.add_argument("--depth", type=int, default=12, help="Search depth (default: 12)")
    eval_parser.add_argument("--engine", default=STOCKFISH_PATH, help=f"Stockfish binary path (default: {STOCKFISH_PATH})")
    eval_parser.add_argument("-n", "--limit", type=int, default=None, help="Max number of positions to evaluate")
    eval_parser.add_argument("-j", "--concurrency", type=int, default=1, help="Number of Stockfish engines to run in parallel")

    fit_parser = subparsers.add_parser("fit", help="Fit linear weights for move ordering features")
    fit_parser.add_argument("--depth", type=int, default=12, help="Search depth for evals if needed (default: 12)")
    fit_parser.add_argument("--engine", default=STOCKFISH_PATH, help=f"Stockfish binary path (default: {STOCKFISH_PATH})")
    fit_parser.add_argument("-n", "--limit", type=int, default=None, help="Max number of positions to use")
    fit_parser.add_argument("-j", "--concurrency", type=int, default=1, help="Number of Stockfish engines for eval")

    args = parser.parse_args()

    if args.command == "print":
        print_dump(args.dump_file)
    elif args.command == "eval":
        run_evals(args.dump_file, depth=args.depth, engine_path=args.engine,
                  limit=args.limit, concurrency=args.concurrency)
    elif args.command == "fit":
        run_fit(args.dump_file, depth=args.depth, engine_path=args.engine,
                concurrency=args.concurrency, limit=args.limit)
