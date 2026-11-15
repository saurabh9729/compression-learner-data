"""Engine check of reused answers in similar-but-not-identical positions.

For a sample of moves in positions that are new to the player exactly but of a familiar kind,
Stockfish evaluates:
  - the best move,
  - the move the player actually played,
  - the player's usual move under each stored representation (kind = pawns + material, pawns
    only, own setup, reply to the opponent's last move, own two-move chunk), even when the
    player did NOT play it.
This shows where a stored answer would have been wrong: the summary's "exceptions".
It also records the best move of a 1-ply search ("shallow look"), so moves can be classed as
matching a quick look versus only a deep calculation.

Losses are in centipawns from the mover's point of view (best minus candidate, >= 0, mates
capped at 1000). Output: one row per sampled move with player/game/ply keys for merging.

Usage: python engine_eval.py moves.parquet [more parts...] --out engine.parquet --n 20000 --depth 12
"""
import argparse
import shutil
import sys

import chess
import chess.engine
import pandas as pd

MATE_CP = 1000
HABITS = {"kind": "struct_habit_move", "pawn": "pawn_habit", "own": "own_habit",
          "reply": "reply_habit", "chunk": "chunk_habit", "cond": "cond_habit",
          "global": "global_habit"}


def score_cp(info, turn):
    return max(-MATE_CP, min(MATE_CP, info["score"].pov(turn).score(mate_score=MATE_CP)))


def pattern_move(board, pattern):
    """Stored pattern 'Nb1c3' -> legal Move, or None."""
    if not isinstance(pattern, str) or len(pattern) < 5:
        return None
    try:
        mv = chess.Move.from_uci(pattern[1:])
    except ValueError:
        return None
    return mv if mv in board.legal_moves else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="+")
    ap.add_argument("--out", default="engine.parquet")
    ap.add_argument("--n", type=int, default=20000, help="moves to evaluate (half fast, half slow games)")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)

    cols = ["player", "game", "ply", "time_class", "move", "epd", "seen_before", "struct_seen"] + list(HABITS.values())
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in a.parquet], ignore_index=True)
    d = d[d.epd.notna() & (d.seen_before == 0) & (d.struct_seen >= 3) & (d.ply >= 2)]
    d["fast"] = d.time_class.isin(["bullet", "blitz"])
    half = a.n // 2
    sample = pd.concat([g.sample(min(half, len(g)), random_state=a.seed) for _, g in d.groupby("fast")])
    print(f"similar-position moves available: {len(d):,}; evaluating {len(sample):,}", file=sys.stderr)

    path = shutil.which("stockfish") or "/usr/games/stockfish"
    eng = chess.engine.SimpleEngine.popen_uci(path)
    eng.configure({"Threads": 2, "Hash": 256})
    limit = chess.engine.Limit(depth=a.depth)
    out = []
    for i, r in enumerate(sample.itertuples(index=False), 1):
        board = chess.Board()
        board.set_epd(r.epd)
        best = eng.analyse(board, limit)
        best_cp = score_cp(best, board.turn)
        best_mv = best["pv"][0] if best.get("pv") else None
        cache = {best_mv: best_cp} if best_mv else {}

        def loss_of(mv):
            if mv is None:
                return None
            if mv not in cache:
                cache[mv] = score_cp(eng.analyse(board, limit, root_moves=[mv]), board.turn)
            return max(0, best_cp - cache[mv])

        played = chess.Move.from_uci(r.move)
        shallow = eng.analyse(board, chess.engine.Limit(depth=1))
        shallow_mv = shallow["pv"][0] if shallow.get("pv") else None
        row = dict(player=r.player, game=r.game, ply=r.ply, best_move=best_mv.uci() if best_mv else None,
                   best_cp=best_cp, loss_played=loss_of(played),
                   shallow_move=shallow_mv.uci() if shallow_mv else None, loss_shallow=loss_of(shallow_mv))
        for k, col in HABITS.items():
            row[f"loss_{k}_habit"] = loss_of(pattern_move(board, getattr(r, col)))
        out.append(row)
        if i % 1000 == 0:
            print(f"{i:,} evaluated", file=sys.stderr, flush=True)
    eng.quit()
    pd.DataFrame(out).to_parquet(a.out, index=False)
    print(f"engine rows: {len(out)}")


if __name__ == "__main__":
    main()
