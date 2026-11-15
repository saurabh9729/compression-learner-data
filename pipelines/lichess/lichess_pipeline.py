"""
Lichess pipeline: does costly computation push players onto stored knowledge?

Input : PGN files from the Lichess open database (database.lichess.org, .pgn or .pgn.zst)
        or per-user API exports (lichess.org/api/games/user/<name>?clocks=true&evals=true).
        Games need [%clk] comments; [%eval] comments are used when present.
Output: one row per move (parquet) with the measures below, plus a per-player summary.

Key measures
- time_class      : bullet / blitz / rapid / classical (Lichess rule: base + 40 * increment).
- seen_before     : how often this player had faced this exact position (same side to move)
                    in earlier games, in any time control.
- habit_move      : the player's most frequent earlier move in this position (their stored answer).
- habit_match     : 1 if they played it now (only defined when seen_before >= MIN_SEEN).
- time_spent      : seconds used on the move (clock before - clock after + increment).
                    Missing for each side's first move: Lichess adds no increment there and
                    tournament 'berserk' halves the clock, so that move's time is not comparable.
- berserk         : 1 if this side started with about half its clock (tournament berserk).
- clock_frac      : remaining clock as a fraction of the starting clock (within-game cost).
- eval_loss_cp    : centipawn loss of the move, from [%eval] before and after (if available).

Privacy (the output repo is public)
- Player names are replaced by anonymous IDs: HMAC-SHA256 of the name with a secret salt
  (env LICHESS_ID_SALT). Without the salt the IDs cannot be turned back into names.
  If the salt is missing, a random one is used and thrown away (IDs then differ between runs).
- No game links (Site), no opponent names, no dates are written. Logs print only totals.

Usage
  python lichess_pipeline.py games.pgn.zst --out moves.parquet --min-games 20
"""
import argparse
import collections
import hashlib
import hmac
import os
import secrets
import io
import math
import re
import sys

import chess
import chess.pgn
import pandas as pd

MIN_SEEN = 3          # earlier visits needed before a position counts as "familiar"
MATE_CP = 1000        # centipawn value used for mate scores

CLK_RE = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
EVAL_RE = re.compile(r"\[%eval (#?-?\d+(?:\.\d+)?)\]")


def open_text(path):
    if path.endswith(".zst"):
        import zstandard
        fh = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
    return open(path, encoding="utf-8")


def time_class(tc):
    """Lichess categories from 'base+inc' in seconds (estimated duration = base + 40 * inc)."""
    if not tc or tc == "-" or "+" not in tc:
        return None, None, None
    base, inc = (int(x) for x in tc.split("+"))
    est = base + 40 * inc
    if est < 180:
        cls = "bullet"
    elif est < 480:
        cls = "blitz"
    elif est < 1500:
        cls = "rapid"
    else:
        cls = "classical"
    return cls, base, inc


def parse_clock(comment):
    m = CLK_RE.search(comment or "")
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def parse_eval(comment):
    """Eval from White's point of view in centipawns; mate mapped to +/- MATE_CP."""
    m = EVAL_RE.search(comment or "")
    if not m:
        return None
    v = m.group(1)
    if v.startswith("#"):
        return MATE_CP if not v.startswith("#-") else -MATE_CP
    return max(-MATE_CP, min(MATE_CP, float(v) * 100))


def iter_games(paths):
    for p in paths:
        with open_text(p) as fh:
            while True:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                yield g


def count_games(paths):
    """First pass: games per player per time class (to select players who play both)."""
    counts = collections.defaultdict(collections.Counter)
    for g in iter_games(paths):
        cls, _, _ = time_class(g.headers.get("TimeControl"))
        if cls is None:
            continue
        for side in ("White", "Black"):
            counts[g.headers.get(side, "?")][cls] += 1
    return counts


def select_players(counts, min_games, classes=("bullet", "classical")):
    return {p for p, c in counts.items() if all(c[k] >= min_games for k in classes)}


def game_order_key(g):
    return (g.headers.get("UTCDate", ""), g.headers.get("UTCTime", ""))


def extract_moves(paths, players):
    """Second pass: one row per move by a selected player, in chronological order per player."""
    games = [g for g in iter_games(paths)
             if g.headers.get("White") in players or g.headers.get("Black") in players]
    games.sort(key=game_order_key)

    memory = collections.defaultdict(collections.Counter)  # (player, epd) -> Counter(move)
    rows = []
    for gi, g in enumerate(games):
        h = g.headers
        cls, base, inc = time_class(h.get("TimeControl"))
        if cls is None:
            continue
        board = g.board()
        prev_clock = {chess.WHITE: None, chess.BLACK: None}  # set from each side's first clock
        berserk = {chess.WHITE: 0, chess.BLACK: 0}
        prev_eval = 20.0  # small white edge at the start
        node = g
        ply = 0
        while node.variations:
            nxt = node.variations[0]
            mover = board.turn
            name = h.get("White") if mover == chess.WHITE else h.get("Black")
            rating = h.get("WhiteElo") if mover == chess.WHITE else h.get("BlackElo")
            epd = board.epd()
            move = nxt.move.uci()
            n_legal = board.legal_moves.count()

            clk = parse_clock(nxt.comment)
            spent = None
            if clk is not None:
                if prev_clock[mover] is None:          # this side's first move
                    berserk[mover] = int(base > 0 and clk <= 0.55 * base)
                else:
                    spent = max(0.0, prev_clock[mover] - clk + (inc or 0))
                prev_clock[mover] = clk
            ev = parse_eval(nxt.comment)
            loss = None
            if ev is not None and prev_eval is not None:
                sign = 1 if mover == chess.WHITE else -1
                loss = max(0.0, sign * (prev_eval - ev))

            if name in players:
                past = memory[(name, epd)]
                seen = sum(past.values())
                habit = past.most_common(1)[0][0] if seen else None
                rows.append(dict(
                    player=name, game=gi, time_class=cls,
                    base=base, inc=inc, rating=pd.to_numeric(rating, errors="coerce"),
                    ply=ply, epd=epd, move=move, n_legal=n_legal,
                    seen_before=seen, habit_move=habit,
                    habit_match=(int(move == habit) if seen >= MIN_SEEN else None),
                    habit_share=(past[habit] / seen if seen else None),
                    time_spent=spent, clock=clk, berserk=berserk[mover],
                    clock_frac=(clk / base if clk is not None and base else None),
                    eval_after=ev, eval_loss_cp=loss,
                ))
                past[move] += 1
            prev_eval = ev if ev is not None else None
            board.push(nxt.move)
            node = nxt
            ply += 1
    return pd.DataFrame(rows)


def player_summary(df):
    fam = df[df.seen_before >= MIN_SEEN]
    out = (fam.groupby(["player", "time_class"])
              .agg(familiar_moves=("habit_match", "size"),
                   habit_match_rate=("habit_match", "mean"),
                   median_time_familiar=("time_spent", "median"))
              .reset_index())
    nov = (df[df.seen_before == 0].groupby(["player", "time_class"])
             .agg(median_time_novel=("time_spent", "median")).reset_index())
    return out.merge(nov, on=["player", "time_class"], how="outer")


def anonymize(names):
    """Map each player name to a stable anonymous ID (secret salt; see Privacy above)."""
    salt = os.environ.get("LICHESS_ID_SALT", "")
    if not salt:
        print("warning: LICHESS_ID_SALT not set; using a one-off random salt", file=sys.stderr)
        salt = secrets.token_hex(32)
    key = salt.encode()
    return {n: "p_" + hmac.new(key, n.encode(), hashlib.sha256).hexdigest()[:12] for n in names}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", nargs="+")
    ap.add_argument("--out", default="moves.parquet")
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--players", help="file with one player name per line (skips selection)")
    a = ap.parse_args(argv)
    if a.players:
        players = {l.strip() for l in open(a.players) if l.strip()}
    else:
        counts = count_games(a.pgn)
        players = select_players(counts, a.min_games)
    print(f"players: {len(players)}", file=sys.stderr)
    df = extract_moves(a.pgn, players)
    df["speed"] = df.time_class.map({"bullet": "fast", "blitz": "medium",
                                     "rapid": "slow", "classical": "slow"})
    ids = anonymize(sorted(players))
    df["player"] = df["player"].map(ids)
    df.to_parquet(a.out, index=False)
    summ = player_summary(df)
    summ.to_csv(a.out.replace(".parquet", "_players.csv"), index=False)
    # Totals only: logs of a public repo are public.
    print(f"moves: {len(df)}, players: {df.player.nunique()}")
    print(summ.groupby("time_class")[["habit_match_rate", "median_time_familiar",
                                      "median_time_novel"]].median().to_string())


if __name__ == "__main__":
    main()
