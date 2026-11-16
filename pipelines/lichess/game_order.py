"""
Order of each player's games ACROSS game types (standard, Chess960, crazyhouse, atomic, antichess),
without dates or names in the output.

The variants run numbered games separately in each game type (column `game` = position of the game
in that type's file, sorted by UTC date and time). This script re-reads only the game headers and
rebuilds the same numbering, so each (player, variant, game) can be placed on one timeline per player.

Step 1, per game type and month (streamed, headers only):
  curl -sL URL | zstdcat | python game_order.py scan --codes codes.txt --type chess960 --out scan_chess960_2018-06.csv
  Keeps exactly the games the variants run kept: a listed player is White or Black (matched by
  anonymous code, using LICHESS_ID_SALT) and the game has clock times. The scan file holds dates and
  stays on the Colab machine; it is never saved to Drive.

Step 2, once all scans exist:
  python game_order.py build --months 2018-06 2018-07 --out game_order.parquet [--check-dir DIR]
  Writes per (player, variant, game): seq (1, 2, 3 ... across all game types), gap_min (minutes since
  the player's previous game of any type), prev_variant, session (new session after a gap > 30 min),
  session_pos, base, inc. No dates, no names. With --check-dir, compares base and inc with the
  variants files (moves_*_{type}_part*.parquet) to confirm the numbering matches.
"""
import argparse
import csv
import glob
import hashlib
import hmac
import os
import sys

TYPES = ["standard", "chess960", "crazyhouse", "atomic", "antichess"]
SESSION_GAP_MIN = 30


def scan(a):
    salt = os.environ.get("LICHESS_ID_SALT", "")
    assert salt, "LICHESS_ID_SALT is not set"
    key = salt.encode()
    codes = {l.strip() for l in open(a.codes) if l.strip()}
    cache = {}

    def code(name):
        c = cache.get(name)
        if c is None:
            c = cache[name] = "p_" + hmac.new(key, name.encode(), hashlib.sha256).hexdigest()[:12]
        return c

    n_read = n_kept = 0
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "date", "time", "white", "black", "tc"])
        heads, in_moves, clk = {}, False, False

        def finish():
            nonlocal n_read, n_kept
            n_read += 1
            if n_read % 1_000_000 == 0:
                print(f"{n_read:,} games read", file=sys.stderr, flush=True)
            if not heads:
                return
            wc, bc = code(heads.get("White", "?")), code(heads.get("Black", "?"))
            if (wc in codes or bc in codes) and clk:
                w.writerow([n_kept, heads.get("UTCDate", ""), heads.get("UTCTime", ""),
                            wc if wc in codes else "", bc if bc in codes else "", heads.get("TimeControl", "")])
                n_kept += 1

        for line in sys.stdin:
            if line.startswith("[") and in_moves:
                finish()
                heads, in_moves, clk = {}, False, False
            if line.startswith("["):
                k, _, v = line[1:].partition(" ")
                heads[k] = v.rstrip().rstrip("]").strip('"')
            elif line.strip():
                in_moves = True
                if not clk and "[%clk" in line:
                    clk = True
        if heads:
            finish()
    print(f"games read: {n_read:,}; games kept: {n_kept:,}", file=sys.stderr, flush=True)


def build(a):
    import pandas as pd
    import numpy as np
    parts = []
    for t in TYPES:
        frames = []
        for m in a.months:
            f = f"scan_{t}_{m}.csv"
            if os.path.exists(f):
                frames.append(pd.read_csv(f, dtype=str, keep_default_na=False))
        if not frames:
            print(f"{t}: no scan files, skipped")
            continue
        d = pd.concat(frames, ignore_index=True)          # stream order: month after month
        d["key"] = d.date + " " + d.time
        d = d.iloc[np.argsort(d.key.values, kind="stable")].reset_index(drop=True)   # same as sorted(..., key=(date, time))
        d["game"] = np.arange(len(d))
        d["variant"] = t
        print(f"{t:11s} games kept: {len(d):,}")
        parts.append(d)
    g = pd.concat(parts, ignore_index=True)
    long = pd.concat([g.assign(player=g.white), g.assign(player=g.black)])
    long = long[long.player != ""].copy()
    long["dt"] = pd.to_datetime(long.date + " " + long.time, format="%Y.%m.%d %H:%M:%S", errors="coerce")
    tc = long.tc.str.split("+", expand=True)
    long["base"] = pd.to_numeric(tc[0], errors="coerce")
    long["inc"] = pd.to_numeric(tc[1], errors="coerce")
    long = long.sort_values(["player", "dt", "variant", "game"], kind="stable")
    long["seq"] = long.groupby("player").cumcount() + 1
    long["gap_min"] = long.groupby("player").dt.diff().dt.total_seconds() / 60
    long["prev_variant"] = long.groupby("player").variant.shift(1)
    long["session"] = (long.gap_min.isna() | (long.gap_min > SESSION_GAP_MIN)).astype(int).groupby(long.player).cumsum()
    long["session_pos"] = long.groupby(["player", "session"]).cumcount() + 1
    out = long[["player", "variant", "game", "seq", "gap_min", "prev_variant", "session", "session_pos", "base", "inc"]]
    out = out.reset_index(drop=True)
    out["gap_min"] = out.gap_min.round(1)
    out.to_parquet(a.out, index=False)
    print(f"rows: {len(out):,}; players: {out.player.nunique()}; written {a.out} (no dates, no names)")
    print("games per player by type (median):", out.groupby(["variant", "player"]).size().groupby("variant").median().to_dict())

    if a.check_dir:   # does our numbering match the variants files? compare time control per (player, game)
        for t in TYPES:
            fs = sorted(glob.glob(os.path.join(a.check_dir, f"moves_*_{t}_part*.parquet")))
            fs = [f for f in fs if "_test_" not in f]
            if not fs:
                continue
            m = pd.concat([pd.read_parquet(f, columns=["player", "game", "base", "inc"]).drop_duplicates(["player", "game"])
                           for f in fs], ignore_index=True)
            j = m.merge(out[out.variant == t], on=["player", "game"], how="left", suffixes=("", "_o"))
            found = j.seq.notna().mean()
            agree = ((j.base == j.base_o) & (j.inc == j.inc_o)).mean()
            print(f"check {t:11s}: {len(m):,} (player, game) pairs; found {found*100:.2f}%; time control agrees {agree*100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--codes", required=True)
    s.add_argument("--type", required=True)
    s.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--months", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--check-dir")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else build(a)


if __name__ == "__main__":
    main()
