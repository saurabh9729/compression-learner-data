"""Per-move store features for the developmental four-in-a-row data.

Board: 4 rows x 9 columns, squares 0..35 row by row; any empty square may be played (no gravity).
For each person, in time order over non-practice games, six stores remember earlier choices:
  exact (whole board), reply (opponent's last square), relational (offset from opponent's last square),
  chunk (my previous square), local (3x3 pattern around opponent's last square), default (favourite squares).
A store has a usual move after >= MIN_SEEN visits; the usual move must be playable (empty) now.
Clear errors: missed win (a winning square existed, not taken) and missed block (exactly one square
stops the opponent's four, I had no win, and did not take it).
"""
import collections
import math

import numpy as np
import pandas as pd

ROWS, COLS, MIN_SEEN = 4, 9, 3
import os
DATA = os.environ.get("FIAR_DIR", "/home/claude/fiar_dev")


def lines_of_four():
    L = []
    for r in range(ROWS):
        for c in range(COLS - 3):
            L.append([r * COLS + c + k for k in range(4)])
    for c in range(COLS):
        L.append([k * COLS + c for k in range(4)])
    for c in range(COLS - 3):
        L.append([k * COLS + c + k for k in range(4)])
    for c in range(3, COLS):
        L.append([k * COLS + c - k for k in range(4)])
    return L


LINES = lines_of_four()
LINES_AT = collections.defaultdict(list)
for ln in LINES:
    for s in ln:
        LINES_AT[s].append(ln)


def winning_squares(mine, empty):
    """Empty squares that complete four of `mine`."""
    out = set()
    for ln in LINES:
        own = sum(s in mine for s in ln)
        emp = [s for s in ln if s in empty]
        if own == 3 and len(emp) == 1:
            out.add(emp[0])
    return out


def offset(a, b):
    """Offset of square b from square a as (drow, dcol)."""
    return (b // COLS - a // COLS, b % COLS - a % COLS)


def apply_offset(a, off):
    r, c = a // COLS + off[0], a % COLS + off[1]
    return r * COLS + c if 0 <= r < ROWS and 0 <= c < COLS else None


def local_key(center, mine, theirs):
    cells = []
    r0, c0 = center // COLS, center % COLS
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < ROWS and 0 <= c < COLS):
                cells.append("x")
            else:
                s = r * COLS + c
                cells.append("m" if s in mine else "o" if s in theirs else ".")
    return "".join(cells)


def build():
    m = pd.read_parquet(f"{DATA}/moves.parquet")
    p = pd.read_csv(f"{DATA}/people.csv", dtype={"sid": str})
    m = m[(m.practice == False) & m.sid.isin(p.sid[p.age.notna()])].copy()
    m = m.sort_values(["sid", "game", "move_in_game"]).reset_index(drop=True)
    stores = ["exact", "reply", "relational", "chunk", "local", "default"]
    out, bad = [], [0]
    for sid, g in m.groupby("sid", sort=False):
        mem = {s: collections.defaultdict(collections.Counter) for s in stores}
        game_rank = {gn: i for i, gn in enumerate(sorted(g.game.unique()))}
        prev_opp, prev_mine_tile, cur_game = set(), None, None
        for r in g.itertuples(index=False):
            if r.game != cur_game:
                cur_game, prev_opp, prev_mine_tile = r.game, set(), None
            bp = {i for i, ch in enumerate(r.bp) if ch == "1"}
            wp = {i for i, ch in enumerate(r.wp) if ch == "1"}
            mine, theirs = (bp, wp) if r.color == "black" else (wp, bp)
            empty = set(range(ROWS * COLS)) - mine - theirs
            new_opp = theirs - prev_opp
            opp_last = next(iter(new_opp)) if len(new_opp) == 1 else None
            tile = int(r.tile)
            if tile not in empty:
                bad[0] += 1
            win_sq = winning_squares(mine, empty)
            threat_sq = winning_squares(theirs, empty)
            critical = "win" if win_sq else ("block" if len(threat_sq) == 1 else None)
            correct = win_sq if win_sq else (threat_sq if len(threat_sq) == 1 else set())
            row = dict(sid=sid, game=r.game, game_rank=game_rank[r.game], level=r.level, color=r.color,
                       move_in_game=r.move_in_game, tile=tile, rt_ms=r.rt_ms, result=getattr(r, "result", None),
                       n_empty=len(empty), critical=critical, correct_move=(tile in correct) if critical else np.nan,
                       n_threats=len(threat_sq))
            keys = {
                "exact": (r.color, r.bp, r.wp),
                "reply": (r.color, opp_last) if opp_last is not None else None,
                "relational": r.color if opp_last is not None else None,
                "chunk": (r.color, prev_mine_tile) if prev_mine_tile is not None else None,
                "local": (r.color, local_key(opp_last, mine, theirs)) if opp_last is not None else None,
                "default": r.color,
            }
            usual_set = set()
            for s in stores:
                k = keys[s]
                row[f"{s}_seen"], row[f"{s}_share"], row[f"{s}_match"], row[f"{s}_usual"] = 0, np.nan, np.nan, np.nan
                if k is None:
                    continue
                past = mem[s][k]
                seen = sum(past.values())
                row[f"{s}_seen"] = seen
                usual = None
                for val, cnt in past.most_common():
                    sq = apply_offset(opp_last, val) if s in ("relational", "local") else val
                    if sq is not None and sq in empty:
                        usual = sq
                        row[f"{s}_share"] = cnt / seen
                        break
                if usual is not None and seen >= MIN_SEEN:
                    row[f"{s}_usual"] = usual
                    row[f"{s}_match"] = float(tile == usual)
                    usual_set.add(usual)
                    if critical:
                        row[f"{s}_usual_ok"] = float(usual in correct)
                if s == "default":   # log prob of the played square under the default policy (add-one over empty squares)
                    denom = sum(past[q] for q in empty) + len(empty)
                    row["default_logp"] = math.log((past[tile] + 1) / denom)
            row["n_usual"] = len(usual_set)
            row["any_match"] = float(tile in usual_set) if usual_set else np.nan
            if critical:
                wrong = usual_set - correct
                row["n_wrong_usual"] = len(wrong)
                row["played_wrong_usual"] = float(tile in wrong)
                n_wrong_sq = len(empty) - len(correct)
                row["chance_wrong_usual"] = len(wrong) / n_wrong_sq if n_wrong_sq else np.nan
            # update memories
            for s in stores:
                k = keys[s]
                if k is None:
                    continue
                val = offset(opp_last, tile) if s in ("relational", "local") else tile
                mem[s][k][val] += 1
            out.append(row)
            prev_opp = set(theirs)
            prev_mine_tile = tile
    print('moves onto an occupied square (board-before check):', bad[0])
    d = pd.DataFrame(out)
    # each person's speed relative to their own usual speed
    d["log_rt"] = np.log(d.rt_ms.clip(lower=100))
    d["rt_z"] = d.log_rt - d.groupby("sid").log_rt.transform("median")
    d["move_bin"] = d.move_in_game.clip(upper=10)
    d.to_parquet(f"{DATA}/store_features.parquet", index=False)
    return d


if __name__ == "__main__":
    d = build()
    print(len(d), "moves;", d.sid.nunique(), "people")
    for s in ["exact", "reply", "relational", "chunk", "local", "default"]:
        c = d[f"{s}_match"]
        print(f"{s:10s} has usual move {c.notna().mean()*100:5.1f}% | played it {c.mean()*100:5.1f}% | mean consistency {d.loc[c.notna(), f'{s}_share'].mean():.2f}")
    print("critical positions:", d.critical.value_counts().to_dict(), "| correct:", d.groupby("critical").correct_move.mean().round(3).to_dict())
