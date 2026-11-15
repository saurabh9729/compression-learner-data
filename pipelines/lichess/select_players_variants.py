"""Pick players active in standard chess AND in at least one variant (same month); sample N of them.

Inputs are count files from `filter_lichess.py --count` (one for standard, one per variant).
Prints only counts (logs may be public).

Usage:
  python select_players_variants.py --standard counts_standard.csv \
      --variant chess960=counts_chess960.csv --variant crazyhouse=counts_crazyhouse.csv \
      --min-standard 50 --min-variant 30 --n 400 --out players.txt
"""
import argparse
import random

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--standard", required=True)
ap.add_argument("--variant", action="append", default=[], help="name=counts.csv (repeatable)")
ap.add_argument("--min-standard", type=int, default=50, help="standard games (all speeds) in the month")
ap.add_argument("--min-variant", type=int, default=30, help="games in a variant (all speeds) in the month")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="players.txt")
a = ap.parse_args()

SPEEDS = ["bullet", "blitz", "rapid", "classical"]


def totals(path):
    c = pd.read_csv(path)
    return c.set_index("player")[SPEEDS].sum(axis=1)


tab = pd.DataFrame({"standard": totals(a.standard)})
for spec in a.variant:
    name, path = spec.split("=", 1)
    tab = tab.join(totals(path).rename(name), how="outer")
tab = tab.fillna(0)
vnames = [s.split("=", 1)[0] for s in a.variant]
ok_std = tab.standard >= a.min_standard
ok_var = (tab[vnames] >= a.min_variant)
eligible = tab[ok_std & ok_var.any(axis=1)]
print(f"players with >= {a.min_standard} standard games: {int(ok_std.sum()):,}")
for v in vnames:
    print(f"  ... and >= {a.min_variant} {v} games: {int((ok_std & ok_var[v]).sum()):,}")
print(f"  ... and two or more variants: {int((ok_std & (ok_var.sum(axis=1) >= 2)).sum()):,}")
print(f"eligible (standard + at least one variant): {len(eligible):,}")

players = sorted(eligible.index)
random.Random(a.seed).shuffle(players)
chosen = players[: a.n]
with open(a.out, "w") as fh:
    fh.write("\n".join(chosen) + "\n")
sel = eligible.loc[chosen]
print(f"selected: {len(chosen)}; of these, with enough games in:",
      ", ".join(f"{v} {int((sel[v] >= a.min_variant).sum())}" for v in vnames))
