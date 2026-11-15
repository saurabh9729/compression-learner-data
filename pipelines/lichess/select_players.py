"""Pick players who play both fast (bullet) and slow (rapid + classical) games; sample N of them."""
import argparse
import random

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("counts")
ap.add_argument("--min-bullet", type=int, default=30)
ap.add_argument("--min-slow", type=int, default=15)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="players.txt")
a = ap.parse_args()

c = pd.read_csv(a.counts)
c["slow"] = c["rapid"] + c["classical"]
ok = c[(c.bullet >= a.min_bullet) & (c.slow >= a.min_slow)]
print(f"eligible players: {len(ok)}")
players = sorted(ok.player)
random.Random(a.seed).shuffle(players)
with open(a.out, "w") as fh:
    fh.write("\n".join(players[: a.n]) + "\n")
print(f"selected: {min(a.n, len(players))}")
