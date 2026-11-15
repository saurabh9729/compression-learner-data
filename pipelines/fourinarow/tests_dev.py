"""Tests D1-D6 (frozen in claude/fourinarow-dev-predictions.md before running)."""
import sys

import numpy as np
import pandas as pd

from stats import fe_ols, ci  # noqa: E402

import os
DATA = os.environ.get("FIAR_DIR", "/home/claude/fiar_dev")
STORES = ["exact", "reply", "relational", "chunk", "local", "default"]
COARSE, FINE = ["relational", "default", "reply"], ["exact", "local"]
d = pd.read_parquet(f"{DATA}/store_features.parquet")
p = pd.read_csv(f"{DATA}/people.csv", dtype={"sid": str})
d["sid"] = d.sid.astype("category")
d["game10"] = d.game_rank / 10
rng = np.random.default_rng(1)


def long_stores(rows, stores=STORES):
    L = []
    for s in stores:
        z = rows[rows[f"{s}_match"].notna()]
        cols = {"y": z[f"{s}_match"], "share": z[f"{s}_share"], "lseen": np.log1p(z[f"{s}_seen"])}
        if f"{s}_usual_ok" in z:
            cols["usual_ok"] = z[f"{s}_usual_ok"]
        L.append(pd.DataFrame({"sid": z.sid, "store": s, "rt_z": z.rt_z, "move_bin": z.move_bin,
                               "game10": z.game10, "level": z.level, **cols}))
    out = pd.concat(L, ignore_index=True)
    out["sid"] = out.sid.astype("category")
    return out


print("== D1 consistency -> use (within person; controls: log visits, move number)")
for s in STORES:
    z = d[d[f"{s}_match"].notna()].copy()
    z["lseen"] = np.log1p(z[f"{s}_seen"]); z["share"] = z[f"{s}_share"]
    r = fe_ols(z, f"{s}_match", ["share", "lseen"], fe="sid", cluster="sid", controls=["move_bin"])
    print(f"  {s:10s} consistency 0->1: {ci(r, 'share')}")

print("\n== D2 rigidity in critical positions (win available, or exactly one square blocks the opponent)")
c = d[d.critical.notna()]
L = long_stores(c)
print("  % usual move played, by whether it is correct:",
      (L.groupby("usual_ok").y.mean() * 100).round(1).rename({0.0: "wrong", 1.0: "correct"}).to_dict())
print("  by store, usual move WRONG:", (L[L.usual_ok == 0].groupby("store").y.agg(["mean", "size"])).round(3).to_dict("index"))
W = L[L.usual_ok == 0].copy()
r = fe_ols(W, "y", ["share", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  wrong usual move played, consistency 0->1:", ci(r, "share"))
e = c[(c.correct_move == 0) & c.played_wrong_usual.notna() & (c.n_wrong_usual > 0)]
obs, chance = e.played_wrong_usual.mean(), e.chance_wrong_usual.mean()
per = e.groupby("sid", observed=True)[["played_wrong_usual", "chance_wrong_usual"]].mean()
boots = [per.sample(len(per), replace=True, random_state=int(i)).diff(axis=1).iloc[:, 1].mean() for i in rng.integers(0, 1e9, 2000)]
diff = obs - chance
lo, hi = np.percentile(-np.array(boots), [2.5, 97.5])
print(f"  errors (missed win/block) that are a store's usual move: {obs*100:.1f}% vs chance {chance*100:.1f}% "
      f"(difference {diff*100:+.1f} pts, person bootstrap 95% CI [{lo*100:+.1f}, {hi*100:+.1f}], n errors {len(e)})")

print("\n== D3 cost per move (rt_z = log RT minus own median; negative = faster than usual)")
A = long_stores(d)
r = fe_ols(A, "y", ["rt_z", "share", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  store usual move played, per +1 log-RT (slower):", ci(r, "rt_z"))
cc = c[c.rt_z.notna()].copy(); cc["missed"] = 1 - cc.correct_move
r = fe_ols(cc, "missed", ["rt_z"], fe="sid", cluster="sid", controls=["critical", "move_bin"])
print("  missed win/block, per +1 log-RT:", ci(r, "rt_z"))
ee = e[e.rt_z.notna()].copy()
r = fe_ols(ee, "played_wrong_usual", ["rt_z", "chance_wrong_usual"], fe="sid", cluster="sid", controls=["critical"])
print("  among errors, error = a store's usual move, per +1 log-RT:", ci(r, "rt_z"))

print("\n== D4 entrenchment across games (per 10 games; controls: opponent level, move number)")
r = fe_ols(A, "y", ["game10", "level", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  store usual move played:", ci(r, "game10"))
r = fe_ols(A, "share", ["game10", "level", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  store consistency:", ci(r, "game10"))

print("\n== D5 early openings persist (later openings, choice among squares used as openings before)")
o = d[d.move_in_game == 0].sort_values(["sid", "game_rank"])
rows, cs = [], 0
for (sid, col), g in o.groupby(["sid", "color"], observed=True):
    hist = []   # (game_rank, square)
    for r_ in g.itertuples():
        if r_.game_rank >= 5 and hist:
            counts = pd.Series([h[1] for h in hist]).value_counts()
            early = {h[1] for h in hist if h[0] < 5}
            last = hist[-1][1]
            for sq, n in counts.items():
                rows.append(dict(sid=sid, cs=cs, chosen=float(sq == r_.tile), lcount=np.log1p(n),
                                 early=float(sq in early), last=float(sq == last)))
            cs += 1
        hist.append((r_.game_rank, r_.tile))
O = pd.DataFrame(rows); O["sid"] = O.sid.astype("category")
r = fe_ols(O, "chosen", ["early", "lcount", "last"], fe="cs", cluster="sid")
print(f"  used in first 5 games (same count, same last-used): {ci(r, 'early')}  [{O.cs.nunique():,} later openings]")

print("\n== D6 capacity (between person; Spearman rho, 95% person bootstrap CI)")
half = d.game_rank >= d.groupby("sid", observed=True).game_rank.transform("median")


def person_metrics(rows):
    Lp = long_stores(rows)
    use = Lp.groupby(["sid", "store"], observed=True).y.mean().unstack()
    m = pd.DataFrame({
        "store_use": Lp.groupby("sid", observed=True).y.mean(),
        "coarse_minus_fine": use[COARSE].mean(axis=1) - use[FINE].mean(axis=1),
        "median_rt": rows.groupby("sid", observed=True).rt_ms.median() / 1000,
    })
    cr = long_stores(rows[rows.critical.notna()])
    cr = cr[cr.usual_ok == 0]
    g = cr.groupby("sid", observed=True)
    m["rigidity"] = g.y.mean().where(g.size() >= 5)          # wrong usual move played (needs >= 5 such cases)
    er = rows[rows.critical.notna()]
    m["miss_rate"] = 1 - er.groupby("sid", observed=True).correct_move.mean()
    m.index = m.index.astype(str)
    return m


def spearman_ci(x, y, z=None, B=2000):
    df = pd.concat([x, y] + ([z] if z is not None else []), axis=1).dropna()
    def rho(t):
        rk = t.rank()
        if z is not None:
            a, b, cz = rk.iloc[:, 0], rk.iloc[:, 1], rk.iloc[:, 2]
            a = a - np.polyval(np.polyfit(cz, a, 1), cz); b = b - np.polyval(np.polyfit(cz, b, 1), cz)
            return np.corrcoef(a, b)[0, 1]
        return np.corrcoef(rk.iloc[:, 0], rk.iloc[:, 1])[0, 1]
    est = rho(df)
    bs = [rho(df.sample(len(df), replace=True, random_state=int(i))) for i in rng.integers(0, 1e9, B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}] (n={len(df)})"


P = p.set_index("sid")
M = person_metrics(d)
M2 = person_metrics(d[half])
print("  age with:")
for k, pred in [("store_use", "+ (younger more? predicted -)"), ("coarse_minus_fine", "predicted -"), ("rigidity", "predicted -"),
                ("miss_rate", "(descriptive)"), ("median_rt", "(descriptive)")]:
    print(f"    {k:18s} {spearman_ci(P.age, M[k])}   {pred}")
for q, sign in [("bis_total", "+"), ("fos_total", "-")]:
    print(f"  {q} (age partialled out), predicted {sign} for use and rigidity, {'-' if sign == '+' else '+'} for RT:")
    for k in ["store_use", "rigidity", "median_rt"]:
        print(f"    {k:18s} {spearman_ci(P[q], M[k], P.age)}")
print("  first-half model parameters -> second-half behaviour (age partialled out):")
print(f"    feature_drop_h1 -> rigidity (pred +):   {spearman_ci(P.feature_drop_h1, M2.rigidity, P.age)}")
print(f"    planning_depth_h1 -> store_use (pred -): {spearman_ci(P.planning_depth_h1, M2.store_use, P.age)}")
print(f"    heuristic_quality_h1 -> store_use (descriptive): {spearman_ci(P.heuristic_quality_h1, M2.store_use, P.age)}")
print(f"  people with a rigidity score: {M.rigidity.notna().sum()} (all games), {M2.rigidity.notna().sum()} (second half)")
M.join(P[["age", "bis_total", "fos_total"]]).to_csv(f"{DATA}/person_metrics.csv")
