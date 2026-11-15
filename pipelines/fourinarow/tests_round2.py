"""Round 2 tests (frozen in claude/fourinarow-dev-predictions-round2.md before running),
plus the cross-dataset reuse-prior check RP (added 15 Nov, 23:25, at the user's request, before running):
does reuse of a stored answer make it hard to drop when it is wrong (rigidity), and help when it fits?"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import fe_ols, ci  # noqa: E402

DATA = os.environ.get("FIAR_DIR", "/home/claude/fiar_dev")
STORES = ["reply", "relational", "chunk", "local", "default"]   # exact never meets a threat situation
d = pd.read_parquet(f"{DATA}/store_features.parquet")
p = pd.read_csv(f"{DATA}/people.csv", dtype={"sid": str}).set_index("sid")
d["game10"] = d.game_rank / 10
d["second_half"] = d.game_rank >= d.groupby("sid").game_rank.transform("median")
d["odd_game"] = d.game_rank % 2
lvl = d.groupby(["sid", "game_rank"])["level"].first().rename("glevel").reset_index()
lvl["level_up"] = (lvl.groupby("sid").glevel.diff() > 0).astype(float)
d = d.merge(lvl[["sid", "game_rank", "level_up"]], on=["sid", "game_rank"], how="left")
rng = np.random.default_rng(1)

# key-level count of visits in the first half of each person's games
early = {}
for s in STORES:
    fh = d[~d.second_half & d[f"{s}_key"].notna()]
    early[s] = fh.groupby(["sid", f"{s}_key"]).size()

rows = []
for s in STORES:
    z = d[d[f"{s}_match"].notna()]
    e = early[s].reindex(pd.MultiIndex.from_arrays([z.sid, z[f"{s}_key"]])).fillna(0).values
    rows.append(pd.DataFrame({
        "sid": z.sid.values, "store": s, "y": z[f"{s}_match"].values, "share": z[f"{s}_share"].values,
        "lseen": np.log1p(z[f"{s}_seen"].values), "lucount": np.log1p(z[f"{s}_ucount"].values),
        "err": z[f"{s}_usual_err"].values if f"{s}_usual_err" in z else np.nan, "err_level": z.err_level.values,
        "rt_z": z.rt_z.values, "move_bin": z.move_bin.values, "game10": z.game10.values,
        "second_half": z.second_half.values, "level_up": z.level_up.values, "learly": np.log1p(e),
        "played_usual": z[f"{s}_match"].values, "odd_game": z.odd_game.values}))
L = pd.concat(rows, ignore_index=True)
L["sid"] = L.sid.astype("category")
F = L[L.err.notna()].copy()                      # positions where right/wrong is defined
F["two_step"] = (F.err_level == 2).astype(float)
W = F[F.err == 1].copy()                          # usual move is wrong here

print("== G1/G2 graded rigidity (wrong usual moves; within person; store dummies)")
print("  raw: wrong usual move played", (W.groupby("err_level").y.mean() * 100).round(1).rename({1.0: "1-step", 2.0: "2-step"}).to_dict(),
      "| n", W.err_level.value_counts().sort_index().to_dict())
r = fe_ols(W, "y", ["two_step", "share", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  G1 2-step vs 1-step:", ci(r, "two_step"))
W["share_x_two"] = W.share * W.two_step
r = fe_ols(W, "y", ["two_step", "share", "share_x_two", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  G2 consistency (1-step):", ci(r, "share"), "| extra for 2-step:", ci(r, "share_x_two"))

print("\n== RP reuse prior x fit (lucount = log(1 + times this usual move was chosen before))")
for lab, sub in [("fits (usual move is right here)", F[F.err == 0]), ("wrong (usual move is an error here)", W)]:
    r = fe_ols(sub, "y", ["lucount"], fe="sid", cluster="sid", controls=["store", "move_bin"])
    r2 = fe_ols(sub, "y", ["lucount", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
    print(f"  {lab}: played, per log reuse: {ci(r, 'lucount')} | same visits: {ci(r2, 'lucount')}")
F["wrong_x_lu"] = F.err * F.lucount
r = fe_ols(F, "y", ["lucount", "err", "wrong_x_lu"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  does reuse narrow the right/wrong gap? reuse x wrong:", ci(r, "wrong_x_lu"), "(negative = reuse helps tell them apart)")
U = L[(L.played_usual == 1) & L.rt_z.notna()]
r = fe_ols(U, "rt_z", ["lucount"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  speed when the usual move is played, per log reuse (log-RT):", ci(r, "lucount", 1, "log-RT"))
r = fe_ols(F[(F.err == 0) & (F.played_usual == 1) & F.rt_z.notna()], "rt_z", ["lucount"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  ... only where it fits:", ci(r, "lucount", 1, "log-RT"))

print("\n== P1/P2 built-upon keys (second half of games; learly = log(1 + first-half visits to this key))")
S2 = L[L.second_half]
r = fe_ols(S2, "y", ["learly", "share", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  P1 all second-half rows:", ci(r, "learly"))
W2 = W[W.second_half]
r = fe_ols(W2, "y", ["learly", "share", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  P1 wrong usual moves:", ci(r, "learly"), f"(n={len(W2)})")
S2 = S2.copy(); S2["g_x_early"] = S2.game10 * S2.learly
r = fe_ols(S2, "y", ["game10", "learly", "g_x_early", "share", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  P2 decline over games:", ci(r, "game10"), "| less for early-reused keys (game x early):", ci(r, "g_x_early"))

print("\n== P3 right after the opponent got stronger")
r = fe_ols(W, "y", ["level_up", "share", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  wrong usual move played after level-up:", ci(r, "level_up"))
L2 = L[L.level_up.notna()].copy(); L2["up_x_lu"] = L2.level_up * L2.lucount
r = fe_ols(L2, "y", ["level_up", "lucount", "up_x_lu", "lseen"], fe="sid", cluster="sid", controls=["store", "move_bin"])
print("  reuse after level-up:", ci(r, "level_up"), "| heavily reused keys drop less (up x reuse):", ci(r, "up_x_lu"))

print("\n== B1 blind spots by line direction (1-step situations with one direction)")
B = d[(d.err_level == 1) & d.orient.isin(["H", "V", "D"])].copy()
B["sid"] = B.sid.astype("category")
for o in ["V", "D"]:
    B[o] = (B.orient == o).astype(float)
r = fe_ols(B, "played_err", ["V", "D"], fe="sid", cluster="sid", controls=["critical"])
print("  raw miss rate:", (B.groupby("orient").played_err.mean() * 100).round(1).to_dict())
print("  vertical vs horizontal:", ci(r, "V"), "| diagonal vs horizontal:", ci(r, "D"))


def gap(df):
    g = df.groupby(["sid", "orient"], observed=True).played_err.agg(["mean", "size"]).unstack()
    ok = (g["size"]["D"] >= 5) & (g["size"]["H"] >= 5)
    return (g["mean"]["D"] - g["mean"]["H"]).where(ok)


# ---------- person measures, reliability, questionnaires ----------
def person_measures(dd, LL, FF):
    WW = FF[FF.err == 1]
    m = pd.DataFrame({
        "store_use": LL.groupby("sid", observed=True).y.mean(),
        "rigid_1step": WW[WW.err_level == 1].groupby("sid", observed=True).y.mean(),
        "rigid_2step": WW[WW.err_level == 2].groupby("sid", observed=True).y.mean(),
        "miss_1step": dd[dd.err_level == 1].groupby("sid").played_err.mean(),
        "err_2step": dd[dd.err_level == 2].groupby("sid").played_err.mean(),
        "median_rt": dd.groupby("sid").rt_ms.median() / 1000,
    })
    m["graded_gap"] = m.rigid_2step - m.rigid_1step
    s2 = LL[LL.second_half]
    m["persistence"] = (s2[s2.learly > 0].groupby("sid", observed=True).y.mean()
                        - s2[s2.learly == 0].groupby("sid", observed=True).y.mean())
    bb = dd[(dd.err_level == 1) & dd.orient.isin(["H", "D"])]
    m["blind_gap"] = gap(bb)
    m.index = m.index.astype(str)
    return m


M = person_measures(d, L, F)
odd = person_measures(d[d.odd_game == 1], L[L.odd_game == 1], F[F.odd_game == 1])
even = person_measures(d[d.odd_game == 0], L[L.odd_game == 0], F[F.odd_game == 0])
print("\n== person measures: odd/even-game split-half reliability (Spearman-Brown)")
REL = {}
for k in M.columns:
    x = pd.concat([odd[k], even[k]], axis=1).dropna()
    rr = x.corr(method="spearman").iloc[0, 1] if len(x) > 10 else np.nan
    REL[k] = 2 * rr / (1 + rr) if rr == rr else np.nan
    print(f"  {k:12s} reliability {REL[k]:+.2f} (n={len(x)})")


def spearman_ci(x, y, z=None, B=2000):
    df = pd.concat([x, y] + ([z] if z is not None else []), axis=1).dropna()
    def rho(t):
        rk = t.rank()
        a, b = rk.iloc[:, 0], rk.iloc[:, 1]
        if z is not None:
            cz = rk.iloc[:, 2]
            a = a - np.polyval(np.polyfit(cz, a, 1), cz); b = b - np.polyval(np.polyfit(cz, b, 1), cz)
        return np.corrcoef(a, b)[0, 1]
    est = rho(df)
    bs = [rho(df.sample(len(df), replace=True, random_state=int(i))) for i in rng.integers(0, 1e9, B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    flag = "*" if lo > 0 or hi < 0 else " "
    return f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}]{flag}"


print("\n== questionnaires and capacity (Spearman rho [95% CI], * = CI excludes 0; BIS/FOS/heuristic quality: age partialled out)")
cols = ["age", "bis_total", "fos_total", "heuristic_quality"]
print(f"  {'measure':12s} {'rel':>5s}  " + "  ".join(f"{c:>24s}" for c in cols))
for k in M.columns:
    cells = []
    for c in cols:
        cells.append(spearman_ci(p[c], M[k], None if c == "age" else p.age))
    print(f"  {k:12s} {REL[k]:+.2f}  " + "  ".join(f"{x:>24s}" for x in cells))
M.join(p[["age", "bis_total", "fos_total", "heuristic_quality", "feature_drop", "planning_depth", "elo"]]).to_csv(f"{DATA}/person_measures_round2.csv")
