"""Clean extract of the developmental four-in-a-row data (Ma, Phaneuf, van Opheusden, Ma & Hartley).
Keeps: study ID, age (0.1 y), gender, BIS-Brief total, FOS total, model parameters, move-level data.
Drops: IP addresses, coordinates, dates of birth, dates, Qualtrics response IDs, race/ethnicity."""
import json, sys
import pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]
import os; os.makedirs(out, exist_ok=True)
d = pd.read_csv(f"{src}/trialdata.csv", header=None, names=["subj", "idx", "t", "ev"], low_memory=False)
d["sid"] = d.subj.astype(str).str.split(":").str[0]
d = d[~d.sid.isin(["null", "test_im", "nan"])]
rows, last_turn, game = [], {}, {}
for r in d.itertuples(index=False):
    try:
        j = json.loads(r.ev)
    except Exception:
        continue
    et, info, tm = j["event_type"], j.get("event_info", {}), j.get("event_time")
    s = r.sid
    if et == "start game":
        game[s] = dict(game=info.get("game_num"), practice=info.get("is_practice"), category=info.get("category"), level=info.get("level"))
        last_turn.pop(s, None)
    elif et == "your turn":
        last_turn[s] = tm
    elif et == "user move" and s in game:
        rt = (tm - last_turn[s]) if s in last_turn and tm and last_turn[s] else None
        rows.append(dict(sid=s, **game[s], color=info.get("user_color"), bp=info.get("bp"), wp=info.get("wp"),
                         tile=info.get("tile"), rt_ms=rt))
        last_turn.pop(s, None)
    elif et == "end game" and s in game:
        res = info.get("result")
        for row in rows[::-1]:
            if row["sid"] != s or row["game"] != game[s]["game"]:
                break
            row["result"] = res
m = pd.DataFrame(rows)
m["move_in_game"] = m.groupby(["sid", "game"]).cumcount()
m.to_parquet(f"{out}/moves.parquet", index=False)

def sid_of(x):
    x = str(x).split(":")[0]
    return x
dem = pd.read_csv(f"{src}/demog.csv")[["subject_ID", "Gender", "ExactAge"]].rename(columns={"subject_ID": "sid", "Gender": "gender", "ExactAge": "age"})
bis = pd.read_csv(f"{src}/bis_age.csv")[["StudyID", "bis_total"]].rename(columns={"StudyID": "sid"})
fos = pd.read_csv(f"{src}/FOS_age.csv")[["StudyID", "fos_total"]].rename(columns={"StudyID": "sid"})
par = pd.read_csv(f"{src}/params_with_metrics.csv")
par["sid"] = par.subject.map(sid_of)
par = par.groupby("sid")[["feature drop rate", "heuristic quality", "planning depth", "elo"]].mean().reset_index()
par.columns = ["sid", "feature_drop", "heuristic_quality", "planning_depth", "elo"]
for t in (dem, bis, fos):
    t["sid"] = t.sid.astype(str)
p = dem.merge(bis, on="sid", how="left").merge(fos, on="sid", how="left").merge(par, on="sid", how="outer")
p["age"] = p.age.round(1)
def half(f, tag):
    h = pd.read_csv(f"{src}/{f}")
    h["sid"] = h.subject.map(sid_of)
    h = h.groupby("sid")[["feature drop rate", "heuristic quality", "planning depth"]].mean().reset_index()
    h.columns = ["sid"] + [f"{c}_{tag}" for c in ["feature_drop", "heuristic_quality", "planning_depth"]]
    return h
p = p.merge(half("params_with_metrics_1st_half.csv", "h1"), on="sid", how="left")
p = p.merge(half("params_with_metrics_2nd_half.csv", "h2"), on="sid", how="left")
p.to_csv(f"{out}/people.csv", index=False)
print("moves", len(m), "people with moves", m.sid.nunique(), "people table", len(p))
print("join check: move sids in people table", m.sid.isin(p.sid).mean().round(3), "| with age", m.sid.isin(p.sid[p.age.notna()]).mean().round(3))
print(p.describe().round(2).to_string())
