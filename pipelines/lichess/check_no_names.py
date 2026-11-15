"""Safety check before anything is committed to this public repo.

Fails (exit code 1) if any player name from the selection list appears in any text cell
of the output files, or if a column that could identify people is present.
Prints only counts, never names, because the Actions logs are public.

Usage: python check_no_names.py players.txt processed/lichess/*.parquet processed/lichess/*.csv
"""
import sys

import pandas as pd

FORBIDDEN_COLUMNS = {"site", "white", "black", "username", "name", "url", "date", "utcdate"}

names = {l.strip() for l in open(sys.argv[1]) if l.strip()}
bad = 0
for path in sys.argv[2:]:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    cols = {c.lower() for c in df.columns} & FORBIDDEN_COLUMNS
    if cols:
        print(f"FAIL {path}: forbidden columns present ({len(cols)})")
        bad += 1
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):  # every text-like column
            hits = df[c].astype(str).isin(names).sum()
            if hits:
                print(f"FAIL {path}: column '{c}' has {hits} cells equal to a player name")
                bad += 1
    print(f"checked {path}: {len(df)} rows")
if bad:
    sys.exit(1)
print("ok: no player names or identifying columns found")
