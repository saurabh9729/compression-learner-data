"""Safety check before anything is committed to this public repo.

Fails (exit code 1) if any player name from the selection list appears in any text cell
of the output files, or if a column that could identify people is present.
Prints only counts, never names, because the Actions logs are public.

Memory-light: parquet files are read one text column at a time, in batches, so files with
tens of millions of rows can be checked on a small machine.

Usage: python check_no_names.py players.txt processed/lichess/*.parquet processed/lichess/*.csv
"""
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FORBIDDEN_COLUMNS = {"site", "white", "black", "username", "name", "url", "date", "utcdate"}
BATCH = 500_000


def check_parquet(path, names_arr):
    f = pq.ParquetFile(path)
    schema = f.schema_arrow
    bad = 0
    cols = {n.lower() for n in schema.names} & FORBIDDEN_COLUMNS
    if cols:
        print(f"FAIL {path}: forbidden columns present ({len(cols)})")
        bad += 1
    for field in schema:
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_boolean(field.type):
            continue
        hits = 0
        for batch in f.iter_batches(batch_size=BATCH, columns=[field.name]):
            col = batch.column(0).cast(pa.string())
            hits += pc.sum(pc.is_in(col, value_set=names_arr)).as_py() or 0
        if hits:
            print(f"FAIL {path}: column '{field.name}' has {hits} cells equal to a player name")
            bad += 1
    print(f"checked {path}: {f.metadata.num_rows} rows")
    return bad


def check_csv(path, names):
    df = pd.read_csv(path)
    bad = 0
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
    return bad


def main():
    names = {l.strip() for l in open(sys.argv[1]) if l.strip()}
    names_arr = pa.array(sorted(names), type=pa.string())
    bad = 0
    for path in sys.argv[2:]:
        bad += check_parquet(path, names_arr) if path.endswith(".parquet") else check_csv(path, names)
    if bad:
        sys.exit(1)
    print("ok: no player names or identifying columns found")


if __name__ == "__main__":
    main()
