"""
Stream a Lichess monthly PGN dump from stdin and filter it without parsing moves.

Pass 1 (--count): count games per player per speed class -> counts.csv
Pass 2 (--extract players.txt): write only games involving the listed players to stdout.

Usage (run on GitHub Actions, never storing the full dump):
  curl -sL URL | zstdcat | python filter_lichess.py --count counts.csv
  curl -sL URL | zstdcat | python filter_lichess.py --extract players.txt | zstd -q -o filtered.pgn.zst
"""
import argparse
import collections
import csv
import sys


def speed_class(tc):
    if not tc or "+" not in tc:
        return None
    base, inc = tc.split("+")
    est = int(base) + 40 * int(inc)
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    if est < 1500:
        return "rapid"
    return "classical"


def games(stream):
    """Yield (headers dict, list of raw lines) per game, by splitting on header blocks."""
    buf, headers, in_moves = [], {}, False
    for line in stream:
        if line.startswith("[") and in_moves:
            yield headers, buf
            buf, headers, in_moves = [], {}, False
        buf.append(line)
        if line.startswith("["):
            k, _, v = line[1:].partition(" ")
            headers[k] = v.rstrip().rstrip("]").strip('"')
        elif line.strip():
            in_moves = True
    if buf:
        yield headers, buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count")
    ap.add_argument("--extract")
    ap.add_argument("--require-clock", action="store_true", default=True)
    a = ap.parse_args()
    src = sys.stdin

    if a.count:
        counts = collections.defaultdict(collections.Counter)
        for h, _ in games(src):
            cls = speed_class(h.get("TimeControl"))
            if cls is None:
                continue
            for side in ("White", "Black"):
                counts[h.get(side, "?")][cls] += 1
        with open(a.count, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["player", "bullet", "blitz", "rapid", "classical"])
            for p, c in counts.items():
                w.writerow([p, c["bullet"], c["blitz"], c["rapid"], c["classical"]])
        return

    if a.extract:
        keep = {l.strip() for l in open(a.extract) if l.strip()}
        out = sys.stdout
        for h, lines in games(src):
            if h.get("White") in keep or h.get("Black") in keep:
                text = "".join(lines)
                if a.require_clock and "[%clk" not in text:
                    continue
                out.write(text)


if __name__ == "__main__":
    main()
