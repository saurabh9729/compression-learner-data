"""
Lichess pipeline: does costly computation push players onto stored knowledge?

Input : PGN files from the Lichess open database (database.lichess.org, .pgn or .pgn.zst)
        or per-user API exports (lichess.org/api/games/user/<name>?clocks=true&evals=true).
        Games need [%clk] comments; [%eval] comments are used when present.
Output: one row per move (parquet) with the measures below, plus a per-player summary.
        Positions are stored as 64-bit hashes: `pos` (exact position) and `kind` (pawn skeleton +
        material + side to move), to keep files small. Raw positions can be rebuilt from the games.

Key measures
- time_class      : bullet / blitz / rapid / classical (Lichess rule: base + 40 * increment).
- seen_before     : how often this player had faced this exact position (same side to move)
                    in earlier games, in any time control.
- habit_move      : the player's most frequent earlier move in this position (their stored answer).
- habit_match     : 1 if they played it now (only defined when seen_before >= MIN_SEEN).
- struct_seen     : how often this player had faced a position of the same KIND: same pawn
                    skeleton, same material, same side to move (pieces may stand elsewhere).
                    Exact positions rarely repeat after the opening; kinds of position do.
- struct_habit_move, struct_habit_match, struct_habit_share: the player's most frequent
                    earlier move pattern (piece + from + to) in this kind of position, among
                    patterns that are legal now; whether they played it (struct_seen >= MIN_SEEN);
                    and how dominant it was. Positions with seen_before == 0 but struct_seen >= 3
                    are 'similar but not identical': where a stored answer is reused by analogy.
- time_spent      : seconds used on the move (clock before - clock after + increment).
                    Missing for each side's first move: Lichess adds no increment there and
                    tournament 'berserk' halves the clock, so that move's time is not comparable.
- berserk         : 1 if this side started with about half its clock (tournament berserk).
- clock_frac      : remaining clock as a fraction of the starting clock (within-game cost).
- eval_loss_cp    : centipawn loss of the move, from [%eval] before and after (if available).

Other stored representations (added 15 Nov, v3). For each key K below: K_seen (earlier visits),
K_habit (usual legal move pattern), K_match (played it; defined when K_seen >= MIN_SEEN), K_share.
- pawn            : pawn skeleton + side to move (coarsest state summary).
- own             : the mover's own setup (all own pieces and pawns); the opponent is ignored.
- reply           : the opponent's last move pattern (stimulus -> response, board ignored).
- chunk           : the mover's own last two move patterns in this game (action sequence, board ignored).
- kind_same_* / kind_other_*: the pawns + material summary learned only in games of the same speed
                    group (bullet+blitz vs rapid+classical) / only in the other group.
- kind_dist       : squares that differ from the last position of the same kind the player met.
- result          : game result for the mover (1, 0.5, 0); opp_rating; time_forfeit (game ended on time).
- cond            : reply to the opponent's last move within this pawn structure (conditional sub-rule).
- global_*        : default policy: the player's favourite move patterns overall (same colour), ignoring
                    the state. global_logp = log probability of the played move under this policy
                    (add-one smoothing over legal moves); policy compression predicts it rises with cost.
- goal_*          : subgoals: where the player's pieces went (piece@square) within 3 own moves after
                    earlier visits to this kind; goal_match = current move lands on one of the top 3
                    (possibly by a different route than before).
- epd             : the position, stored only for similar-but-not-identical positions and a 1% sample
                    (for engine checks); the output stays in the user's private Drive.

Privacy (the output repo is public)
- Player names are replaced by anonymous IDs: HMAC-SHA256 of the name with a secret salt
  (env LICHESS_ID_SALT). Without the salt the IDs cannot be turned back into names.
  If the salt is missing, a random one is used and thrown away (IDs then differ between runs).
- No game links (Site), no opponent names, no dates are written. Logs print only totals.

Usage
  python lichess_pipeline.py games.pgn.zst --out moves.parquet --min-games 20
"""
import argparse
import collections
import hashlib
import hmac
import os
import secrets
import io
import math
import re
import sys

import chess
import chess.pgn
import pandas as pd

MIN_SEEN = 3          # earlier visits needed before a position counts as "familiar"
MATE_CP = 1000        # centipawn value used for mate scores

CLK_RE = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
EVAL_RE = re.compile(r"\[%eval (#?-?\d+(?:\.\d+)?)\]")


def open_text(path):
    if path.endswith(".zst"):
        import zstandard
        fh = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
    return open(path, encoding="utf-8")


def time_class(tc):
    """Lichess categories from 'base+inc' in seconds (estimated duration = base + 40 * inc)."""
    if not tc or tc == "-" or "+" not in tc:
        return None, None, None
    base, inc = (int(x) for x in tc.split("+"))
    est = base + 40 * inc
    if est < 180:
        cls = "bullet"
    elif est < 480:
        cls = "blitz"
    elif est < 1500:
        cls = "rapid"
    else:
        cls = "classical"
    return cls, base, inc


def parse_clock(comment):
    m = CLK_RE.search(comment or "")
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def parse_eval(comment):
    """Eval from White's point of view in centipawns; mate mapped to +/- MATE_CP."""
    m = EVAL_RE.search(comment or "")
    if not m:
        return None
    v = m.group(1)
    if v.startswith("#"):
        return MATE_CP if not v.startswith("#-") else -MATE_CP
    return max(-MATE_CP, min(MATE_CP, float(v) * 100))


def iter_games(paths):
    for p in paths:
        with open_text(p) as fh:
            while True:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                yield g


def count_games(paths):
    """First pass: games per player per time class (to select players who play both)."""
    counts = collections.defaultdict(collections.Counter)
    for g in iter_games(paths):
        cls, _, _ = time_class(g.headers.get("TimeControl"))
        if cls is None:
            continue
        for side in ("White", "Black"):
            counts[g.headers.get(side, "?")][cls] += 1
    return counts


def select_players(counts, min_games, classes=("bullet", "classical")):
    return {p for p, c in counts.items() if all(c[k] >= min_games for k in classes)}


def iter_raw_games(paths, players):
    """Raw text of games involving the selected players, with their sort key.

    Games are kept as text (about 2 KB each) and parsed one at a time later, so a whole
    month fits in memory; parsed games are about 20 times larger."""
    for p in paths:
        with open_text(p) as fh:
            lines, heads, in_moves = [], {}, False
            for line in fh:
                if line.startswith("[") and in_moves:
                    if heads.get("White") in players or heads.get("Black") in players:
                        yield (heads.get("UTCDate", ""), heads.get("UTCTime", "")), "".join(lines)
                    lines, heads, in_moves = [], {}, False
                lines.append(line)
                if line.startswith("["):
                    k, _, v = line[1:].partition(" ")
                    heads[k] = v.rstrip().rstrip("]").strip('"')
                elif line.strip():
                    in_moves = True
            if lines and (heads.get("White") in players or heads.get("Black") in players):
                yield (heads.get("UTCDate", ""), heads.get("UTCTime", "")), "".join(lines)


def hash64(text):
    """Stable signed 64-bit hash (used for position keys in memory and in the output)."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big", signed=True)


def struct_key(board):
    """Kind of position: pawn skeleton + material + side to move."""
    wp = int(board.pieces(chess.PAWN, chess.WHITE))
    bp = int(board.pieces(chess.PAWN, chess.BLACK))
    mat = "".join(str(len(board.pieces(pt, c)))
                  for c in (chess.WHITE, chess.BLACK)
                  for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
    return hash64(f"{int(board.turn)}:{wp:x}:{bp:x}:{mat}")


def pawn_key(board):
    """Pawn skeleton + side to move only (coarsest summary)."""
    wp = int(board.pieces(chess.PAWN, chess.WHITE))
    bp = int(board.pieces(chess.PAWN, chess.BLACK))
    return hash64(f"P{int(board.turn)}:{wp:x}:{bp:x}")


def own_key(board):
    """Own setup: where the mover's pieces (incl. pawns) stand; opponent ignored."""
    c = board.turn
    parts = [f"{pt}:{int(board.pieces(pt, c)):x}" for pt in chess.PIECE_TYPES]
    return hash64(f"O{int(c)}:" + "|".join(parts))


def expand_board(board):
    """64-character string, one per square ('.' = empty), for counting differing squares."""
    out = []
    for ch in board.board_fen():
        if ch.isdigit():
            out.append("." * int(ch))
        elif ch != "/":
            out.append(ch)
    return "".join(out)


def square_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def move_pattern(board, move):
    """Piece letter + from + to, e.g. 'Nb1c3'. Promotion suffix kept by uci()."""
    return sys.intern(board.piece_at(move.from_square).symbol().upper() + move.uci())


def legal_pattern(board, pattern):
    """Is this stored pattern (same piece, same squares) playable in the current position?"""
    try:
        mv = chess.Move.from_uci(pattern[1:])
    except ValueError:
        return False
    pc = board.piece_at(mv.from_square)
    return mv in board.legal_moves and pc is not None and pc.symbol().upper() == pattern[0]


class Memory:
    """What each player did before, per key: key -> one move (seen once) or a Counter.

    Most positions are met only once, so storing a single interned string for them
    (instead of a Counter) keeps a month of history within a few GB."""

    def __init__(self):
        self.d = {}

    def get(self, key):
        v = self.d.get(key)
        if v is None:
            return collections.Counter()
        if isinstance(v, str):
            return collections.Counter({v: 1})
        return v

    def add(self, key, item):
        v = self.d.get(key)
        if v is None:
            self.d[key] = item
        elif isinstance(v, str):
            c = collections.Counter({v: 1})
            c[item] += 1
            self.d[key] = c
        else:
            v[item] += 1


SCHEMA = None  # built lazily (pyarrow imported only when writing)


def _schema():
    import pyarrow as pa
    f64, i64, s = pa.float64(), pa.int64(), pa.string()
    return pa.schema([
        ("player", s), ("game", i64), ("time_class", s), ("speed", s), ("base", i64), ("inc", i64),
        ("rating", f64), ("ply", i64), ("pos", i64), ("kind", i64), ("move", s), ("n_legal", i64),
        ("seen_before", i64), ("habit_move", s), ("habit_match", f64), ("habit_share", f64),
        ("struct_seen", i64), ("struct_habit_move", s), ("struct_habit_match", f64),
        ("struct_habit_share", f64), ("time_spent", f64), ("clock", f64), ("berserk", i64),
        ("clock_frac", f64), ("eval_after", f64), ("eval_loss_cp", f64),
        # added 15 Nov (v3): more summary keys, chunks, speed-split history, distance, outcomes
        *[(f"{k}_{c}", t) for k in EXTRA_KEYS for c, t in
          (("seen", i64), ("habit", s), ("match", f64), ("share", f64))],
        ("kind_same_seen", i64), ("kind_same_match", f64),
        ("kind_other_seen", i64), ("kind_other_match", f64),
        ("kind_dist", i64), ("result", f64), ("opp_rating", f64), ("time_forfeit", i64),
        ("epd", s),
        # v4: default policy (state-independent), subgoals
        ("global_seen", i64), ("global_habit", s), ("global_match", f64), ("global_share", f64),
        ("global_logp", f64), ("goal_seen", i64), ("goal_top", s), ("goal_match", f64),
    ])


EXTRA_KEYS = ("pawn", "own", "reply", "chunk", "cond")


SPEED = {"bullet": "fast", "blitz": "medium", "rapid": "slow", "classical": "slow"}


def lookup(mem, key, board):
    """Earlier visits under this key: count, usual pattern that is legal now, its share."""
    past = mem.get(key)
    seen = sum(past.values())
    habit = next((q for q, _ in past.most_common() if legal_pattern(board, q)), None) if seen else None
    return seen, habit, (past[habit] / seen if habit else None)


def extract_moves(paths, players, out_path, ids, chunk_rows=100_000):
    """Second pass: one row per move by a selected player, in chronological order per player.

    Writes the parquet file in chunks (player names already replaced by `ids`)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = _schema()
    games = sorted(iter_raw_games(paths, players), key=lambda t: t[0])
    print(f"games to process: {len(games)}", file=sys.stderr, flush=True)

    memory, smemory = Memory(), Memory()   # exact position / kind of position
    xmem = {k: Memory() for k in EXTRA_KEYS}  # other summaries and chunks
    spmem = Memory()                           # kind of position, split by speed group
    lastboard = {}                             # (player, kind) -> last board as 64 characters
    gpol = collections.defaultdict(collections.Counter)  # (player, colour) -> all move patterns
    goalmem = Memory()                         # (player, kind) -> destinations of next 3 own moves
    pidx = {name: i for i, name in enumerate(sorted(players))}   # small int per player for keys
    rows, n_rows = [], 0
    writer = pq.ParquetWriter(out_path, schema)

    def flush():
        nonlocal rows, n_rows
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            n_rows += len(rows)
            rows = []

    for gi in range(len(games)):
        g = chess.pgn.read_game(io.StringIO(games[gi][1]))
        games[gi] = None                     # free the text once parsed
        if g is None:
            continue
        h = g.headers
        cls, base, inc = time_class(h.get("TimeControl"))
        if cls is None:
            continue
        board = g.board()
        prev_clock = {chess.WHITE: None, chess.BLACK: None}  # set from each side's first clock
        berserk = {chess.WHITE: 0, chess.BLACK: 0}
        prev_eval = 20.0  # small white edge at the start
        res = h.get("Result", "")
        white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res)
        forfeit = int("time" in h.get("Termination", "").lower())
        own_hist = {chess.WHITE: [], chess.BLACK: []}   # this game's own move patterns
        own_dest = {chess.WHITE: [], chess.BLACK: []}   # this game's own destinations ('N@d5')
        pending_goals = []                               # (kind key, own move index, colour)
        last_pat = None                                  # opponent's last move pattern
        node = g
        ply = 0
        while node.variations:
            nxt = node.variations[0]
            mover = board.turn
            name = h.get("White") if mover == chess.WHITE else h.get("Black")
            rating = h.get("WhiteElo") if mover == chess.WHITE else h.get("BlackElo")

            clk = parse_clock(nxt.comment)
            spent = None
            if clk is not None:
                if prev_clock[mover] is None:          # this side's first move
                    berserk[mover] = int(base > 0 and clk <= 0.55 * base)
                else:
                    spent = max(0.0, prev_clock[mover] - clk + (inc or 0))
                prev_clock[mover] = clk
            ev = parse_eval(nxt.comment)
            loss = None
            if ev is not None and prev_eval is not None:
                sign = 1 if mover == chess.WHITE else -1
                loss = max(0.0, sign * (prev_eval - ev))

            if name in players:
                move = sys.intern(nxt.move.uci())
                pos = hash64(board.epd())
                kind = struct_key(board)
                pkey, kkey = pos * 4096 + pidx[name], kind * 4096 + pidx[name]  # one int per key
                past = memory.get(pkey)
                seen = sum(past.values())
                habit = past.most_common(1)[0][0] if seen else None
                pat = move_pattern(board, nxt.move)
                spast = smemory.get(kkey)
                sseen = sum(spast.values())
                shabit = next((q for q, _ in spast.most_common() if legal_pattern(board, q)), None)
                try:
                    rt = float(rating)
                except (TypeError, ValueError):
                    rt = None
                opp = h.get("BlackElo") if mover == chess.WHITE else h.get("WhiteElo")
                try:
                    opp_rt = float(opp)
                except (TypeError, ValueError):
                    opp_rt = None
                pi = pidx[name]
                xkeys = {
                    "pawn": pawn_key(board),
                    "own": own_key(board),
                    "reply": hash64(f"R{int(mover)}:{last_pat}") if last_pat else None,
                    "chunk": (hash64(f"C{int(mover)}:{own_hist[mover][-2]}>{own_hist[mover][-1]}")
                              if len(own_hist[mover]) >= 2 else None),
                    "cond": (hash64(f"Q{int(mover)}:{last_pat}:{pawn_key(board)}") if last_pat else None),
                }
                extra = {}
                for k, kv in xkeys.items():
                    if kv is None:
                        extra.update({f"{k}_seen": 0, f"{k}_habit": None, f"{k}_match": None, f"{k}_share": None})
                        continue
                    xs, xh, xsh = lookup(xmem[k], kv * 4096 + pi, board)
                    extra.update({f"{k}_seen": xs, f"{k}_habit": xh,
                                  f"{k}_match": (float(pat == xh) if xs >= MIN_SEEN and xh else None),
                                  f"{k}_share": xsh})
                # default policy: the player's favourite move patterns overall, ignoring the state
                gc = gpol[(pi, mover)]
                gseen = sum(gc.values()) if gc else 0
                legal_pats = [move_pattern(board, m) for m in board.legal_moves]
                cnts = [gc.get(q, 0) for q in legal_pats]
                denom = sum(cnts) + len(legal_pats)          # add-one smoothing over legal moves
                gbest = max(range(len(cnts)), key=cnts.__getitem__) if cnts and max(cnts) > 0 else None
                ghabit = legal_pats[gbest] if gbest is not None else None
                # subgoals: where the player's pieces usually went within 3 moves in this kind
                gpast = goalmem.get(kkey)
                goal_seen = sum(gpast.values())
                top3 = [q for q, _ in gpast.most_common(3)]
                dest = sys.intern(pat[0] + "@" + chess.square_name(nxt.move.to_square))
                extra.update(
                    global_seen=gseen, global_habit=ghabit,
                    global_match=(float(pat == ghabit) if gseen >= MIN_SEEN and ghabit else None),
                    global_share=(gc[ghabit] / gseen if ghabit else None),
                    global_logp=math.log((gc.get(pat, 0) + 1) / denom) if denom else None,
                    goal_seen=goal_seen, goal_top=(top3[0] if top3 else None),
                    goal_match=(float(dest in top3) if goal_seen >= MIN_SEEN else None),
                )
                pending_goals.append((kkey, len(own_hist[mover]), mover))
                sg = 0 if cls in ("bullet", "blitz") else 1
                same_s, same_h, _ = lookup(spmem, kkey * 2 + sg, board)
                oth_s, oth_h, _ = lookup(spmem, kkey * 2 + (1 - sg), board)
                cur = expand_board(board)
                prev_b = lastboard.get(kkey)
                store_epd = (seen == 0 and sseen >= MIN_SEEN) or pos % 100 == 0
                extra.update(
                    kind_same_seen=same_s,
                    kind_same_match=(float(pat == same_h) if same_s >= MIN_SEEN and same_h else None),
                    kind_other_seen=oth_s,
                    kind_other_match=(float(pat == oth_h) if oth_s >= MIN_SEEN and oth_h else None),
                    kind_dist=(square_distance(cur, prev_b) if prev_b else None),
                    result=(None if white_score is None else
                            (white_score if mover == chess.WHITE else 1.0 - white_score)),
                    opp_rating=opp_rt, time_forfeit=forfeit,
                    epd=(board.epd() if store_epd else None),
                )
                rows.append(dict(**extra,
                    player=ids[name], game=gi, time_class=cls, speed=SPEED[cls],
                    base=base, inc=inc, rating=rt, ply=ply, pos=pos, kind=kind,
                    move=move, n_legal=board.legal_moves.count(),
                    seen_before=seen, habit_move=habit,
                    habit_match=(float(move == habit) if seen >= MIN_SEEN else None),
                    habit_share=(past[habit] / seen if seen else None),
                    struct_seen=sseen, struct_habit_move=shabit,
                    struct_habit_match=(float(pat == shabit) if sseen >= MIN_SEEN and shabit else None),
                    struct_habit_share=(spast[shabit] / sseen if shabit else None),
                    time_spent=spent, clock=clk, berserk=berserk[mover],
                    clock_frac=(clk / base if clk is not None and base else None),
                    eval_after=ev, eval_loss_cp=loss,
                ))
                memory.add(pkey, move)
                smemory.add(kkey, pat)
                for k, kv in xkeys.items():
                    if kv is not None:
                        xmem[k].add(kv * 4096 + pi, pat)
                spmem.add(kkey * 2 + sg, pat)
                lastboard[kkey] = cur
                gc[pat] += 1
                if len(rows) >= chunk_rows:
                    flush()
                    print(f"{n_rows:,} moves written ({gi + 1:,} of {len(games):,} games)",
                          file=sys.stderr, flush=True)
            prev_eval = ev if ev is not None else None
            mp = move_pattern(board, nxt.move)
            own_hist[mover].append(mp)
            own_dest[mover].append(sys.intern(mp[0] + "@" + chess.square_name(nxt.move.to_square)))
            last_pat = mp
            board.push(nxt.move)
            node = nxt
            ply += 1
        # game over: store where each visited kind led within the next 3 own moves (subgoals)
        for kk, i, col in pending_goals:
            for dst in set(own_dest[col][i:i + 3]):
                goalmem.add(kk, dst)
    flush()
    writer.close()
    return n_rows


def player_summary(df):
    fam = df[df.seen_before >= MIN_SEEN]
    out = (fam.groupby(["player", "time_class"])
              .agg(familiar_moves=("habit_match", "size"),
                   habit_match_rate=("habit_match", "mean"),
                   median_time_familiar=("time_spent", "median"))
              .reset_index())
    nov = (df[df.seen_before == 0].groupby(["player", "time_class"])
             .agg(median_time_novel=("time_spent", "median")).reset_index())
    near = (df[(df.seen_before == 0) & df.struct_habit_match.notna()]
              .groupby(["player", "time_class"])
              .agg(near_moves=("struct_habit_match", "size"),
                   near_match_rate=("struct_habit_match", "mean")).reset_index())
    out = out.merge(nov, on=["player", "time_class"], how="outer")
    return out.merge(near, on=["player", "time_class"], how="outer")


def anonymize(names):
    """Map each player name to a stable anonymous ID (secret salt; see Privacy above)."""
    salt = os.environ.get("LICHESS_ID_SALT", "")
    if not salt:
        print("warning: LICHESS_ID_SALT not set; using a one-off random salt", file=sys.stderr)
        salt = secrets.token_hex(32)
    key = salt.encode()
    return {n: "p_" + hmac.new(key, n.encode(), hashlib.sha256).hexdigest()[:12] for n in names}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", nargs="+")
    ap.add_argument("--out", default="moves.parquet")
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--players", help="file with one player name per line (skips selection)")
    a = ap.parse_args(argv)
    if a.players:
        players = {l.strip() for l in open(a.players) if l.strip()}
    else:
        counts = count_games(a.pgn)
        players = select_players(counts, a.min_games)
    print(f"players: {len(players)}", file=sys.stderr)
    ids = anonymize(sorted(players))
    n = extract_moves(a.pgn, players, a.out, ids)
    cols = ["player", "time_class", "seen_before", "habit_match", "time_spent",
            "struct_habit_match"]
    df = pd.read_parquet(a.out, columns=cols)
    summ = player_summary(df)
    summ.to_csv(a.out.replace(".parquet", "_players.csv"), index=False)
    # Totals only: logs of a public repo are public.
    print(f"moves: {n}, players: {df.player.nunique()}")
    print(summ.groupby("time_class")[["habit_match_rate", "median_time_familiar",
                                      "median_time_novel", "near_match_rate"]].median().to_string())


if __name__ == "__main__":
    main()
