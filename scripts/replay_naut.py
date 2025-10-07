import argparse
import sqlite3
import time
from datetime import datetime
import sys


def parse_ts(x):
    """Convert ISO8601 or numeric timestamp to float seconds."""
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return datetime.fromisoformat(x).timestamp()
    except Exception:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-Src", required=True, help="source DB path")
    p.add_argument("-Dst", required=True, help="destination DB path")
    p.add_argument("-Date", required=True, help="target date YYYY-MM-DD")
    p.add_argument("-Symbols", required=True, help="comma-separated symbols")
    p.add_argument("-Speed", type=float, default=1.0, help="replay speed multiplier")
    p.add_argument("-MaxSleep", type=float, default=0.5, help="max sleep seconds")
    p.add_argument("-LogEvery", type=int, default=5000, help="log every N rows")
    p.add_argument("-NoSleep", action="store_true", help="disable all sleeping")
    args = p.parse_args()

    src = sqlite3.connect(args.Src)
    dst = sqlite3.connect(args.Dst)
    src.row_factory = sqlite3.Row

    symbols = [s.strip() for s in args.Symbols.split(",") if s.strip()]
    if not symbols:
        print("ERROR: empty symbols.")
        sys.exit(1)

    # both source and target are features_stream
    t_source = "features_stream"
    t_target = "features_stream"

    print("===== Running replay_naut.py =====")
    print(f"Replay source: {args.Src}")
    print(f"Replay target: {args.Dst}")
    print(f"Date: {args.Date}")
    print(f"Symbols: {symbols}")
    print(f"Speed: {args.Speed}x")
    if args.NoSleep:
        print("NoSleep mode: sleep disabled")
    print()

    # --- inspect source table ---
    info = list(src.execute(f"PRAGMA table_info({t_source})"))
    if not info:
        print(f"ERROR: table not found: {t_source}")
        sys.exit(1)
    src_cols = [r[1] for r in info]
    types = {r[1]: (r[2] or "").upper() for r in info}

    time_candidates = ("t_exec", "ts", "timestamp", "event_ts", "time", "t_recv")
    time_cols = [c for c in src_cols if c in time_candidates]
    if not time_cols:
        print(f"ERROR: no timestamp column found in {t_source}")
        sys.exit(1)
    tcol = time_cols[0]
    is_numeric_ts = types.get(tcol, "").startswith(("REAL", "INT"))

    # --- target columns must exist in destination ---
    dst_cols = [r[1] for r in dst.execute(f"PRAGMA table_info({t_target})")]
    if not dst_cols:
        print(f"ERROR: target table missing: {t_target}")
        sys.exit(1)

    # --- build SQLs ---
    insert_cols = ", ".join(dst_cols)
    placeholders = ", ".join(["?"] * len(dst_cols))
    ins = f"INSERT INTO {t_target} ({insert_cols}) VALUES ({placeholders})"

    # date filter depends on ts type
    if is_numeric_ts:
        date_filter = f"date(datetime({tcol},'unixepoch','localtime')) = ?"
        date_param = args.Date
    else:
        date_filter = f"{tcol} LIKE ?"
        date_param = f"{args.Date}%"

    sym_clause = " OR ".join(["symbol=?"] * len(symbols))
    sel_sql = f"""
        SELECT * FROM {t_source}
         WHERE {date_filter}
           AND ({sym_clause})
         ORDER BY {tcol} ASC
    """
    sel_params = [date_param] + symbols

    # --- fetch source rows ---
    rows = src.execute(sel_sql, sel_params).fetchall()
    if not rows:
        print("No rows found for that date/symbols.")
        return

    print(f"Starting replay: {len(rows)} rows\n")

    # --- clear existing rows in destination for the same date+symbols ---
    marks = ",".join(["?"] * len(symbols))
    if is_numeric_ts:
        del_where = f"date(datetime(t_exec,'unixepoch','localtime')) = ? AND symbol IN ({marks})"
        del_params = [args.Date] + symbols
    else:
        del_where = f"t_exec LIKE ? AND symbol IN ({marks})"
        del_params = [f"{args.Date}%"] + symbols

    cur = dst.execute(f"DELETE FROM {t_target} WHERE {del_where}", del_params)
    dst.commit()
    print(f"Deleted existing rows in target: {cur.rowcount}")

    # --- replay loop ---
    prev_ts = None
    n = 0
    t0 = time.time()

    for r in rows:
        cur_ts = parse_ts(r[tcol])
        if prev_ts is not None and not args.NoSleep:
            delay = (cur_ts - prev_ts) / max(args.Speed, 0.01)
            if delay > 0:
                time.sleep(min(delay, args.MaxSleep))
        prev_ts = cur_ts

        values = [r[k] if k in r.keys() else None for k in dst_cols]
        dst.execute(ins, values)
        n += 1

        if n % args.LogEvery == 0:
            try:
                human = datetime.fromtimestamp(cur_ts).strftime("%H:%M:%S")
            except Exception:
                human = str(cur_ts)
            print(f"{n}/{len(rows)} rows ({human})")

    dst.commit()
    print(f"\nReplay finished. inserted={n}, elapsed={time.time()-t0:.2f}s")

    src.close()
    dst.close()


if __name__ == "__main__":
    main()
