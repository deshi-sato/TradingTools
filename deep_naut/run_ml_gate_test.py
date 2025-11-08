import argparse, sqlite3, datetime
from typing import Optional
from deep_naut.ml_gate import MlGateConfig, MlBreakoutGate

JST = datetime.timezone(datetime.timedelta(hours=9))

def detect_unit(cur, symbol:str):
    mn = cur.execute("SELECT MIN(t_exec) FROM features_stream WHERE symbol=?", (symbol,)).fetchone()[0]
    # 1e12 以上ならミリ秒、1e9 以上なら秒とみなす
    if mn is None: 
        return ("ms", 1000)  # デフォルト
    if mn >= 1_000_000_000_000:
        return ("ms", 1000)
    elif mn >= 1_000_000_000:
        return ("s", 1)
    else:
        return ("s", 1)

def tsfmt(v:int, unit:str) -> str:
    # unit=="ms" のときはそのまま、"s" のときは *1000 にして表示
    ms = v if unit=="ms" else v*1000
    return datetime.datetime.fromtimestamp(ms/1000, JST).strftime("%H:%M:%S.%f")[:-3]

def parse_hhmm_range(hhmm:str, base_date:datetime.date, unit:str):
    a,b = hhmm.split("-")
    h0,m0 = map(int, a.split(":"))
    h1,m1 = map(int, b.split(":"))
    t0 = int(datetime.datetime(base_date.year, base_date.month, base_date.day, h0, m0, tzinfo=JST).timestamp() * (1000 if unit=="ms" else 1))
    t1 = int(datetime.datetime(base_date.year, base_date.month, base_date.day, h1, m1, tzinfo=JST).timestamp() * (1000 if unit=="ms" else 1))
    return t0, t1

def pick_day_bounds(cur, symbol:str, unit:str):
    mn, mx = cur.execute("SELECT MIN(t_exec), MAX(t_exec) FROM features_stream WHERE symbol=?", (symbol,)).fetchone()
    if mn is None or mx is None:
        raise SystemExit(f"no features_stream rows for symbol={symbol}")
    # 最初の行のJST日付を基準日とする
    first = mn if unit=="ms" else mn*1000
    d = datetime.datetime.fromtimestamp(first/1000, JST).date()
    return mn, mx, d

def parse_time_range(hhmm:str, base_date:datetime.date, unit:str):
    # HH:MM または HH:MM:SS を許容
    def parse_one(s:str) -> int:
        parts = s.strip().split(":")
        if len(parts) == 2:
            h, m = map(int, parts); sec = 0
        elif len(parts) == 3:
            h, m, sec = map(int, parts)
        else:
            raise ValueError("time must be HH:MM or HH:MM:SS")
        dt = datetime.datetime(base_date.year, base_date.month, base_date.day, h, m, sec, tzinfo=JST)
        return int(dt.timestamp() * (1000 if unit=="ms" else 1))
    a, b = hhmm.split("-")
    return parse_one(a), parse_one(b)

def main():
    ap = argparse.ArgumentParser(description="Replay ML breakout gate on DB window.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--window", help="HH:MM-HH:MM in JST")
    ap.add_argument("--start-ms", type=int)
    ap.add_argument("--end-ms", type=int)

    # gate knobs
    ap.add_argument("--prob-up-len", type=int, default=3)
    ap.add_argument("--vol-ma3-thr", type=float, default=700.0)
    ap.add_argument("--vol-rate-thr", type=float, default=1.30)
    ap.add_argument("--vol-gate", choices=["OR","AND"], default="OR")
    ap.add_argument("--sync-ticks", type=int, default=3)
    ap.add_argument("--cooldown-ms", type=int, default=1500)
    ap.add_argument("--head", type=int, default=10)

    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 秒/ミリ秒の自動判定
    unit, _ = detect_unit(cur, args.symbol)

    # 列構成
    cur.execute("PRAGMA table_info(features_stream)")
    cols = {r[1] for r in cur.fetchall()}
    has_v3 = "vol_ma3" in cols
    has_vr = "vol_rate" in cols

    # ウィンドウ決定
    mn, mx, day = pick_day_bounds(cur, args.symbol, unit)
    if args.window:
        t0, t1 = parse_time_range(args.window, day, unit)
    elif args.start_ms and args.end_ms:
        # 明示ミリ秒指定 → unitが秒なら秒に落とす
        if unit == "s":
            t0, t1 = args.start_ms // 1000, args.end_ms // 1000
        else:
            t0, t1 = args.start_ms, args.end_ms
    else:
        t0, t1 = mn, mx  # 全域

    # クエリ（ml_probは LEFT JOIN）
    q = f"""
    SELECT
      f.t_exec,
      { 'f.vol_ma3'  if has_v3 else 'NULL' } AS vol_ma3,
      f.candle_up,
      { 'f.vol_rate' if has_vr else 'NULL' } AS vol_rate,
      p.prob AS pstar
    FROM features_stream f
    LEFT JOIN ml_prob p ON p.symbol=f.symbol AND p.t_exec=f.t_exec
    WHERE f.symbol=? AND f.t_exec BETWEEN ? AND ?
    ORDER BY f.t_exec
    """
    rows = list(cur.execute(q, (args.symbol, t0, t1)))
    con.close()

    cfg = MlGateConfig(
        prob_up_len=args.prob_up_len,
        vol_ma3_thr=args.vol_ma3_thr,
        vol_rate_thr=args.vol_rate_thr,
        use_and=(args.vol_gate=="AND"),
        sync_ticks=args.sync_ticks,
        cooldown_ms=args.cooldown_ms,
    )
    gate = MlBreakoutGate(cfg)

    hits = []
    for i,r in enumerate(rows,1):
        feat = {
            "idx": i,
            "t_exec": r["t_exec"],
            "candle_up": r["candle_up"],
            "vol_ma3": r["vol_ma3"],
            "vol_rate": r["vol_rate"],
            "pstar": r["pstar"],
        }
        h = gate.check(feat)
        if h: hits.append(h)

    print(f"window={tsfmt(t0,unit)}→{tsfmt(t1,unit)}  ticks={len(rows)}")
    print(f"gate: prob↑{cfg.prob_up_len}, vol_ma3>={cfg.vol_ma3_thr}, vol_rate>={cfg.vol_rate_thr} "
          f"{'AND' if cfg.use_and else 'OR'}, sync≤{cfg.sync_ticks}, cooldown={cfg.cooldown_ms}ms")
    print(f"hits={len(hits)}")
    for h in hits[:args.head]:
        vr = "NA" if h['vol_rate'] is None else f"{h['vol_rate']:.2f}"
        v3 = 0 if h['vol_ma3'] is None else int(h['vol_ma3'])
        print(f"{tsfmt(h['t_exec'],unit)}  p*={h['pstar']:.3f}  v3={v3}  vr={vr}  {h['reason']}")

if __name__ == "__main__":
    main()


