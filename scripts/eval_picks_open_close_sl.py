import argparse, sys, re, sqlite3
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, List

import pandas as pd
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

DB_TABLE = "daily_bars"
ENCODINGS = ["utf-8-sig", "utf-8", "cp932"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Open→Close 評価（損切り対応版）: picks_* を集計し勝率や曲線を出力"
    )
    p.add_argument("--picks-dir", required=True, type=Path)
    p.add_argument("--db-path", required=True, type=Path)
    p.add_argument("--start", required=True, type=str)
    p.add_argument("--end", required=True, type=str)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--alloc", type=float, default=0.5, help="BUY比率（SELLは1-alloc）")
    p.add_argument("--commission-bps", type=float, default=0.0, help="往復コスト(bps)")
    p.add_argument("--recursive", action="store_true", help="picks-dir を再帰探索")
    p.add_argument("--plot", action="store_true", help="エクイティ曲線PNG保存")
    p.add_argument("--out-dir", type=Path, default=Path("./reports"), help="出力フォルダ（既定: reports）")
    # New: optional stop-loss per side (percent). If not provided or <=0, disabled.
    p.add_argument("--buy-stop-pct", type=float, default=None, help="BUY損切り%（例 2.0 で-2%）")
    p.add_argument("--sell-stop-pct", type=float, default=None, help="SELL損切り%（例 2.0 で+2%不利）")
    return p.parse_args()


def bps_to_frac(bps: float) -> float:
    return bps / 10000.0


def parse_date_from_filename(name: str) -> Optional[date]:
    m = re.search(r"picks_(\d{4}-\d{2}-\d{2})", name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def read_csv_flex(fp: Path) -> Optional[pd.DataFrame]:
    for enc in ENCODINGS:
        try:
            return pd.read_csv(fp, encoding=enc)
        except Exception:
            continue
    print(f"[WARN] CSV読み込み失敗: {fp}")
    return None


def find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in low:
            return cols[low.index(cand.lower())]
    return None


def get_ohlc(conn, ticker: str, d: date) -> Optional[Tuple[float, float, float, float]]:
    q = f"SELECT open, high, low, close FROM {DB_TABLE} WHERE ticker=? AND date=?"
    cur = conn.execute(q, (ticker, d.strftime("%Y-%m-%d")))
    row = cur.fetchone()
    if row:
        o, h, l, c = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        if o > 0 and c > 0:
            return o, h, l, c
    return None


def gather_pick_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in root.rglob("picks_*.csv") if p.is_file()])
    return sorted([p for p in root.glob("picks_*.csv") if p.is_file()])


def main():
    args = parse_args2()
    start_str = args.date_from or args.start
    end_str = args.date_to or args.end
    if not start_str or not end_str:
        print("[ERROR] --from/--to or --start/--end are required")
        sys.exit(2)
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()

    # Map shared SL/TP if provided (fraction -> percent)
    if args.sl is not None:
        try:
            args.buy_stop_pct = float(args.sl) * 100.0
            args.sell_stop_pct = float(args.sl) * 100.0
        except Exception:
            pass
    tp_pct = (float(args.tp) * 100.0) if (hasattr(args, "tp") and args.tp is not None) else None
    if str(getattr(args, "mode", "open_close")).lower() != "open_close":
        print("[WARN] Unsupported --mode; defaulting to open_close")

    # If a single aggregated picks CSV is provided, evaluate it directly
    if args.picks is not None:
        dfp = read_csv_flex(Path(args.picks))
        if dfp is None or dfp.empty:
            print(f"[ERROR] picks CSV is empty or unreadable: {args.picks}")
            sys.exit(1)

        side_col = find_col(dfp.columns.tolist(), ["side", "Side", "サイド", "種別"])
        code_col = find_col(dfp.columns.tolist(), ["code", "ticker", "symbol", "銘柄コード", "銘柄"])
        date_col = find_col(dfp.columns.tolist(), ["date", "Date", "日付"])
        score_col = find_col(dfp.columns.tolist(), ["score", "Score"])
        if side_col is None or code_col is None or date_col is None:
            print("[ERROR] picks CSV must contain date/side/code columns")
            sys.exit(1)

        # Normalize
        dfp[side_col] = dfp[side_col].astype(str).str.upper().str.strip()
        dfp[date_col] = pd.to_datetime(dfp[date_col]).dt.date
        # Filter by range
        dfp = dfp[(dfp[date_col] >= start) & (dfp[date_col] <= end)].copy()
        # Per-date selection: top by score if available
        records = []  # (date, buy_code, sell_code)
        for d, g in dfp.groupby(date_col):
            buy_code = None
            sell_code = None
            if score_col is not None and score_col in g.columns:
                gb = g[g[side_col] == "BUY"].sort_values(score_col, ascending=False)
                gs = g[g[side_col] == "SELL"].sort_values(score_col, ascending=False)
            else:
                gb = g[g[side_col] == "BUY"]
                gs = g[g[side_col] == "SELL"]
            if not gb.empty:
                buy_code = str(gb[code_col].iloc[0]).strip()
            if not gs.empty:
                sell_code = str(gs[code_col].iloc[0]).strip()
            records.append((d, buy_code, sell_code))

        # Evaluate
        conn = sqlite3.connect(args.db_path)
        comm = bps_to_frac(args.commission_bps)
        rows, missing = [], []
        for d, buy_code, sell_code in sorted(records, key=lambda x: x[0]):
            buy_open = buy_close = buy_ret = None
            sell_open = sell_close = sell_ret = None

            if buy_code:
                ohlc = get_ohlc(conn, buy_code, d)
                if ohlc:
                    o, h, l, c = ohlc
                    buy_open, buy_close = o, c
                    # BUY: stop first; else take-profit; else close
                    if (args.buy_stop_pct is not None) and (args.buy_stop_pct > 0) and (l <= o * (1.0 - args.buy_stop_pct / 100.0)):
                        stop_price = o * (1.0 - args.buy_stop_pct / 100.0)
                        buy_ret = (stop_price - o) / o - 2 * comm
                    elif (tp_pct is not None) and (tp_pct > 0) and (h >= o * (1.0 + tp_pct / 100.0)):
                        tp_price = o * (1.0 + tp_pct / 100.0)
                        buy_ret = (tp_price - o) / o - 2 * comm
                    else:
                        buy_ret = (c - o) / o - 2 * comm
                else:
                    missing.append({"date": d.strftime("%Y-%m-%d"), "side": "BUY", "ticker": buy_code})

            if sell_code:
                ohlc = get_ohlc(conn, sell_code, d)
                if ohlc:
                    o, h, l, c = ohlc
                    sell_open, sell_close = o, c
                    # SELL: stop first; else take-profit; else close
                    if (args.sell_stop_pct is not None) and (args.sell_stop_pct > 0) and (h >= o * (1.0 + args.sell_stop_pct / 100.0)):
                        stop_price = o * (1.0 + args.sell_stop_pct / 100.0)
                        sell_ret = (o - stop_price) / o - 2 * comm
                    elif (tp_pct is not None) and (tp_pct > 0) and (l <= o * (1.0 - tp_pct / 100.0)):
                        tp_price = o * (1.0 - tp_pct / 100.0)
                        sell_ret = (o - tp_price) / o - 2 * comm
                    else:
                        sell_ret = (o - c) / o - 2 * comm
                else:
                    missing.append({"date": d.strftime("%Y-%m-%d"), "side": "SELL", "ticker": sell_code})

            port_ret = None
            if (buy_ret is not None) or (sell_ret is not None):
                br = buy_ret if buy_ret is not None else 0.0
                sr = sell_ret if sell_ret is not None else 0.0
                port_ret = args.alloc * br + (1 - args.alloc) * sr

            rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "buy_code": buy_code,
                    "buy_open": buy_open,
                    "buy_close": buy_close,
                    "buy_ret_pct": None if buy_ret is None else round(buy_ret * 100, 3),
                    "sell_code": sell_code,
                    "sell_open": sell_open,
                    "sell_close": sell_close,
                    "sell_ret_pct": None if sell_ret is None else round(sell_ret * 100, 3),
                    "portfolio_ret_pct": (None if port_ret is None else round(port_ret * 100, 3)),
                    "portfolio_pnl_yen": (None if port_ret is None else round(args.capital * port_ret)),
                }
            )

        # Output detailed
        out_dir = args.out_dir if isinstance(args.out_dir, Path) else Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        res = pd.DataFrame(rows).sort_values("date")
        out_csv = out_dir / "picks_eval_open_close_sl.csv"
        res.to_csv(out_csv, index=False, encoding="utf-8-sig")
        if missing:
            miss_df = pd.DataFrame(missing)
            miss_df.sort_values(["date","side","ticker"]).to_csv(out_dir / "picks_eval_missing_prices_sl.csv", index=False, encoding="utf-8-sig")

        # Metrics
        port_series = (res["portfolio_ret_pct"] / 100.0).astype(float)
        wins = int(((res["buy_ret_pct"].fillna(-1) > 0).sum()) + ((res["sell_ret_pct"].fillna(-1) > 0).sum()))
        trades = int(res["buy_ret_pct"].notna().sum() + res["sell_ret_pct"].notna().sum())
        avg_ret = float(port_series.mean()) if len(port_series) > 0 else 0.0
        pos = port_series[port_series > 0]
        neg = -port_series[port_series < 0]
        rr = (float(pos.mean()) / float(neg.mean())) if (len(pos) > 0 and len(neg) > 0 and float(neg.mean()) > 0) else None

        # 1-line metrics CSV
        metrics_path = args.out if args.out is not None else (out_dir / "metrics_summary.csv")
        mrow = {
            "WinRate": (wins / trades) if trades > 0 else 0.0,
            "AvgReturn": avg_ret,
            "RewardRisk": ("" if rr is None else rr),
            "Trades": trades,
        }
        pd.DataFrame([mrow]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

        # Equity curve
        equity = (1.0 + port_series.fillna(0)).cumprod() * args.capital
        eq_df = pd.DataFrame({"date": res["date"], "equity": equity})
        eq_df.to_csv(out_dir / "picks_eval_equity_curve_sl.csv", index=False, encoding="utf-8-sig")
        if args.plot and not eq_df.empty:
            plt.figure(); plt.plot(pd.to_datetime(eq_df["date"]), eq_df["equity"]); plt.tight_layout(); plt.savefig(out_dir / "picks_eval_equity_curve_sl.png", dpi=150)

        # Summary print
        print(f"[SUMMARY] WinRate={mrow['WinRate']:.2%}, AvgReturn={mrow['AvgReturn']:.4f}, RewardRisk={mrow['RewardRisk']}, Trades={mrow['Trades']}")
        return

    files = gather_pick_files(args.picks_dir, args.recursive)
    if not files:
        print(f"[ERROR] picks_*.csv が見つかりません: {args.picks_dir.resolve()}")
        sys.exit(1)

    dated = []
    for f in files:
        d = parse_date_from_filename(f.name)
        if d:
            dated.append((d, f))
        else:
            print(f"[INFO] 日付不明のためスキップ: {f.name}")
    in_range = [(d, f) for d, f in dated if start <= d <= end]
    print(f"[INFO] 期間 {start}～{end}: {len(in_range)} 件 / 総ファイル: {len(files)}")

    conn = sqlite3.connect(args.db_path)
    comm = bps_to_frac(args.commission_bps)

    rows, missing = [], []
    buy_wins = sell_wins = buy_n = sell_n = 0

    for d, fp in in_range:
        df = read_csv_flex(fp)
        if df is None or df.empty:
            continue

        side_col = find_col(df.columns.tolist(), ["side", "サイド", "種別"])
        code_col = find_col(df.columns.tolist(), ["code", "ticker", "symbol", "銘柄コード", "銘柄"])
        if side_col is None or code_col is None:
            print(f"[WARN] 列検出失敗: {fp}")
            continue

        df[side_col] = df[side_col].astype(str).str.upper().str.strip()
        buy_row = df[df[side_col] == "BUY"].head(1)
        sell_row = df[df[side_col] == "SELL"].head(1)
        buy_code = str(buy_row[code_col].iloc[0]).strip() if not buy_row.empty else None
        sell_code = str(sell_row[code_col].iloc[0]).strip() if not sell_row.empty else None

        buy_open = buy_close = buy_ret = None
        sell_open = sell_close = sell_ret = None

        if buy_code:
            ohlc = get_ohlc(conn, buy_code, d)
            if ohlc:
                o, h, l, c = ohlc
                buy_open, buy_close = o, c
                if (args.buy_stop_pct is not None) and (args.buy_stop_pct > 0):
                    stop_price = o * (1.0 - args.buy_stop_pct / 100.0)
                    if l <= stop_price:
                        buy_ret = (stop_price - o) / o - 2 * comm
                    elif (tp_pct is not None) and (tp_pct > 0) and (h >= o * (1.0 + tp_pct / 100.0)):
                        tp_price = o * (1.0 + tp_pct / 100.0)
                        buy_ret = (tp_price - o) / o - 2 * comm
                    else:
                        buy_ret = (c - o) / o - 2 * comm
                else:
                    if (tp_pct is not None) and (tp_pct > 0) and (h >= o * (1.0 + tp_pct / 100.0)):
                        tp_price = o * (1.0 + tp_pct / 100.0)
                        buy_ret = (tp_price - o) / o - 2 * comm
                    else:
                        buy_ret = (c - o) / o - 2 * comm
                buy_n += 1
                buy_wins += int(buy_ret > 0)
            else:
                missing.append({"date": d.strftime("%Y-%m-%d"), "side": "BUY", "ticker": buy_code})

        if sell_code:
            ohlc = get_ohlc(conn, sell_code, d)
            if ohlc:
                o, h, l, c = ohlc
                sell_open, sell_close = o, c
                if (args.sell_stop_pct is not None) and (args.sell_stop_pct > 0):
                    stop_price = o * (1.0 + args.sell_stop_pct / 100.0)
                    if h >= stop_price:
                        sell_ret = (o - stop_price) / o - 2 * comm
                    elif (tp_pct is not None) and (tp_pct > 0) and (l <= o * (1.0 - tp_pct / 100.0)):
                        tp_price = o * (1.0 - tp_pct / 100.0)
                        sell_ret = (o - tp_price) / o - 2 * comm
                    else:
                        sell_ret = (o - c) / o - 2 * comm
                else:
                    if (tp_pct is not None) and (tp_pct > 0) and (l <= o * (1.0 - tp_pct / 100.0)):
                        tp_price = o * (1.0 - tp_pct / 100.0)
                        sell_ret = (o - tp_price) / o - 2 * comm
                    else:
                        sell_ret = (o - c) / o - 2 * comm
                sell_n += 1
                sell_wins += int(sell_ret > 0)
            else:
                missing.append({"date": d.strftime("%Y-%m-%d"), "side": "SELL", "ticker": sell_code})

        port_ret = None
        if (buy_ret is not None) or (sell_ret is not None):
            br = buy_ret if buy_ret is not None else 0.0
            sr = sell_ret if sell_ret is not None else 0.0
            port_ret = args.alloc * br + (1 - args.alloc) * sr

        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "buy_code": buy_code,
                "buy_open": buy_open,
                "buy_close": buy_close,
                "buy_ret_pct": None if buy_ret is None else round(buy_ret * 100, 3),
                "sell_code": sell_code,
                "sell_open": sell_open,
                "sell_close": sell_close,
                "sell_ret_pct": None if sell_ret is None else round(sell_ret * 100, 3),
                "portfolio_ret_pct": (None if port_ret is None else round(port_ret * 100, 3)),
                "portfolio_pnl_yen": (None if port_ret is None else round(args.capital * port_ret)),
            }
        )

    # === 出力 ===
    out_dir = args.out_dir if isinstance(args.out_dir, Path) else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows).sort_values("date")
    out_csv = out_dir / "picks_eval_open_close_sl.csv"
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")

    miss_df = pd.DataFrame(missing)
    if not miss_df.empty:
        miss_csv = out_dir / "picks_eval_missing_prices_sl.csv"
        miss_df.sort_values(["date", "side", "ticker"]).to_csv(miss_csv, index=False, encoding="utf-8-sig")

    # Summary metrics
    def compound(series_pct: pd.Series) -> float:
        s = (series_pct.dropna() / 100.0) + 1.0
        return float(s.prod() - 1.0)

    def max_drawdown(series_pct: pd.Series) -> float:
        eq = (1.0 + (series_pct.fillna(0) / 100.0)).cumprod()
        peak = eq.cummax()
        dd = (eq - peak) / peak
        return float(dd.min())

    buy_series = (res["buy_ret_pct"] / 100.0).astype(float)
    sell_series = (res["sell_ret_pct"] / 100.0).astype(float)
    port_series = (res["portfolio_ret_pct"] / 100.0).astype(float)

    buy_total = compound(res["buy_ret_pct"]) if "buy_ret_pct" in res else 0.0
    sell_total = compound(res["sell_ret_pct"]) if "sell_ret_pct" in res else 0.0
    port_total = compound(res["portfolio_ret_pct"]) if "portfolio_ret_pct" in res else 0.0
    port_mdd = max_drawdown(res["portfolio_ret_pct"]) if "portfolio_ret_pct" in res else 0.0

    # Equity curve (optional plot)
    equity = (1.0 + port_series.fillna(0)).cumprod() * args.capital
    eq_df = pd.DataFrame({"date": res["date"], "equity": equity})
    eq_df.to_csv(out_dir / "picks_eval_equity_curve_sl.csv", index=False, encoding="utf-8-sig")

    if args.plot and not eq_df.empty:
        plt.figure()
        plt.plot(pd.to_datetime(eq_df["date"]), eq_df["equity"])
        plt.title("Equity Curve (Open->Close with Stop)")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(out_dir / "picks_eval_equity_curve_sl.png", dpi=150)

    # Print summary
    buy_n = int(res["buy_ret_pct"].notna().sum())
    sell_n = int(res["sell_ret_pct"].notna().sum())
    buy_wins = int((res["buy_ret_pct"].fillna(-1) > 0).sum())
    sell_wins = int((res["sell_ret_pct"].fillna(-1) > 0).sum())

    lines = []
    lines.append("\n=== 評価サマリー（Open→Close, Stop対応） ===")
    lines.append(f"期間: {args.start}～{args.end}")
    if buy_n > 0:
        lines.append(f"[BUY ] 件数:{buy_n} 勝率:{buy_wins/max(1,buy_n):.1%} 累積収益:{buy_total:.2%}")
    if sell_n > 0:
        lines.append(f"[SELL] 件数:{sell_n} 勝率:{sell_wins/max(1,sell_n):.1%} 累積収益:{sell_total:.2%}")
    lines.append(f"[PORT] 日次平均:{port_series.mean():.3%} 累積:{port_total:.2%} 最大DD:{port_mdd:.2%}")
    lines.append(f"\n出力CSV: {out_csv.resolve()}")

    text = "\n".join(lines)
    print(text)
    (out_dir / "picks_eval_summary_sl.txt").write_text(text, encoding="utf-8")

    # One-line metrics CSV if requested
    if args.out is not None:
        wins = int(buy_wins + sell_wins)
        trades = int(buy_n + sell_n)
        avg_ret = float(port_series.mean()) if len(port_series) > 0 else 0.0
        pos = (port_series[port_series > 0])
        neg = -(port_series[port_series < 0])
        rr = (float(pos.mean()) / float(neg.mean())) if (len(pos) > 0 and len(neg) > 0 and float(neg.mean()) > 0) else None
        mrow = {
            "WinRate": (wins / trades) if trades > 0 else 0.0,
            "AvgReturn": avg_ret,
            "RewardRisk": ("" if rr is None else rr),
            "Trades": trades,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([mrow]).to_csv(args.out, index=False, encoding="utf-8-sig")


def parse_args2():
    p = argparse.ArgumentParser(
        description="Open->Close evaluator with SL/TP and metrics CSV"
    )
    p.add_argument("--picks-dir", required=False, default=None, type=Path)
    p.add_argument("--picks", required=False, default=None, type=Path, help="Single picks CSV (expects date/side/code/score)")
    p.add_argument("--db-path", required=False, default=Path("./rss_daily.db"), type=Path)
    p.add_argument("--start", required=False, default=None, type=str)
    p.add_argument("--end", required=False, default=None, type=str)
    p.add_argument("--from", dest="date_from", required=False, default=None, type=str)
    p.add_argument("--to", dest="date_to", required=False, default=None, type=str)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--alloc", type=float, default=0.5)
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("./reports"))
    p.add_argument("--out", type=Path, required=False, default=None)
    p.add_argument("--mode", type=str, default="open_close")
    p.add_argument("--buy-stop-pct", type=float, default=None)
    p.add_argument("--sell-stop-pct", type=float, default=None)
    p.add_argument("--sl", type=float, default=None)
    p.add_argument("--tp", type=float, default=None)
    return p.parse_args()

if __name__ == "__main__":
    main()
