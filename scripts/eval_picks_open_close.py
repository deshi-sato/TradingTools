import argparse, sys, re, sqlite3
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict

import pandas as pd
import matplotlib.pyplot as plt

DB_TABLE = "daily_bars"
ENCODINGS = ["utf-8-sig", "utf-8", "cp932"]


def parse_args():
    p = argparse.ArgumentParser(
        description="寄り→大引け バックテスト（実運用の picks_* を評価）"
    )
    p.add_argument("--picks-dir", required=True, type=Path)
    p.add_argument("--db-path", required=True, type=Path)
    p.add_argument("--start", required=True, type=str)
    p.add_argument("--end", required=True, type=str)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--alloc", type=float, default=0.5, help="BUY比率（SELLは1-alloc）")
    p.add_argument("--commission-bps", type=float, default=0.0, help="往復手数料(bps)")
    p.add_argument("--recursive", action="store_true", help="picks-dir を再帰検索")
    p.add_argument("--plot", action="store_true", help="エクイティカーブPNG保存")
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
    print(f"[WARN] CSV読込失敗: {fp}")
    return None


def find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in low:
            return cols[low.index(cand.lower())]
    return None


def get_price(conn, ticker: str, d: date) -> Optional[Tuple[float, float]]:
    q = f"SELECT open, close FROM {DB_TABLE} WHERE ticker=? AND date=?"
    cur = conn.execute(q, (ticker, d.strftime("%Y-%m-%d")))
    row = cur.fetchone()
    if row:
        o, c = float(row[0]), float(row[1])
        if o > 0 and c > 0:
            return o, c
    return None


def gather_pick_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in root.rglob("picks_*.csv") if p.is_file()])
    return sorted([p for p in root.glob("picks_*.csv") if p.is_file()])


def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

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
            print(f"[INFO] 日付抽出できずスキップ: {f.name}")
    in_range = [(d, f) for d, f in dated if start <= d <= end]
    print(
        f"[INFO] 期間内 {start}〜{end}: {len(in_range)} 件 / 総ファイル: {len(files)}"
    )

    conn = sqlite3.connect(args.db_path)
    comm = bps_to_frac(args.commission_bps)

    rows, missing = [], []  # 明細 / 価格欠損
    buy_wins = sell_wins = buy_n = sell_n = 0

    for d, fp in in_range:
        df = read_csv_flex(fp)
        if df is None or df.empty:
            continue

        side_col = find_col(df.columns.tolist(), ["side", "サイド", "方向"])
        code_col = find_col(
            df.columns.tolist(), ["code", "ticker", "symbol", "銘柄コード", "銘柄"]
        )
        if side_col is None or code_col is None:
            print(f"[WARN] 列特定不可: {fp}")
            continue

        df[side_col] = df[side_col].astype(str).str.upper().str.strip()
        buy_row = df[df[side_col] == "BUY"].head(1)
        sell_row = df[df[side_col] == "SELL"].head(1)
        buy_code = str(buy_row[code_col].iloc[0]).strip() if not buy_row.empty else None
        sell_code = (
            str(sell_row[code_col].iloc[0]).strip() if not sell_row.empty else None
        )

        buy_open = buy_close = buy_ret = None
        sell_open = sell_close = sell_ret = None

        if buy_code:
            ochl = get_price(conn, buy_code, d)
            if ochl:
                buy_open, buy_close = ochl
                buy_ret = (buy_close - buy_open) / buy_open - 2 * comm
                buy_n += 1
                buy_wins += int(buy_ret > 0)
            else:
                missing.append(
                    {"date": d.strftime("%Y-%m-%d"), "side": "BUY", "ticker": buy_code}
                )

        if sell_code:
            ochl = get_price(conn, sell_code, d)
            if ochl:
                sell_open, sell_close = ochl
                sell_ret = (sell_open - sell_close) / sell_open - 2 * comm
                sell_n += 1
                sell_wins += int(sell_ret > 0)
            else:
                missing.append(
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "side": "SELL",
                        "ticker": sell_code,
                    }
                )

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
                "portfolio_ret_pct": (
                    None if port_ret is None else round(port_ret * 100, 3)
                ),
                "portfolio_pnl_yen": (
                    None if port_ret is None else round(args.capital * port_ret)
                ),
            }
        )

    # === 出力 ===
    reports = Path("./reports")
    reports.mkdir(exist_ok=True)
    res = pd.DataFrame(rows).sort_values("date")
    out_csv = reports / "picks_eval_open_close.csv"
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")

    miss_df = pd.DataFrame(missing)
    if not miss_df.empty:
        miss_csv = reports / "picks_eval_missing_prices.csv"
        miss_df.sort_values(["date", "side", "ticker"]).to_csv(
            miss_csv, index=False, encoding="utf-8-sig"
        )

    # サマリー
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
    port_total = (
        compound(res["portfolio_ret_pct"]) if "portfolio_ret_pct" in res else 0.0
    )
    port_mdd = (
        max_drawdown(res["portfolio_ret_pct"]) if "portfolio_ret_pct" in res else 0.0
    )

    # エクイティ
    equity = (1.0 + port_series.fillna(0)).cumprod() * args.capital
    eq_df = pd.DataFrame({"date": res["date"], "equity": equity})
    eq_df.to_csv(
        reports / "picks_eval_equity_curve.csv", index=False, encoding="utf-8-sig"
    )

    # PNG（任意）
    if args.plot and not eq_df.empty:
        plt.figure()
        plt.plot(pd.to_datetime(eq_df["date"]), eq_df["equity"])
        plt.title("Equity Curve (Open->Close)")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(reports / "picks_eval_equity_curve.png", dpi=150)

    # 表示 & TXT
    lines = []
    lines.append("\n=== 評価サマリー（寄り→大引け） ===")
    lines.append(f"期間: {args.start}〜{args.end}")
    lines.append(f"対象日: {len(res)}（有効ポート: {port_series.notna().sum()}）")
    if buy_n > 0:
        lines.append(
            f"[BUY ] 件数:{buy_n} 勝率:{buy_wins/max(1,buy_n):.1%} 合成収益率:{buy_total:.2%}"
        )
    if sell_n > 0:
        lines.append(
            f"[SELL] 件数:{sell_n} 勝率:{sell_wins/max(1,sell_n):.1%} 合成収益率:{sell_total:.2%}"
        )
    lines.append(
        f"[PORT] 日次平均:{port_series.mean():.3%} 合成収益率:{port_total:.2%} 最大DD:{port_mdd:.2%}"
    )
    lines.append(f"\n明細CSV: {out_csv.resolve()}")
    if not miss_df.empty:
        lines.append(f"欠損一覧: {(reports/'picks_eval_missing_prices.csv').resolve()}")

    text = "\n".join(lines)
    print(text)
    (reports / "picks_eval_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
