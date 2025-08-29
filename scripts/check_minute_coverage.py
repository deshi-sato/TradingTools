# -*- coding: utf-8 -*-
import argparse, sqlite3
from pathlib import Path
import pandas as pd

def ticker_candidates(ticker: str):
    import re
    m = re.match(r"^(\d{4})", str(ticker))
    code4 = m.group(1) if m else str(ticker)
    return code4 + ".T", code4

def load_watchlist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "ticker" in df.columns, "watchlist に 'ticker' 列が必要です"
    if "side" not in df.columns:
        df["side"] = "BUY"
    df["ticker"] = df["ticker"].astype(str)
    return df[["ticker", "side"]].drop_duplicates()


def get_cov_for_ticker(con, ticker: str, date_str: str):
    t_with, t_plain = ticker_candidates(ticker)
    q = """
    SELECT COUNT(*) as rows,
           MIN(time(datetime)) as first_time,
           MAX(time(datetime)) as last_time
    FROM minute_data
    WHERE (ticker = ? OR ticker = ?) AND substr(datetime,1,10) = ?
    """
    row = pd.read_sql(q, con, params=[t_with, t_plain, date_str]).iloc[0]
    return int(row["rows"]), row["first_time"], row["last_time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--watchlist", required=True)
    ap.add_argument("--minutedb", required=True)
    ap.add_argument("-d", "--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument(
        "--out_filtered",
        default=None,
        help="データがある銘柄だけの watchlist を出力（任意）",
    )
    args = ap.parse_args()

    wl = load_watchlist(Path(args.watchlist))
    con = sqlite3.connect(args.minutedb)

    rows = []
    for _, r in wl.iterrows():
        rows_count, first_t, last_t = get_cov_for_ticker(con, r["ticker"], args.date)
        rows.append(
            {
                "ticker": r["ticker"],
                "side": r["side"],
                "date": args.date,
                "rows_count": rows_count,
                "first_time": first_t,
                "last_time": last_t,
            }
        )
    con.close()

    df = pd.DataFrame(rows).sort_values(
        ["rows_count", "ticker"], ascending=[False, True]
    )
    out = Path(args.out or f"data/analysis/minute_coverage_{args.date}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"✅ Wrote: {out.resolve()} (rows={len(df)})")

    # データがある銘柄だけでフィルタ済みウォッチリストを出力（任意）
    if args.out_filtered:
        df2 = df[df["rows_count"] > 0][["ticker", "side"]]
        Path(args.out_filtered).parent.mkdir(parents=True, exist_ok=True)
        df2.to_csv(args.out_filtered, index=False, encoding="utf-8")
        print(
            f"✅ Wrote filtered watchlist: {Path(args.out_filtered).resolve()} (rows={len(df2)})"
        )


if __name__ == "__main__":
    main()
