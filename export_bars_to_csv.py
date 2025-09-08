# export_bars_to_csv.py
#!/usr/bin/env python3
"""
SQLiteの任意テーブルから、timestamp,open,high,low,close,volume,symbol 列でCSVを書き出す汎用ツール。
- 例：当日1分足（rss_snapshot.db / today_data）→ dataset_1min/*.csv
- 例：日足（rss_daily.db / daily_bars）→ dataset_daily/*.csv

使い方例は下に記載。
"""
from __future__ import annotations
import argparse, os, sys, csv, sqlite3
from datetime import datetime, timezone
from typing import List, Dict


def ensure_iso8601(ts: object, is_date: bool = False) -> str:
    if ts is None:
        return ""
    if isinstance(ts, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            return dt.isoformat(timespec="seconds")
        except Exception:
            return str(ts)
    s = str(ts).strip()
    if not s:
        return s
    # 既にISO8601っぽい場合の軽整形
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        return dt.isoformat(timespec="seconds")
    except Exception:
        pass
    # 代表的なフォーマット
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            if is_date and len(s) <= 10:
                # 日付だけなら 00:00:00 を付与
                return dt.replace(hour=0, minute=0, second=0).isoformat(
                    timespec="seconds"
                )
            return dt.isoformat(timespec="seconds")
        except Exception:
            continue
    # 最後の手段：空白をTに置換
    return s.replace(" ", "T")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite DB パス")
    ap.add_argument(
        "--table",
        required=True,
        help="読み出しテーブル名（例: today_data / daily_bars）",
    )
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    ap.add_argument("--start", required=True, help="開始（含む）")
    ap.add_argument("--end", required=True, help="終了（含む）")
    ap.add_argument(
        "--symbols", default="", help="カンマ区切りの銘柄絞り込み（例: 7011,5803）"
    )
    # カラム名（既定は一般的想定）
    ap.add_argument("--time-col", default="datetime")
    ap.add_argument("--symbol-col", default="ticker")
    ap.add_argument("--open-col", default="open")
    ap.add_argument("--high-col", default="high")
    ap.add_argument("--low-col", default="low")
    ap.add_argument("--close-col", default="close")
    ap.add_argument("--vol-col", default="volume")
    # time_col が「日付のみ」かどうか（daily_barsなど）
    ap.add_argument(
        "--time-is-date",
        action="store_true",
        help="time-colが日付のみの場合に指定（00:00:00を補う）",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.db):
        print(f"[ERROR] DBが見つかりません: {args.db}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cols = {
        "t": args.time_col,
        "s": args.symbol_col,
        "o": args.open_col,
        "h": args.high_col,
        "l": args.low_col,
        "c": args.close_col,
        "v": args.vol_col,
    }

    sql = f"""
      SELECT {cols['s']} AS symbol,
             {cols['t']} AS t,
             {cols['o']} AS o,
             {cols['h']} AS h,
             {cols['l']} AS l,
             {cols['c']} AS c,
             {cols['v']} AS v
      FROM {args.table}
      WHERE 1=1
        AND {cols['t']} >= ? AND {cols['t']} <= ?
    """
    params: List[object] = [args.start, args.end]

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if syms:
        q = ",".join(["?"] * len(syms))
        sql += f" AND {cols['s']} IN ({q})"
        params += syms

    sql += f" ORDER BY {cols['s']} ASC, {cols['t']} ASC"

    try:
        cur.execute(sql, params)
    except sqlite3.OperationalError as e:
        print(
            f"[ERROR] SQL失敗: {e}\n- テーブル/カラム名や日時の型を確認してください。",
            file=sys.stderr,
        )
        sys.exit(3)

    writers: Dict[str, tuple[csv.writer, any]] = {}
    written = 0
    try:
        for row in cur:
            sym = str(row["symbol"]) if row["symbol"] is not None else ""
            ts = ensure_iso8601(row["t"], is_date=args.time_is_date)
            out_path = os.path.join(args.out, f"{sym}.csv")
            if sym not in writers:
                f = open(out_path, "w", newline="", encoding="utf-8")
                w = csv.writer(f)
                w.writerow(
                    ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
                )
                writers[sym] = (w, f)
            w, f = writers[sym]
            w.writerow([ts, row["o"], row["h"], row["l"], row["c"], row["v"], sym])
            written += 1
    finally:
        for _w, f in writers.values():
            try:
                f.flush()
            except Exception:
                pass
            f.close()
        cur.close()
        conn.close()

    if written == 0:
        print(
            "[ERROR] 条件に合致する行がありません（symbols / 期間の見直しを）",
            file=sys.stderr,
        )
        sys.exit(4)
    print(f"Export completed: {written} rows -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[ERROR] 中断されました", file=sys.stderr)
        sys.exit(130)
