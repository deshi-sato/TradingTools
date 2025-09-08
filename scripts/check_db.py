\"\"\"check_db.py : Auto-generated placeholder

- file: scripts/check_db.py
- updated: 2025-09-08

TODO: このモジュールの概要をここに書いてください。
\"\"\"
# check_db.py
# data/rss_daily.db を健全性チェックして要約を表示・CSV出力
import os, sqlite3, csv

DB = r"data/rss_daily.db"
CODES_TXT = r"data/topix100_codes.txt"  # あれば .T を付けて突合します


def q(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    return cols, rows


def print_rows(title, cols, rows, limit=10):
    print(f"\n--- {title} ---")
    if not rows:
        print("(no rows)")
        return
    print(" | ".join(cols))
    for r in rows[:limit]:
        print(" | ".join(str(x) if x is not None else "" for x in r))
    if len(rows) > limit:
        print(f"... ({len(rows)-limit} more)")


def main():
    if not os.path.exists(DB):
        print(f"[ERROR] DB not found: {DB}")
        return

    conn = sqlite3.connect(DB)

    # 1) daily_bars の存在とスキーマ
    cols, rows = q(
        conn,
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name='daily_bars';",
    )
    if not rows:
        print("[ERROR] table 'daily_bars' not found")
        return
    print_rows("sqlite_master (daily_bars)", cols, rows, limit=1)

    # 2) 先頭5件
    cols, rows = q(conn, "SELECT * FROM daily_bars LIMIT 5;")
    print_rows("sample rows", cols, rows, limit=5)

    # 3) 行数・銘柄数・日付範囲
    cols, rows = q(
        conn,
        """
      SELECT
        (SELECT COUNT(*) FROM daily_bars) AS rows,
        (SELECT COUNT(DISTINCT ticker) FROM daily_bars) AS n_tickers,
        (SELECT MIN(date) FROM daily_bars) AS min_date,
        (SELECT MAX(date) FROM daily_bars) AS max_date
    """,
    )
    print_rows("overview", cols, rows, limit=1)

    # 4) 銘柄ごとのカバレッジ（CSVにも保存）
    cols, rows = q(
        conn,
        """
      SELECT ticker,
             MIN(date) AS start_date,
             MAX(date) AS end_date,
             COUNT(*)  AS n_rows
      FROM daily_bars
      GROUP BY ticker
      ORDER BY ticker;
    """,
    )
    print_rows("per-ticker coverage (first 10)", cols, rows, limit=10)

    os.makedirs("data", exist_ok=True)
    out_csv = r"data/coverage.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    print(f"\n✅ Exported coverage to {out_csv}")

    # 5) 欠け銘柄の洗い出し（codesファイルがある場合）
    if os.path.exists(CODES_TXT):
        want = []
        with open(CODES_TXT, "r", encoding="utf-8") as f:
            for line in f:
                c = line.strip()
                if c:
                    want.append(c if c.endswith(".T") else c + ".T")
        want = sorted(set(want))

        have_cols, have_rows = q(conn, "SELECT DISTINCT ticker FROM daily_bars;")
        have = sorted(set(r[0] for r in have_rows))

        missing = sorted(set(want) - set(have))
        print(f"\nMissing tickers: {len(missing)}")
        for t in missing[:20]:
            print("  ", t)
        if len(missing) > 20:
            print("  ...")

        miss_path = r"data/missing.txt"
        with open(miss_path, "w", encoding="utf-8") as f:
            for t in missing:
                f.write(t + "\n")
        print(f"📝 Missing list saved to {miss_path}")
    else:
        print(
            "\n(補足) data/topix100_codes.txt が見つからないため、欠け銘柄突合はスキップしました。"
        )

    conn.close()


if __name__ == "__main__":
    main()
