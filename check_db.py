import os, sqlite3, csv

DB = r"data/rss_daily.db"


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
        raise SystemExit(f"[ERROR] DB not found: {DB}")
    conn = sqlite3.connect(DB)

    # 1) スキーマ
    cols, rows = q(conn, "SELECT name, sql FROM sqlite_master WHERE type='table' AND name='daily_bars';")
    print_rows("sqlite_master (daily_bars)", cols, rows, limit=1)

    # 2) サンプル
    cols, rows = q(conn, "SELECT * FROM daily_bars LIMIT 5;")
    print_rows("sample rows", cols, rows, limit=5)

    # 3) 概要
    cols, rows = q(conn, """
      SELECT
        (SELECT COUNT(*) FROM daily_bars) AS rows,
        (SELECT COUNT(DISTINCT ticker) FROM daily_bars) AS n_tickers,
        (SELECT MIN(date) FROM daily_bars) AS min_date,
        (SELECT MAX(date) FROM daily_bars) AS max_date
    """)
    print_rows("overview", cols, rows, limit=1)

    # 4) 銘柄別件数
    cols, rows = q(conn, """
      SELECT ticker, MIN(date) AS start_date, MAX(date) AS end_date, COUNT(*) AS n_rows
      FROM daily_bars
      GROUP BY ticker
      ORDER BY ticker;
    """)
    print_rows("per-ticker coverage (first 10)", cols, rows, limit=10)

    os.makedirs("data", exist_ok=True)
    with open(r"data/coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    print("\nExported coverage to data/coverage.csv")

    conn.close()


if __name__ == "__main__":
    main()
