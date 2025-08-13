# app.py — DB直読でスナップショット一覧を表示
# 起動:  python app.py
from pathlib import Path
import sqlite3
from flask import Flask, render_template, jsonify, request

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "rss_data.db"

app = Flask(__name__)


def get_conn():
    # 接続は都度開閉（Flaskの簡易構成として安全）
    return sqlite3.connect(str(DB_PATH))


# JSON API（テーブル更新用）
@app.get("/api/snapshots")
def api_snapshots():
    sort = request.args.get(
        "sort", "updated_at"
    )  # diff_pct / diff / last / volume など
    order = request.args.get("order", "desc")
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", "500"))

    # 許可するソートキーだけ許容
    allowed = {
        "updated_at",
        "diff_pct",
        "diff",
        "last",
        "turnover",
        "volume",
        "high",
        "low",
        "open",
        "prev_close",
        "ticker",
        "sheet_name",
    }
    if sort not in allowed:
        sort = "updated_at"
    direction = "DESC" if order.lower() == "desc" else "ASC"

    # 文字列検索（ticker, sheet_name 対象）
    where = ""
    params = []
    if q:
        where = "WHERE ticker LIKE ? OR sheet_name LIKE ?"
        like = f"%{q}%"
        params.extend([like, like])

    sql = f"""
        SELECT
          ticker, sheet_name, last, prev_close, open, high, low,
          volume, turnover, diff, diff_pct, updated_at
        FROM quote_latest
        {where}
        ORDER BY {sort} {direction}
        LIMIT ?
    """
    params.append(limit)

    with get_conn() as conn:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()

    cols = [
        "ticker",
        "sheet_name",
        "last",
        "prev_close",
        "open",
        "high",
        "low",
        "volume",
        "turnover",
        "diff",
        "diff_pct",
        "updated_at",
    ]
    data = [dict(zip(cols, r)) for r in rows]
    return jsonify(data)


# 画面（表のみ）
@app.get("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"❌ DBが見つかりません: {DB_PATH}")
        raise SystemExit(1)
    app.run(debug=True)
