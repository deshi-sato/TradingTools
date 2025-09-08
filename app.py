\"\"\"app.py : Auto-generated placeholder

- file: app.py
- updated: 2025-09-08

TODO: このモジュールの概要をここに書いてください。
\"\"\"
# app.py — DB直読でスナップショット一覧を表示
# 起動:  python app.py
from pathlib import Path
import sqlite3
from flask import Flask, render_template, jsonify, request
from score_table import compute_trend_score_for_snapshots

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data" / "rss_data.db"

app = Flask(__name__)


def get_conn():
    # 接続は都度開閉（Flaskの簡易構成として安全）
    return sqlite3.connect(str(DB_PATH))


# JSON API（テーブル更新用）
@app.get("/api/snapshots")
def api_snapshots():
    sort = request.args.get(
        "sort", "updated_at"
    )  # diff_pct / diff / last / volume / score など
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
        "score",  # フロントからの並び替え拡張
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

    base_sql = (
        "SELECT "
        "ticker, sheet_name, last, prev_close, open, high, low, "
        "volume, turnover, diff, diff_pct, updated_at "
        "FROM quote_latest "
        f"{where}"
    )

    with get_conn() as conn:
        if sort == "score":
            # スコアはDB列に無いため、全件(条件付き)を取得後にPython側で並び替え
            sql = base_sql  # 取得のみ（必要なら既定順序を付与してもOK）
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        else:
            sql = base_sql + f" ORDER BY {sort} {direction} LIMIT ?"
            cur = conn.execute(sql, [*params, limit])
            rows = cur.fetchall()

    # Compute trend scores map per ticker
    scores_map = compute_trend_score_for_snapshots(str(DB_PATH))

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
    data = []
    for r in rows:
        d = dict(zip(cols, r))
        d["score"] = scores_map.get(d["ticker"])
        data.append(d)

    if sort == "score":
        asc = (direction == "ASC")
        # Noneは常に末尾に回す
        def key_fn(item):
            s = item.get("score")
            if s is None:
                return (1, 0)  # None扱い
            return (0, s if asc else -s)

        data.sort(key=key_fn)
        # 限定数を最後に適用
        data = data[:limit]

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
