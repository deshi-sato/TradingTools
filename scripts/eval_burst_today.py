import sqlite3, argparse
from scripts.common_config import load_json_utf8
from datetime import datetime


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Since", default=None)  # "2025-09-14 00:00"
    ap.add_argument("-Ticker", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_json_utf8(args.Config)
    conn = sqlite3.connect(cfg["db_path"])
    c = conn.cursor()

    q = "SELECT ticker, ts, burst_buy, burst_sell, burst_score, uptick_ratio, depth_imbalance, vol_sum, spread_bp FROM features_stream"
    cond = []
    if args.Since:
        cond.append(f"ts >= '{args.Since}'")
    if args.Ticker:
        cond.append(f"ticker = '{args.Ticker}'")
    if cond:
        q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY ticker, ts"
    rows = c.execute(q).fetchall()

    # 集計（銘柄別の発火回数・平均スコア）
    from collections import defaultdict

    stats = defaultdict(lambda: {"buy": 0, "sell": 0, "score_sum": 0.0, "n": 0})
    for tkr, ts, bb, ss, sc, *_ in rows:
        s = stats[tkr]
        if bb:
            s["buy"] += 1
        if ss:
            s["sell"] += 1
        s["score_sum"] += sc or 0.0
        s["n"] += 1

    print("ticker,buy_cnt,sell_cnt,avg_burst_score")
    for tkr, s in stats.items():
        avg = (s["score_sum"] / s["n"]) if s["n"] else 0.0
        print(f"{tkr},{s['buy']},{s['sell']},{avg:.3f}")


if __name__ == "__main__":
    main()
