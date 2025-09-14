import argparse, json, sqlite3, time
from collections import deque, defaultdict
from datetime import datetime, timedelta


def ema_update(prev, x, alpha):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Verbose", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = json.load(open(args.Config, "r", encoding="utf-8"))
    db_path = cfg["db_path"]
    B = cfg["burst"]
    K = int(B["window_count"])
    upt_buy = float(B["uptick_thr_buy"])
    upt_sell = float(B["uptick_thr_sell"])
    imb_thr = float(B["imb_thr"])
    max_spread = float(B["max_spread_bp"])
    vol_gate = float(B["vol_gate"])
    vol_sum_gate = float(B["vol_sum_gate"])
    allow_spread_none = bool(B["allow_spread_none_with_vol"])
    cooldown = float(B["cooldown_sec"])
    w = B["burst_score_weights"]
    ema_span = float(B["ema_span_sec"])  # 秒ベース、≈K*window_msで調整
    # 300ms窓想定なら 60s / (0.3s) ≈ 200サンプル
    # 雑にalpha計算（1-exp(-1/N)）
    N = max(5, int(ema_span / 0.3))
    alpha = 1.0 - pow(2.718281828, -1.0 / N)

    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 進捗管理
    last_rowid = 0
    window_q = defaultdict(lambda: deque(maxlen=K))
    ema_vol = defaultdict(lambda: None)
    last_sig_time = defaultdict(lambda: 0.0)

    # 連続ポーリング
    while True:
        # 新着のみ読む
        cur.execute(
            """
            SELECT rowid, ticker, ts, uptick_ratio, vol_sum, spread_bp, depth_imbalance
            FROM features_stream
            WHERE rowid > ?
            ORDER BY rowid ASC
            LIMIT 500
        """,
            (last_rowid,),
        )
        rows = cur.fetchall()
        if not rows:
            time.sleep(0.15)
            continue

        now_epoch = time.time()
        to_update = []
        for r in rows:
            last_rowid = r["rowid"]
            tkr = r["ticker"]
            upt = r["uptick_ratio"] or 0.5
            vol = r["vol_sum"] or 0.0
            spr = r["spread_bp"]  # Noneあり
            imb = r["depth_imbalance"] or 0.0

            window_q[tkr].append((upt, vol, spr, imb))
            ema_vol[tkr] = ema_update(ema_vol[tkr], vol, alpha)
            surge_ratio = (vol / max(1.0, ema_vol[tkr])) if ema_vol[tkr] else 1.0

            # 直近K窓の集計
            w_upt = [x[0] for x in window_q[tkr]]
            w_vol = [x[1] for x in window_q[tkr]]
            w_spr = [x[2] for x in window_q[tkr]]
            w_imb = [x[3] for x in window_q[tkr]]
            if len(w_upt) < K:  # まだ溜まってない
                continue

            # spread条件
            if all(s is None for s in w_spr):
                spread_ok = allow_spread_none and sum(w_vol) >= vol_sum_gate
            else:
                latest_spread = w_spr[-1]
                spread_ok = (latest_spread is not None) and (
                    latest_spread <= max_spread
                )

            # 基本ゲート
            vol_ok = (vol >= vol_gate) and (sum(w_vol) >= vol_sum_gate)

            # 一貫性：多数決
            up_votes = sum(1 for x in w_upt if x >= upt_buy)
            dn_votes = sum(1 for x in w_upt if x <= upt_sell)
            imb_up = sum(1 for x in w_imb if x >= +imb_thr)
            imb_dn = sum(1 for x in w_imb if x <= -imb_thr)

            buy_consistent = (up_votes >= (K // 2 + 1)) and (imb_up >= (K // 2))
            sell_consistent = (dn_votes >= (K // 2 + 1)) and (imb_dn >= (K // 2))

            burst_buy = 1 if (spread_ok and vol_ok and buy_consistent) else 0
            burst_sell = 1 if (spread_ok and vol_ok and sell_consistent) else 0

            # スコア（0〜1）
            avg_upt = sum(w_upt) / K
            avg_imb = sum(abs(x) for x in w_imb) / K
            vol_term = (
                min(3.0, sum(w_vol) / max(1.0, ema_vol[tkr] * K)) / 3.0
            )  # 0〜1に丸め
            burst_score = (
                (w["uptick"] * avg_upt)
                + (w["imbalance"] * avg_imb)
                + (w["vol_surge"] * vol_term)
            )
            burst_score = max(0.0, min(1.0, burst_score))

            # クールダウン
            if (burst_buy or burst_sell) and (
                now_epoch - last_sig_time[tkr] < cooldown
            ):
                burst_buy = 0
                burst_sell = 0  # 抑止

            if burst_buy or burst_sell:
                last_sig_time[tkr] = now_epoch

            # streak（方向一貫性の長さ）
            streak = 0
            if avg_upt >= 0.5:
                for u in reversed(w_upt):
                    if u >= upt_buy:
                        streak += 1
                    else:
                        break
            else:
                for u in reversed(w_upt):
                    if u <= upt_sell:
                        streak += 1
                    else:
                        break

            to_update.append(
                (
                    burst_buy,
                    burst_sell,
                    burst_score,
                    streak,
                    surge_ratio,
                    datetime.utcnow().isoformat(timespec="seconds"),
                    r["rowid"],
                )
            )

        if to_update:
            cur.executemany(
                """
                UPDATE features_stream
                SET burst_buy=?, burst_sell=?, burst_score=?, streak_len=?, surge_vol_ratio=?, last_signal_ts=?
                WHERE rowid=?
            """,
                to_update,
            )
            conn.commit()


if __name__ == "__main__":
    main()
