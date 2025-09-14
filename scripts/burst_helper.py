# scripts/burst_helper.py への差分上書き（関数だけ更新）
import sqlite3, math
from datetime import datetime, timezone

def _to_dt(ts: str):
    try:
        dt = datetime.fromisoformat(ts.replace("Z",""))
        if dt.tzinfo: dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None

def fetch_recent_burst_strength(conn: sqlite3.Connection, ticker: str,
                                within_sec: float | None = 8.0,
                                min_gate: float = 0.0):
    """
    within_sec が None の場合は「時間条件なし」で最新のシグナルのみを見る。
    戻り値: (score_max, ts_max) / 見つからなければ (0.0, None)
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT ts, COALESCE(burst_score,0.0)
        FROM features_stream
        WHERE ticker=? AND (burst_buy=1 OR burst_sell=1)
        ORDER BY ts DESC
        LIMIT 50
    """, (ticker,))
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    score_max, ts_max = 0.0, None
    for ts, sc in cur.fetchall():
        dt = _to_dt(ts)
        if not dt:
            continue
        if sc < min_gate:
            continue
        if within_sec is not None:
            if (now_utc - dt).total_seconds() > within_sec:
                continue
        if sc > score_max:
            score_max, ts_max = sc, dt
    return score_max, ts_max

def burst_bonus(conn: sqlite3.Connection, ticker: str,
                base_score: float,
                k: float = 0.30,
                tau_sec: float = 10.0,
                lookback_sec: float | None = 8.0,
                min_gate: float = 0.0):
    """
    lookback_sec=None で時間ゲート無効化（テスト用）。
    bonus = k * burst_score * exp(-Δt/tau)
    """
    score, ts = fetch_recent_burst_strength(conn, ticker, lookback_sec, min_gate)
    if score <= 0 or ts is None:
        return base_score, 0.0, None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    age = max(0.0, (now - ts).total_seconds())
    decay = math.exp(-age / tau_sec) if tau_sec > 0 else 1.0
    bonus = k * score * decay
    return base_score + bonus, bonus, ts
