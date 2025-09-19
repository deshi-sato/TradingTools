# orchestrate/run_intraday.py
from __future__ import annotations
import os, time, json, random, csv
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Optional

from exec.kabu_exec import place_ifdoco, MODE
from exec.api_client import get_board
from signals.core import Tick, scan_signals
from signals.scorer import score_long, score_short

JST = ZoneInfo("Asia/Tokyo")
STOP_PCT = float(os.environ.get("STOP_PCT", "0.005"))
TAKE_PCT = float(os.environ.get("TAKE_PCT", "0.007"))
USE_FAKE  = os.environ.get("USE_FAKE_TICKS", "1") in ("1","true","yes","on")
POLL_MS   = int(os.environ.get("POLL_MS", "250"))
VERBOSE   = os.environ.get("VERBOSE", "0").lower() in ("1","true","yes","on")
HEARTBEAT_SEC = int(os.environ.get("HEARTBEAT_SEC", "10"))

# すでに import 済みの os/random/datetime/JST, Tick を前提
FAKE_TREND = os.environ.get("FAKE_TREND", "flat").lower()  # 'up' / 'down' / 'flat'
FAKE_DRIFT = float(os.environ.get("FAKE_DRIFT", "0.35"))   # 毎ティックの平均的な上昇/下降幅
FAKE_NOISE = float(os.environ.get("FAKE_NOISE", "0.9"))    # ランダム揺れ幅
FAKE_VOL_BASE = int(os.environ.get("FAKE_VOL_BASE", "800"))
FAKE_VOL_JITTER = int(os.environ.get("FAKE_VOL_JITTER", "250"))

# ---------- helpers ----------

def _get_cum_volume(t) -> Optional[int]:
    """Tick が持つ累積出来高の候補名を順に探す。無ければ None。"""
    for name in ("volume", "vol", "cum", "cum_volume", "trading_volume", "TradingVolume"):
        v = getattr(t, name, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return None

def _parse_code_exch(s: str) -> Tuple[str, int]:
    s = s.strip()
    if "@" in s:
        c, e = s.split("@", 1)
        try:
            return c, int(e)
        except:
            return c, 1
    return s, 1

def _load_symbols(path: str="data/watchlist_today.csv") -> List[Tuple[str,int]]:
    out: List[Tuple[str,int]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        header = [c.strip().lower() for c in f.readline().split(",")]
        idx = 0
        for i, c in enumerate(header):
            if c in ("code","symbol","ticker"):
                idx = i; break
        exch_idx = None
        for i, c in enumerate(header):
            if c in ("exchange","exch","market"):
                exch_idx = i; break
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            code = parts[idx]
            if exch_idx is not None and exch_idx < len(parts) and parts[exch_idx]:
                try:
                    out.append((code, int(parts[exch_idx])))
                except:
                    out.append(_parse_code_exch(code))
            else:
                out.append(_parse_code_exch(code))
    return out

# 価格・出来高の状態を銘柄ごとに保持
class _FakeGen:
    px = {}   # symbol -> last price
    vol = {}  # symbol -> last vol

def _fake_tick(sym: str) -> Tick:
    prev_px = _FakeGen.px.get(sym, 1000.0)
    drift = FAKE_DRIFT if FAKE_TREND == "up" else (-FAKE_DRIFT if FAKE_TREND == "down" else 0.0)
    noise = random.uniform(-FAKE_NOISE, FAKE_NOISE)
    px = max(10.0, prev_px + drift + noise)
    _FakeGen.px[sym] = px

    v_prev = _FakeGen.vol.get(sym, FAKE_VOL_BASE)
    vol = max(1, int(v_prev + random.randint(-FAKE_VOL_JITTER, FAKE_VOL_JITTER)))
    _FakeGen.vol[sym] = vol

    ts = int(datetime.now(tz=JST).timestamp() * 1000)
    return Tick(sym, ts, float(px), vol)

def _poll_tick(sym: str, exch: int) -> Optional[Tick]:
    try:
        b = get_board(sym, exchange=exch)
        ts = int(datetime.now(tz=JST).timestamp()*1000)
        last = float(b.get("CurrentPrice") or b.get("AskPrice") or b.get("BidPrice") or 0)
        vol  = int(b.get("TradingVolume") or 0)  # 当日累積出来高
        if last <= 0:
            return None
        return Tick(sym, ts, last, vol)
    except Exception as e:
        print(f"[poll_err] {sym}@{exch}: {e}")
        return None

def _load_prev_ohlc(path: str="data/prev_ohlc.csv") -> Dict[str, Tuple[float,float,float]]:
    """
    任意: symbol,prev_high,prev_low,prev_close を持つCSVを読めば精度UP。
    無ければ空dictを返し、当日ベースで代替評価します。
    """
    res: Dict[str, Tuple[float,float,float]] = {}
    if not os.path.exists(path):
        return res
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                res[row["symbol"]] = (float(row["prev_high"]), float(row["prev_low"]), float(row["prev_close"]))
            except:
                pass
    return res

def _today_hilo_and_cum(L: List[Tick]) -> Tuple[float, float, int]:
    hi = max(t.last for t in L)
    lo = min(t.last for t in L)
    vols = [v for v in (_get_cum_volume(t) for t in L) if v is not None]
    # 累積出来高が取れない（FAKEなど）場合は、疑似的に「ティック数」を累積出来高の代替にする
    cum = (max(vols) if vols else len(L))
    return hi, lo, cum

def _vol_windows(L: List[Tick], now_ms: int,
                 w10m: int = 10*60*1000, w30m: int = 30*60*1000) -> Tuple[int,int]:
    # 累積出来高ベース
    pairs = [(t.ts, _get_cum_volume(t)) for t in L]
    pairs = [(ts, v) for ts, v in pairs if v is not None]
    if pairs:
        def _diff(start_ms, end_ms):
            v_start = None; v_end = None
            for ts, cv in pairs:
                if ts <= start_ms: v_start = cv
                if ts <= end_ms:   v_end   = cv
                else: break
            if v_start is None: v_start = pairs[0][1]
            if v_end   is None: v_end   = pairs[-1][1]
            return max(0, v_end - v_start)
        v10     = _diff(now_ms - w10m, now_ms)
        v30prev = _diff(now_ms - (w10m + w30m), now_ms - w10m)
        return v10, v30prev
    # 代替（FAKE等）：ティック本数を出来高の代理に
    v10     = sum(1 for t in L if now_ms - w10m <= t.ts <= now_ms)
    v30prev = sum(1 for t in L if now_ms - (w10m + w30m) <= t.ts < now_ms - w10m)
    return v10, v30prev

def _hh_hl_like(L: List[Tick], lookback: int = 30) -> Tuple[bool,bool,bool,bool]:
    """
    今日のデータだけで“切り上げ/切り下げ”っぽさを代替判定。
    直近 lookback 本の高値・安値スイングで見る簡易版。
    """
    if len(L) < 4:
        return False, False, False, False
    sub = L[-lookback:] if len(L) > lookback else L[:]
    highs = [t.last for t in sub]
    # 簡易な傾向判定（必要に応じて改良可）
    hh_like = max(highs[-5:]) > max(highs[:-5]) if len(highs) > 10 else highs[-1] >= max(highs)
    hl_like = min(highs[-5:]) > min(highs[:-5]) if len(highs) > 10 else highs[-1] >= sorted(highs)[len(highs)//2]
    lh_like = max(highs[-5:]) < max(highs[:-5]) if len(highs) > 10 else highs[-1] <= sorted(highs)[len(highs)//2]
    ll_like = min(highs[-5:]) < min(highs[:-5]) if len(highs) > 10 else highs[-1] <= min(highs)
    return hh_like, hl_like, lh_like, ll_like

# ---------- main loop ----------

def main():
    syms = _load_symbols()
    if not syms:
        print("watchlist empty; abort")
        return

    prev_map = _load_prev_ohlc()
    print(f"[{datetime.now(tz=JST):%F %T}] MODE={MODE} symbols={len(syms)} "
          f"USE_FAKE={USE_FAKE} POLL_MS={POLL_MS}")

    buf: Dict[Tuple[str,int], List[Tick]] = {s: [] for s in syms}

    # HB 初期化（monotonic を使用）
    last_hb = time.monotonic() - HEARTBEAT_SEC  # すぐ出したくない場合は = time.monotonic()

    # 15分上限（必要なら環境変数化）
    deadline = time.time() + 15 * 60
    while time.time() < deadline:
        for (code, exch) in syms:
            tick = _fake_tick(code) if USE_FAKE else _poll_tick(code, exch)

            if tick:
                if VERBOSE:
                    print(f"[tick] {code}@{exch} last={tick.last}")
            else:
                if VERBOSE:
                    print(f"[tick-miss] {code}@{exch}")
                continue

            L = buf[(code,exch)]
            L.append(tick)
            if len(L) > 1200:  # 保険
                del L[:len(L)-1200]

            # 一次トリガー（既存シグナル）
            sigs = scan_signals(L)

            # スコア用特徴量
            today_hi, today_lo, vol_today = _today_hilo_and_cum(L)
            now_ms = L[-1].ts
            vol_10m, vol_prev30m = _vol_windows(L, now_ms)
            hh_like, hl_like, lh_like, ll_like = _hh_hl_like(L)
            prev_high = prev_low = prev_close = None
            if code in prev_map:
                prev_high, prev_low, prev_close = prev_map[code]

            for sg in sigs:
                last_price = L[-1].last

                if sg.side == "BUY":
                    stop = round(last_price * (1 - STOP_PCT))
                    take = round(last_price * (1 + TAKE_PCT))
                    sc = score_long(prev_high, prev_low, prev_close,
                                    today_hi, today_lo, last_price,
                                    vol_today, None,  # vol5d_avg 未使用なら None
                                    vol_10m, vol_prev30m,
                                    hh_like, hl_like)
                    if sc.side != "BUY":
                        print(json.dumps({"symbol": sg.symbol, "skip": "score_long", "score": sc.total},
                                         ensure_ascii=False))
                        continue
                    reason = f"{sg.reason}|score_long={sc.total}"
                else:
                    stop = round(last_price * (1 + STOP_PCT))
                    take = round(last_price * (1 - TAKE_PCT))
                    sc = score_short(prev_high, prev_low, prev_close,
                                     today_hi, today_lo, last_price,
                                     vol_today, None,
                                     vol_10m, vol_prev30m,
                                     lh_like, ll_like, gapdown_open=False)
                    if sc.side != "SELL":
                        print(json.dumps({"symbol": sg.symbol, "skip": "score_short", "score": sc.total},
                                         ensure_ascii=False))
                        continue
                    reason = f"{sg.reason}|score_short={sc.total}"

                res = place_ifdoco(
                    symbol=sg.symbol, side=sg.side, qty=100,
                    entry=None, stop=stop, take=take,
                    reason=reason, ref_price=last_price  # DRYRUN の約定代金見積り用
                )
                print(json.dumps(
                    {"symbol": sg.symbol, "side": sg.side, "ok": res.ok, "msg": res.msg},
                    ensure_ascii=False
                ))

        # ハートビート（monotonic）
        now = time.monotonic()
        if now - last_hb >= HEARTBEAT_SEC:
            snap = []
            for (c, e) in syms:
                L = buf[(c, e)]
                if L:
                    snap.append(f"{c}@{e}:{L[-1].last}")
            if snap:
                print("[hb]", ", ".join(snap))
            last_hb = now

        time.sleep(max(POLL_MS, 50)/1000)

if __name__ == "__main__":
    main()
