from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Literal, Set
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from .api_client import get_board, send_order

# ===== 基本設定 =====
JST = ZoneInfo("Asia/Tokyo")
Side = Literal["BUY", "SELL"]

# ウォッチリスト（許可銘柄）をCSVから読む
ALLOW_SYMBOLS_FILE = os.environ.get("ALLOW_SYMBOLS_FILE", "data/watchlist_today.csv")

# 1トレードの推定約定代金の上限（円）
MAX_NOTIONAL = int(os.environ.get("MAX_NOTIONAL", "1000000"))

# 実行モード: DRYRUN / PAPER / LIVE
MODE = os.environ.get("MODE", "DRYRUN").upper()

# ログ
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
ORD_LOG = os.path.join(LOG_DIR, f"orders-{datetime.now(tz=JST):%Y%m%d}.jsonl")


# ===== 共通ユーティリティ =====
def _log(obj: dict) -> None:
    """JSON Lines で監査ログを追記"""
    with open(ORD_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _now_jst() -> datetime:
    return datetime.now(tz=JST)


def _in_time_window() -> bool:
    if os.environ.get("ALLOW_ANYTIME", "0").lower() in ("1", "true", "yes", "on"):
        return True

    start_s = os.environ.get("TW_START", "09:00")
    end_s = os.environ.get("TW_END", "10:15")
    try:
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        start, end = dtime(sh, sm), dtime(eh, em)
    except Exception:
        start, end = dtime(9, 0), dtime(10, 15)

    now = _now_jst().time()
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end


def _load_allow_symbols() -> Set[str]:
    path = ALLOW_SYMBOLS_FILE
    if not os.path.exists(path):
        return set()
    syms: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return syms
        cols = [c.strip().lower() for c in header.split(",")]
        idx = 0
        for i, c in enumerate(cols):
            if c in ("code", "symbol", "ticker"):
                idx = i
                break
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) > idx and parts[idx]:
                syms.add(parts[idx])
    return syms


@dataclass
class ExecResult:
    ok: bool
    order_id: str | None
    msg: str


def _estimate_notional(symbol: str, qty: int, px: float | None, ref_price: float | None = None) -> int:
    if px and px > 0:
        return int(px * qty)
    if MODE == "DRYRUN" or MODE == "PAPER":
        base = ref_price if (ref_price and ref_price > 0) else 1000
        return int(base * qty)

    try:
        b = get_board(symbol)
        ref = b.get("CurrentPrice") or b.get("AskPrice") or b.get("BidPrice") or 0
        return int(float(ref) * qty) if ref else 999_999_999
    except Exception:
        return 999_999_999


# ===== 発注ラッパ =====
def place_ifdoco(
    symbol: str,
    side: Side,
    qty: int,
    entry: float | None,
    stop: float,
    take: float,
    reason: str | None = None,
) -> ExecResult:
    record_base = {
        "ts": f"{_now_jst():%F %T}",
        "mode": MODE,
        "action": "IFDOCO",
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry": entry,
        "stop": stop,
        "take": take,
        "reason": reason,
    }

    allow = _load_allow_symbols()
    if symbol not in allow:
        _log({**record_base, "phase": "deny", "reason": "symbol_not_allowed"})
        return ExecResult(False, None, f"DENY symbol {symbol} not in watchlist")

    if qty <= 0:
        _log({**record_base, "phase": "deny", "reason": "qty_le_0"})
        return ExecResult(False, None, "qty must be positive")

    if not _in_time_window():
        _log({**record_base, "phase": "deny", "reason": "outside_window"})
        return ExecResult(False, None, "outside trading window")

    notion = _estimate_notional(symbol, qty, entry)
    if notion > MAX_NOTIONAL:
        _log({**record_base, "phase": "deny", "reason": "notional_exceeded", "est": notion})
        return ExecResult(False, None, f"DENY notional {notion} > {MAX_NOTIONAL}")

    _log({**record_base, "phase": "precheck", "ok": True, "est_notional": notion})

    # ==== 擬似発注モード（DRYRUN/PAPER/環境変数で指定時） ====
    if MODE in ("DRYRUN", "PAPER") or os.environ.get("SIMULATE_EXEC") == "1":
        last = None
        try:
            b = get_board(symbol)
            last = b.get("CurrentPrice") or b.get("AskPrice") or b.get("BidPrice")
        except Exception:
            pass
        bps = float(os.environ.get("SIM_SLIPPAGE_BPS", "5"))
        if last:
            slip = (bps/10000.0) * float(last)
            sim_fill_price = float(last) + slip if side == "BUY" else float(last) - slip
        else:
            sim_fill_price = None
        _log({**record_base, "phase": "simulated", "ok": True,
              "order_id": f"SIM-{int(time.time()*1000)}",
              "sim_fill_price": sim_fill_price,
              "sim_slippage_bps": bps})
        return ExecResult(True, "simulated", "skipped sending order (SIMULATED)")

    # ==== LIVE のみ実注文 ====
    payload_entry = {
        "Symbol": symbol,
        "Side": side.upper(),
        "Qty": qty,
        "Price": int(entry) if entry and entry > 0 else 0,
        "OrderType": "LIMIT" if entry and entry > 0 else "MARKET",
    }
    payload_stop = {
        "Symbol": symbol,
        "Side": "SELL" if side == "BUY" else "BUY",
        "Qty": qty,
        "OrderType": "STOP",
        "Price": int(stop),
    }
    payload_take = {
        "Symbol": symbol,
        "Side": "SELL" if side == "BUY" else "BUY",
        "Qty": qty,
        "OrderType": "LIMIT",
        "Price": int(take),
    }

    try:
        ent = send_order(payload_entry)
        oid = ent.get("OrderId") or ent.get("ID")
        _log({**record_base, "phase": "entry_sent", "resp": ent})

        stp = send_order(payload_stop)
        _log({**record_base, "phase": "stop_sent", "resp": stp})

        tk = send_order(payload_take)
        _log({**record_base, "phase": "take_sent", "resp": tk})

        return ExecResult(True, str(oid) if oid else None, "ok")

    except Exception as e:
        _log({**record_base, "phase": "error", "error": str(e)})
        return ExecResult(False, None, f"error: {e}")


# ==== closeout 連携用: 注文/建玉 操作用の薄ラッパ =============================
def list_open_orders() -> list[dict]:
    if MODE != "LIVE":
        return []
    return []

def cancel_order(order_id: str) -> tuple[bool, str]:
    if MODE != "LIVE":
        return True, "noop (MODE != LIVE)"
    try:
        import json as _json, requests
        base = os.environ.get("KABU_BASE_URL", "http://localhost:18080").rstrip("/")
        headers = {"Content-Type": "application/json"}
        tok = os.environ.get("KABU_TOKEN") or os.environ.get("KABU_API_KEY")
        if tok:
            headers["X-API-KEY"] = tok
        resp = requests.put(f"{base}/kabusapi/cancelorder",
                            headers=headers,
                            data=_json.dumps({"OrderId": order_id}),
                            timeout=8)
        ok = (resp.status_code == 200)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return ok, f"{resp.status_code} {body}"
    except Exception as e:
        return False, f"cancel error: {e}"

def list_positions() -> list[dict]:
    if MODE != "LIVE":
        return []
    return []

def close_position_market(pos: dict, qty: int) -> tuple[bool, str]:
    if MODE != "LIVE":
        return True, "noop (MODE != LIVE)"
    side0 = str(pos.get("Side", "")).upper()
    symbol = pos.get("Symbol") or pos.get("symbol") or pos.get("Code")
    if not symbol:
        return False, "missing symbol"
    side = "SELL" if side0 == "BUY" else "BUY"
    try:
        payload = {"Symbol": symbol, "Side": side, "Qty": int(qty), "OrderType": "MARKET", "Price": 0}
        r = send_order(payload)
        return True, str(r)
    except Exception as e:
        return False, f"close error: {e}"
