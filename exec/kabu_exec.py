from __future__ import annotations

import os
import json
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
    """
    取引時間ガード。
    - ALLOW_ANYTIME=1/true/on なら常に許可（テスト用）
    - TW_START 'HH:MM', TW_END 'HH:MM' で任意の時間帯に変更可能
    - 日跨ぎ（例: 23:00-01:00）も許容
    既定は 09:00-10:15
    """
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
        # 通常（同日内）ウィンドウ
        return start <= now <= end
    # 日跨ぎウィンドウ（例: 23:00-01:00）
    return now >= start or now <= end


def _load_allow_symbols() -> Set[str]:
    """watchlist_today.csv の code/symbol/ticker カラムから許可銘柄を読み込む"""
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
    # 指定価格があればそれで
    if px and px > 0:
        return int(px * qty)
    # DRYRUN は API に触らない（板が取れず 999_999_999 になるのを防ぐ）
    if MODE == "DRYRUN":
        base = ref_price if (ref_price and ref_price > 0) else 1000
        return int(base * qty)

    try:
        b = get_board(symbol)
        ref = b.get("CurrentPrice") or b.get("AskPrice") or b.get("BidPrice") or 0
        return int(float(ref) * qty) if ref else 999_999_999
    except Exception:
        return 999_999_999


# ===== 発注ラッパ（唯一のAPI接点） =====
def place_ifdoco(
    symbol: str,
    side: Side,
    qty: int,
    entry: float | None,
    stop: float,
    take: float,
    reason: str | None = None,
) -> ExecResult:
    """
    IFDOCO 相当の発注を行うラッパ。
    - 許可銘柄 / 時間帯 / 上限金額 / 数量 を事前チェック
    - MODE=DRYRUN の場合は送信せず検証のみ
    - 実装簡略化のためエントリー→STOP→TAKE の順に個別送信（本番はIFDOCO APIに差し替え）
    """
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

    # ---- 事前チェック（NGは deny ログを残して終了）----
    allow = _load_allow_symbols()
    if symbol not in allow:
        _log({**record_base, "phase": "deny", "reason": "symbol_not_allowed"})
        return ExecResult(False, None, f"DENY symbol {symbol} not in watchlist")

    if qty <= 0:
        _log({**record_base, "phase": "deny", "reason": "qty_le_0"})
        return ExecResult(False, None, "qty must be positive")

    if not _in_time_window():
        _log({**record_base, "phase": "deny", "reason": "outside_window"})
        return ExecResult(False, None, "outside trading window (09:00-10:15 JST)")

    notion = _estimate_notional(symbol, qty, entry, ref_price)
    if notion > MAX_NOTIONAL:
        _log(
            {**record_base, "phase": "deny", "reason": "notional_exceeded", "est": notion}
        )
        return ExecResult(False, None, f"DENY notional {notion} > {MAX_NOTIONAL}")

    # ---- ここから送信準備 ----
    _log({**record_base, "phase": "precheck", "ok": True, "est_notional": notion})

    if MODE == "DRYRUN":
        # 検証のみ
        _log({**record_base, "phase": "dryrun", "ok": True})
        return ExecResult(True, "dryrun", "skipped sending order (DRYRUN)")

    # NOTE: PAPER / LIVE は送信（接続先は環境変数で切替）
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
