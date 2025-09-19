# exec/closeout.py
from __future__ import annotations
import os
import sys
import json
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Dict, Any

# --- 設定・ユーティリティ ----------------------------------------------------

JST = timezone(timedelta(hours=9))
LOG_DIR = Path(".logs")
LOG_DIR = Path(os.environ.get("CLOSE_LOG_DIR", "logs"))


def jst_now() -> datetime:
    return datetime.now(JST)


def log_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def today_log_path(prefix: str) -> Path:
    return LOG_DIR / f"{prefix}-{jst_now().date().isoformat().replace('-', '')}.jsonl"


def is_live_mode() -> bool:
    # LIVE 以外（PAPER, DRYRUN など）は実行しても発注系は必ず noop
    return os.environ.get("MODE", "").upper() == "LIVE"


# --- 既存ラッパー（あれば利用） ----------------------------------------------

"""
あなたの環境にある発注ラッパーを優先利用します。
- exec.kabu_exec から以下が import できればそれを使用
    - list_open_orders() -> Iterable[dict]
    - cancel_order(order_id: str) -> tuple[bool, str]
    - list_positions() -> Iterable[dict]
    - close_position_market(pos: dict, qty: int) -> tuple[bool, str]
"""
USE_NATIVE_WRAPPERS = False
try:
    from exec.kabu_exec import (
        list_open_orders as kabu_list_open_orders,
        cancel_order as kabu_cancel_order,
        list_positions as kabu_list_positions,
        close_position_market as kabu_close_position_market,
    )

    USE_NATIVE_WRAPPERS = True
except Exception:
    # フォールバックへ（最低限の /cancelorder のみ提供）
    USE_NATIVE_WRAPPERS = False

# --- フォールバック：最低限のREST直叩き（取消のみ） -------------------------


def _api_request(
    method: str,
    endpoint: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import urllib.request
    import urllib.error

    base = os.environ.get("KABU_BASE_URL", "http://localhost:18080")
    url = f"{base.rstrip('/')}/kabusapi/{endpoint.lstrip('/')}"
    req = urllib.request.Request(url, method=method.upper())
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    api_key = os.environ.get("KABU_TOKEN") or os.environ.get("KABU_API_KEY")
    if api_key:
        req.add_header("X-API-KEY", api_key)

    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
    else:
        data = None

    try:
        with urllib.request.urlopen(req, data=data, timeout=10) as resp:
            txt = resp.read().decode("utf-8")
            return {
                "ok": True,
                "status": resp.status,
                "body": json.loads(txt) if txt else {},
            }
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "ignore")
        return {"ok": False, "status": getattr(e, "code", None), "body": txt}
    except Exception as e:
        return {"ok": False, "status": None, "body": str(e)}


def _fallback_cancel_order(order_id: str) -> tuple[bool, str]:
    """
    /cancelorder による取消（公式OpenAPIに定義あり）
    """
    payload = {"OrderId": order_id}
    r = _api_request("PUT", "cancelorder", json_body=payload)
    ok = bool(r.get("ok")) and (r.get("status") == 200)
    return ok, f"{r.get('status')} {r.get('body')}"


# --- 型と主処理 ---------------------------------------------------------------


@dataclass
class CloseoutStats:
    canceled: int = 0
    cancel_err: int = 0
    flattened: int = 0
    flat_err: int = 0


def cancel_all_orders(stats: CloseoutStats, reason: str = "end_of_session") -> None:
    """
    すべての未約定注文を取消。LIVE以外はログのみ（noop）
    """
    logp = today_log_path("closeout")
    ts = jst_now().isoformat()

    # 注文一覧の取得はユーザーラッパー前提（存在しなければスキップ）
    if not USE_NATIVE_WRAPPERS:
        log_jsonl(
            logp,
            {
                "t": ts,
                "phase": "cancel_all_orders",
                "ok": True,
                "msg": "no native list_open_orders(); skipped listing",
            },
        )
        return

    orders = list(kabu_list_open_orders())
    log_jsonl(logp, {"t": ts, "phase": "cancel_all_orders", "found": len(orders)})

    for od in orders:
        oid = od.get("ID") or od.get("OrderId") or od.get("OrderID")
        if not oid:
            continue

        if not is_live_mode():
            log_jsonl(
                logp,
                {
                    "t": jst_now().isoformat(),
                    "action": "cancel",
                    "order_id": oid,
                    "mode": os.environ.get("MODE", ""),
                    "ok": True,
                    "msg": "DRYRUN: skipped",
                },
            )
            stats.canceled += 1
            continue

        if USE_NATIVE_WRAPPERS:
            ok, msg = kabu_cancel_order(oid)
        else:
            ok, msg = _fallback_cancel_order(oid)

        log_jsonl(
            logp,
            {
                "t": jst_now().isoformat(),
                "action": "cancel",
                "order_id": oid,
                "mode": "LIVE",
                "ok": ok,
                "msg": msg,
                "reason": reason,
            },
        )
        if ok:
            stats.canceled += 1
        else:
            stats.cancel_err += 1


def close_all_positions(stats: CloseoutStats, reason: str = "end_of_session") -> None:
    """
    すべての建玉を成行クローズ。LIVE以外はログのみ（noop）
    """
    logp = today_log_path("closeout")
    ts = jst_now().isoformat()

    if not USE_NATIVE_WRAPPERS:
        log_jsonl(
            logp,
            {
                "t": ts,
                "phase": "close_all_positions",
                "ok": True,
                "msg": "no native close_position_market(); skipped",
            },
        )
        return

    positions = list(kabu_list_positions())
    log_jsonl(logp, {"t": ts, "phase": "close_all_positions", "found": len(positions)})

    for pos in positions:
        qty = int(pos.get("Qty") or pos.get("LeavesQty") or 0)
        if qty <= 0:
            continue

        if not is_live_mode():
            log_jsonl(
                logp,
                {
                    "t": jst_now().isoformat(),
                    "action": "flatten",
                    "symbol": pos.get("Symbol"),
                    "qty": qty,
                    "mode": os.environ.get("MODE", ""),
                    "ok": True,
                    "msg": "DRYRUN: skipped",
                },
            )
            stats.flattened += 1
            continue

        ok, msg = kabu_close_position_market(pos, qty)
        log_jsonl(
            logp,
            {
                "t": jst_now().isoformat(),
                "action": "flatten",
                "symbol": pos.get("Symbol"),
                "qty": qty,
                "mode": "LIVE",
                "ok": ok,
                "msg": msg,
                "reason": reason,
            },
        )
        if ok:
            stats.flattened += 1
        else:
            stats.flat_err += 1


# --- CLI ----------------------------------------------------------------------


def parse_time_hhmm(s: str) -> dtime:
    s = s.strip()
    h, m = (
        (int(s[:2]), int(s[2:]))
        if s.isdigit() and len(s) == 4
        else map(int, s.split(":"))
    )
    return dtime(hour=h, minute=m, tzinfo=JST)


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m exec.closeout",
        description="Cancel all orders & flatten all positions safely.",
    )
    p.add_argument(
        "--deadline",
        default="09:25",
        help="実施基準時刻（JST）。例: 09:25 / 0925 / 9:25",
    )
    p.add_argument(
        "--force", action="store_true", help="現在時刻がdeadline前でも実行する"
    )
    p.add_argument(
        "--cancel-only",
        action="store_true",
        help="取消のみ実施（建玉クローズはしない）",
    )
    p.add_argument(
        "--flatten-only",
        action="store_true",
        help="建玉クローズのみ実施（取消はしない）",
    )
    p.add_argument("--reason", default="end_of_session", help="ログに残す理由タグ")
    args = p.parse_args(list(argv) if argv is not None else None)

    deadline = parse_time_hhmm(args.deadline)
    now = jst_now()
    do_exec = args.force or (now.time() >= deadline)

    logp = today_log_path("closeout")
    log_jsonl(
        logp,
        {
            "t": now.isoformat(),
            "phase": "start",
            "mode": os.environ.get("MODE", ""),
            "deadline": deadline.strftime("%H:%M"),
            "force": args.force,
            "exec": do_exec,
        },
    )

    if not do_exec:
        print(
            f"[INFO] now<{deadline.strftime('%H:%M')} なのでスキップ（--forceで強制実行可能）"
        )
        return 0

    stats = CloseoutStats()

    if not args.flatten_only:
        cancel_all_orders(stats, reason=args.reason)

    if not args.cancel_only:
        close_all_positions(stats, reason=args.reason)

    log_jsonl(logp, {"t": jst_now().isoformat(), "phase": "done", **stats.__dict__})
    print(
        f"[DONE] canceled={stats.canceled}/{stats.cancel_err}  flattened={stats.flattened}/{stats.flat_err}"
    )
    print(f"[LOG FILE] {today_log_path('closeout')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
