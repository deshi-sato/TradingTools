import argparse
import json
import time
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from scripts.common_config import load_json_utf8
import urllib.request
import urllib.error
import ssl
import traceback


# ===== HTTP utils ============================================================

def _req(method: str, url: str, token: str, body: Optional[dict] = None,
         timeout: float = 5.0) -> Tuple[int, str]:
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": token
    }
    req = urllib.request.Request(url=url, data=data, method=method, headers=headers)
    # kabuステはローカルHTTP。社内CA環境でも失敗しないよう簡易コンテキスト
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as res:
            return res.getcode(), res.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, f"EXC:{type(e).__name__}:{e}"


def _rest_json(method: str, url: str, token: str, body: Optional[dict] = None,
               retries: int = 3, backoff: float = 0.6) -> Tuple[int, Any]:
    """戻り: (status, json or text)"""
    for i in range(retries):
        st, txt = _req(method, url, token, body)
        # 一部エラーはリトライ（429/503/0）
        if st in (429, 503, 0):
            time.sleep(backoff * (2 ** i))
            continue
        try:
            return st, json.loads(txt)
        except Exception:
            return st, txt
    # 最終
    st, txt = _req(method, url, token, body)
    try:
        return st, json.loads(txt)
    except Exception:
        return st, txt


# ===== Rate guard（500ms間隔・1分上限）========================================

class RateLimiter:
    def __init__(self, min_interval_ms: int = 500, max_per_minute: int = 90):
        self.min_interval = max(0.001, min_interval_ms / 1000.0)
        self.max_per_min = max_per_minute
        self._sent_ts: List[float] = []
        self._last_ts: float = 0.0

    def wait(self):
        now = time.monotonic()
        # 500ms間隔
        dt = now - self._last_ts
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        # 1分上限
        now = time.monotonic()
        self._sent_ts = [t for t in self._sent_ts if now - t < 60.0]
        if len(self._sent_ts) >= self.max_per_min:
            sleep_sec = 60.0 - (now - self._sent_ts[0]) + 0.01
            time.sleep(max(0.01, sleep_sec))

        # 更新
        self._last_ts = time.monotonic()
        self._sent_ts.append(self._last_ts)


# ===== Flatten core ==========================================================

def load_cfg(path: str) -> dict:
    return load_json_utf8(path)


def list_active_orders(host: str, port: int, token: str) -> List[dict]:
    url = f"http://{host}:{port}/kabusapi/orders"  # 取得API（環境に合わせて）
    st, res = _rest_json("GET", url, token)
    if st == 200 and isinstance(res, list):
        return [o for o in res if str(o.get("State")) not in ("5", "6")]  # 5=失効,6=取消
    logging.warning("[FLATTEN] orders GET status=%s res=%s", st, res)
    return []


def cancel_order(host: str, port: int, token: str, order_id: str) -> Tuple[int, Any]:
    url = f"http://{host}:{port}/kabusapi/cancelorder"
    body = {"OrderId": order_id}
    return _rest_json("PUT", url, token, body)


def list_positions(host: str, port: int, token: str) -> List[dict]:
    url = f"http://{host}:{port}/kabusapi/positions"
    st, res = _rest_json("GET", url, token)
    if st == 200 and isinstance(res, list):
        return res
    logging.warning("[FLATTEN] positions GET status=%s res=%s", st, res)
    return []


def make_close_order_from_template(pos: dict, tpl: dict) -> dict:
    """
    テンプレに Position の情報を流し込む（symbol, exchange, side(反対), qty など）
    - kabu のフィールド名はテンプレ側に合わせる想定
    """
    symbol = str(pos.get("Symbol") or pos.get("symbol"))
    exch = int(pos.get("Exchange") or pos.get("exchange") or 1)
    side_pos = str(pos.get("Side") or pos.get("side") or "1")  # "1"=売, "2"=買 が多い
    qty = int(pos.get("LeavesQty") or pos.get("Qty") or pos.get("Quantity") or 0)

    # 反対側
    side_close = "2" if side_pos in ("1", 1) else "1"

    # テンプレを浅コピー
    body = json.loads(json.dumps(tpl, ensure_ascii=False))

    # よくあるフィールド（テンプレで上書き可）
    body.setdefault("Symbol", symbol)
    body.setdefault("Exchange", exch)
    body["Side"] = body.get("Side", side_close)
    body["Qty"] = body.get("Qty", qty)

    # 建玉クローズ指定が必要なAPI（例: ClosePosition=...）に備えて id/hold を入れておく
    if "ClosePositions" in body and isinstance(body["ClosePositions"], list):
        # 例: [{"HoldID": "...", "Qty": 100}]
        hold = (pos.get("HoldID") or pos.get("HoldId") or pos.get("PositionId") or "")
        body["ClosePositions"] = [{"HoldID": hold, "Qty": qty}]

    return body


def send_order(host: str, port: int, token: str, order_body: dict) -> Tuple[int, Any]:
    url = f"http://{host}:{port}/kabusapi/sendorder"
    return _rest_json("POST", url, token, order_body)


# ===== Main =================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Verbose", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="実発注せずにドライラン")
    ap.add_argument("--interval-ms", type=int, default=500, help="コマンド放出間隔(ms)")
    ap.add_argument("--per-minute", type=int, default=90, help="1分あたり上限")
    args = ap.parse_args()

    cfg = load_cfg(args.Config)

    host = cfg.get("host", "localhost")
    port = int(cfg.get("port", 18080))
    token = (cfg.get("token") or "").strip()

    # ログ設定
    Path("./logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("./logs/flatten_closeout.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("[FLATTEN] start dry=%s port=%s", args.dry, port)

    # ---- 1) 未約定の取消 ----------------------------------------------------
    limiter = RateLimiter(min_interval_ms=args.interval_ms, max_per_minute=args.per_minute)
    try:
        active = list_active_orders(host, port, token)
        logging.info("[FLATTEN] active_orders=%s", len(active))
        for o in active:
            oid = str(o.get("OrderId") or o.get("Id") or "")
            if not oid:
                continue
            logging.info("[CANCEL] id=%s", oid)
            if not args.dry:
                limiter.wait()
                st, res = cancel_order(host, port, token, oid)
                logging.info("[CANCEL] status=%s res=%s", st, res)
            else:
                logging.info("[CANCEL] dry-run skip")
    except Exception:
        logging.error("[CANCEL] exception\n%s", traceback.format_exc())

    # ---- 2) 全建玉のクローズ -------------------------------------------------
    # テンプレは config の flatten.order_template に置く（例は下に記載）
    order_tpl = (cfg.get("flatten") or {}).get("order_template") or {}
    if not order_tpl:
        logging.warning("[FLATTEN] order_template が config に見つかりません。ドライランで続行します。")

    try:
        poss = list_positions(host, port, token)
        logging.info("[FLATTEN] positions=%s", len(poss))

        # 反対売買の注文を生成
        for p in poss:
            qty = int(p.get("LeavesQty") or p.get("Qty") or p.get("Quantity") or 0)
            if qty <= 0:
                continue
            body = make_close_order_from_template(p, order_tpl)
            logging.info("[CLOSE] %s exch=%s side(close)=%s qty=%s",
                         body.get("Symbol"), body.get("Exchange"), body.get("Side"), body.get("Qty"))
            logging.debug("[CLOSE] body=%s", json.dumps(body, ensure_ascii=False))
            if not args.dry and order_tpl:
                limiter.wait()
                st, res = send_order(host, port, token, body)
                logging.info("[SEND] status=%s res=%s", st, res)
            else:
                logging.info("[SEND] dry-run または order_template 未設定のため送信しません")
    except Exception:
        logging.error("[CLOSE] exception\n%s", traceback.format_exc())

    logging.info("[FLATTEN] done")


if __name__ == "__main__":
    main()
