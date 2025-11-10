from flask import Flask, request, jsonify, render_template
from datetime import datetime
import threading
import time
import random
import os
import json

import requests

from .quote_queue import QuoteQueue
from .push_registry import PushRegistry
from .funds_monitor import FundsMonitor

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

# ---- 状態保持 ----
runners = {}   # runner_id -> {symbol, mode, last_seen}
funds = {"cash": 1000000, "positions": [], "pnl_today": 0}
queue_stats = {
    "requests": 0,
    "processed": 0,
    "dropped": 0,
    "push_registered": 0,
    "funds_ok": False,
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SENTINEL_CONFIG_PATH = os.path.join(BASE_DIR, "config", "sentinel.json")

# Quote queue (100ms rate limit)
quote_queue = QuoteQueue(min_interval_ms=100)
push_reg = PushRegistry(
    config_path=os.path.join(os.path.dirname(__file__), "..", "config", "sentinel.json"),
    max_symbols=40,
    ttl_sec=60,
)
funds_mon = FundsMonitor(
    config_path=os.path.join(os.path.dirname(__file__), "..", "config", "sentinel.json"),
    poll_sec=2.0,
    min_cash=300000,
)


# --- kabuステーション REST取得関数 ---
def _fetch_quote_kabu(symbol: str) -> dict:
    """Fetch quote data from kabu Station REST API."""
    try:
        with open(SENTINEL_CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
            cfg = json.load(cfg_file)
        api_cfg = cfg.get("api", {})
        base_url = api_cfg.get("base_url", "").rstrip("/")
        if not base_url:
            raise ValueError("base_url missing in sentinel config")
        timeout = api_cfg.get("timeout_sec", 2.0)
        token_path = api_cfg.get("token_path", "./config/kabu_token.json")
        if not token_path:
            raise ValueError("token_path missing in sentinel config")
        if not os.path.isabs(token_path):
            token_path = os.path.normpath(os.path.join(BASE_DIR, token_path.lstrip("./\\")))

        with open(token_path, "r", encoding="utf-8") as token_file:
            token_data = json.load(token_file)
        token = token_data.get("Token") or token_data.get("token")
        if not token:
            raise ValueError("API token not found in kabu_token.json")

        headers = {"X-API-KEY": token}
        url = f"{base_url}/board/{symbol}"
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        price = data.get("CurrentPrice") or data.get("AskSign") or 0.0
        volume = data.get("TradingVolume") or 0
        return {"symbol": symbol, "ts": ts, "price": price, "volume": volume}
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc)}


quote_queue.set_fetcher(_fetch_quote_kabu)


# ---- API ----
@app.route("/api/runner/heartbeat", methods=["POST"])
def runner_heartbeat():
    data = request.get_json(force=True)
    rid = data.get("runner_id")
    runners[rid] = {
        "symbol": data.get("symbol"),
        "mode": data.get("mode"),
        "last_seen": datetime.now().strftime("%H:%M:%S")
    }
    print(f"[HB] {rid} {data.get('symbol')} {data.get('mode')}")
    return jsonify({"ok": True, "count": len(runners)})


@app.route("/api/funds/snapshot")
def funds_snapshot():
    snap = funds_mon.get_snapshot()
    queue_stats["funds_ok"] = snap.get("ok", False)
    return jsonify(snap)


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        runners=runners,
        funds=funds_mon.get_snapshot(),
        queue=queue_stats,
        now=datetime.now().strftime("%H:%M:%S"),
    )


# ---- 背景処理 ----
def background_mock():
    """ダミー資金の変動を定期更新"""
    while True:
        funds["cash"] += random.randint(-5000, 5000)
        funds["pnl_today"] += random.randint(-2000, 2000)
        time.sleep(3)


@app.route("/api/quote", methods=["POST"])
def get_quote():
    data = request.get_json(force=True)
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    priority = int(data.get("priority", 1))

    result_box = {}

    def _resolve(res):
        result_box["data"] = res

    quote_queue.enqueue(kind="quote", symbol=symbol, priority=priority, resolve=_resolve)

    import time as _t
    deadline = _t.time() + 2.0
    while "data" not in result_box and _t.time() < deadline:
        _t.sleep(0.005)

    if "data" not in result_box:
        return jsonify({"error": "timeout"}), 408

    stats = quote_queue.get_stats()
    queue_stats["requests"] = stats["enq"]
    queue_stats["processed"] = stats["deq"]
    queue_stats["dropped"] = stats["dropped"]

    return jsonify(result_box["data"])


@app.route("/api/push/register", methods=["POST"])
def push_register():
    data = request.get_json(force=True)
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    priority = int(data.get("priority", 1))
    res = push_reg.refresh(symbol, priority)
    push_reg.ensure_registered(symbol)
    queue_stats["push_registered"] = len(push_reg.symbols)
    return jsonify(res)


@app.route("/api/test/ping")
def ping():
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[PING] pong at {ts}")
    return jsonify({"ok": True, "msg": "pong", "time": ts})


if __name__ == "__main__":
    threading.Thread(target=background_mock, daemon=True).start()
    print("[Sentinel] Debug dashboard running at http://127.0.0.1:58900/dashboard")
    app.run(host="127.0.0.1", port=58900)
