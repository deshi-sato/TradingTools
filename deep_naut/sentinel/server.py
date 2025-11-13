from flask import Flask, request, jsonify, render_template
from datetime import datetime
import threading
import time
import random
import os
import json
import glob
import sqlite3
import re
from collections import deque

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
current_mode = {
    "value": "PAPER",
    "updated": datetime.now().strftime("%H:%M:%S"),
}
comm_log = deque(maxlen=50)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SENTINEL_CONFIG_PATH = os.path.join(BASE_DIR, "config", "sentinel.json")
DEFAULT_REF_FEED_DIR = os.path.join(PROJECT_ROOT, "db")

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


def _load_config() -> dict:
    with open(SENTINEL_CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
        return json.load(cfg_file)


def _save_config(cfg: dict) -> None:
    with open(SENTINEL_CONFIG_PATH, "w", encoding="utf-8") as cfg_file:
        json.dump(cfg, cfg_file, ensure_ascii=False, indent=2)


def _ensure_abs_path(path_value: str) -> str:
    if not path_value:
        return DEFAULT_REF_FEED_DIR
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_value))


def _resolve_refeed_dir(cfg: dict | None = None) -> str:
    cfg = cfg or _load_config()
    paper_cfg = cfg.setdefault("paper", {})
    directory = paper_cfg.get("refeed_db_dir") or DEFAULT_REF_FEED_DIR
    directory = _ensure_abs_path(directory)
    paper_cfg["refeed_db_dir"] = directory
    return directory


def _resolve_refeed_db(cfg: dict | None = None) -> str:
    cfg = cfg or _load_config()
    paper_cfg = cfg.setdefault("paper", {})
    current = paper_cfg.get("refeed_db")
    if current:
        current_abs = _ensure_abs_path(current)
        if os.path.isfile(current_abs):
            return current_abs
    directory = _resolve_refeed_dir(cfg)
    pattern = os.path.join(directory, "naut_market_*_refeed.db")
    latest_key = ""
    latest_path = ""
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        match = re.search(r"naut_market_(\d{8})_refeed\.db$", name)
        key = match.group(1) if match else ""
        if key >= latest_key:
            latest_key = key
            latest_path = os.path.abspath(path)
    if latest_path:
        paper_cfg["refeed_db"] = latest_path
        _save_config(cfg)
        return latest_path
    return os.path.join(directory, "naut_market_00000000_refeed.db")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return cur.fetchone() is not None
    except sqlite3.Error:
        return False


def _pick_table(conn: sqlite3.Connection, symbol: str) -> str | None:
    for table in ("refeed_ticks", "features_stream", "raw_push"):
        if not _table_exists(conn, table):
            continue
        try:
            cur = conn.execute(
                f"SELECT 1 FROM {table} WHERE symbol=? LIMIT 1", (symbol,)
            )
            if cur.fetchone():
                return table
        except sqlite3.Error:
            continue
    return None


def _pick_time_col(conn: sqlite3.Connection, table: str) -> str | None:
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cur.fetchall()}
    except sqlite3.Error:
        return None
    for col in ("t_exec", "t_recv", "ts_ms"):
        if col in cols:
            return col
    return None


def _set_mode(value: str):
    current_mode["value"] = value
    current_mode["updated"] = datetime.now().strftime("%H:%M:%S")
    print(f"[MODE] switched to {value}")


def _log_comm(kind: str, detail: str, ok: bool):
    comm_log.appendleft(
        {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "kind": kind,
            "detail": detail,
            "ok": ok,
        }
    )


# --- kabuステーション REST取得関数 ---
def _format_board_symbol(symbol: str) -> str:
    """Return kabu board endpoint symbol (append @exchange if missing)."""
    return symbol if "@" in symbol else f"{symbol}@1"


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
        board_symbol = _format_board_symbol(symbol)
        url = f"{base_url}/board/{board_symbol}"
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        _log_comm("quote", f"{board_symbol} board {resp.status_code}", True)

        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        price = data.get("CurrentPrice") or data.get("AskSign") or 0.0
        volume = data.get("TradingVolume") or 0
        return {"symbol": symbol, "ts": ts, "price": price, "volume": volume}
    except Exception as exc:
        _log_comm("quote", f"{symbol} failed: {exc}", False)
        return {"symbol": symbol, "error": str(exc)}


quote_queue.set_fetcher(_fetch_quote_kabu)


def _kabu_healthcheck() -> tuple[bool, str]:
    """
    ヘルスチェック優先度:
      1) wallet/margin
      2) wallet/cash
      3) board/{symbol}@{market}
    api.health_endpoint で固定も可能（auto/margin/cash/board）
    """
    try:
        with open(SENTINEL_CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
            cfg = json.load(cfg_file)
        api_cfg = cfg.get("api", {})
        base = api_cfg.get("base_url", "").rstrip("/")
        if not base:
            return False, "base_url missing"
        timeout = api_cfg.get("timeout_sec", 2.0)
        pref = (api_cfg.get("health_endpoint") or "auto").lower()

        token_path = api_cfg.get("token_path", "./config/kabu_token.json")
        if not os.path.isabs(token_path):
            token_path = os.path.normpath(os.path.join(BASE_DIR, token_path.lstrip("./\\")))
        with open(token_path, "r", encoding="utf-8") as token_file:
            token_data = json.load(token_file)
        token = token_data.get("Token") or token_data.get("token")
        if not token:
            return False, f"token not found in {token_path}"

        headers = {"X-API-KEY": token}

        def _fmt_err(resp):
            try:
                data = resp.json()
                return f"code={data.get('Code')} msg={data.get('Message')}"
            except Exception:
                return resp.text[:200] if hasattr(resp, "text") else "no body"

        def _try_margin():
            url = f"{base}/wallet/margin"
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                _log_comm("health", "wallet/margin ok", True)
                return True, "wallet/margin ok"
            return False, f"http {resp.status_code}: {_fmt_err(resp)}"

        def _try_cash():
            url = f"{base}/wallet/cash"
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                _log_comm("health", "wallet/cash ok", True)
                return True, "wallet/cash ok"
            return False, f"http {resp.status_code}: {_fmt_err(resp)}"

        def _try_board():
            health_symbol = _format_board_symbol("6501")
            url = f"{base}/board/{health_symbol}"
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                _log_comm("health", f"board/{health_symbol} ok", True)
                return True, f"board/{health_symbol} ok"
            return False, f"http {resp.status_code}: {_fmt_err(resp)}"

        order = {
            "margin": [_try_margin],
            "cash": [_try_cash],
            "board": [_try_board],
            "auto": [_try_margin, _try_cash, _try_board],
        }.get(pref, [_try_margin, _try_cash, _try_board])

        last_err = None
        for fn in order:
            ok, msg = fn()
            if ok:
                return True, msg
            last_err = msg
        _log_comm("health", f"all endpoints failed: {last_err}", False)
        return False, last_err or "healthcheck failed"
    except Exception as exc:
        _log_comm("health", f"error: {exc}", False)
        return False, str(exc)


def _kabu_issue_token() -> tuple[bool, dict]:
    """
    kabuステーション /token を叩いて新トークンを保存。
    戻り:
      (True, {"status":200,"result_code":0,"token":"..."})
      (False,{"status":xxx,"error_code":code,"error_message":"..."})
    """
    try:
        with open(SENTINEL_CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
            cfg = json.load(cfg_file)
        base = cfg["api"]["base_url"].rstrip("/")
        timeout = cfg["api"].get("timeout_sec", 3.0)
        api_password = cfg["api"].get("password")
        if not api_password:
            info = {
                "status": 400,
                "error_code": -1,
                "error_message": "api.password is empty in sentinel.json",
            }
            _log_comm("token", info["error_message"], False)
            return False, info

        url = f"{base}/token"
        payload = {"APIPassword": api_password}
        resp = requests.post(url, json=payload, timeout=timeout)
        status = resp.status_code

        if status == 200:
            data = resp.json()
            result_code = int(data.get("ResultCode", 9999))
            token = data.get("Token")
            if result_code == 0 and token:
                token_path = cfg["api"].get("token_path", "./config/kabu_token.json")
                if not os.path.isabs(token_path):
                    token_path = os.path.normpath(os.path.join(BASE_DIR, token_path.lstrip("./\\")))
                os.makedirs(os.path.dirname(token_path), exist_ok=True)
                with open(token_path, "w", encoding="utf-8") as token_file:
                    json.dump({"Token": token}, token_file, ensure_ascii=False, indent=2)
                _log_comm("token", "issued new token", True)
                return True, {"status": 200, "result_code": 0, "token": token}
            info = {
                "status": 200,
                "error_code": result_code,
                "error_message": "token issue failed (ResultCode!=0)",
            }
            _log_comm("token", info["error_message"], False)
            return False, info

        try:
            err = resp.json()
            code = err.get("Code")
            msg = err.get("Message")
        except Exception:
            code = None
            msg = resp.text[:200] if hasattr(resp, "text") else "no body"
        info = {"status": status, "error_code": code, "error_message": msg}
        _log_comm("token", f"issue failed {status}: {msg}", False)
        return False, info
    except Exception as exc:
        info = {"status": 500, "error_code": -2, "error_message": f"issue error: {exc}"}
        _log_comm("token", info["error_message"], False)
        return False, info


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
    funds_snapshot = funds_mon.get_snapshot() if "funds_mon" in globals() else funds
    try:
        cfg = _load_config()
        paper_cfg = cfg.get("paper", {})
        paper_refeed_current = os.path.basename(paper_cfg.get("refeed_db", "") or "")
    except Exception:
        paper_refeed_current = ""
    return render_template(
        "dashboard.html",
        runners=runners,
        funds=funds_snapshot,
        queue=queue_stats,
        mode=current_mode,
        comms=list(comm_log),
        now=datetime.now().strftime("%H:%M:%S"),
        paper_refeed_current=paper_refeed_current,
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


@app.route("/api/paper/refeed_list")
def paper_refeed_list():
    cfg = _load_config()
    paper_cfg = cfg.setdefault("paper", {})
    directory = _resolve_refeed_dir(cfg)
    files: list[str] = []
    if os.path.isdir(directory):
        pattern = os.path.join(directory, "naut_market_*_refeed.db")
        files = sorted(
            [os.path.basename(path) for path in glob.glob(pattern)],
            reverse=True,
        )
    current_path = paper_cfg.get("refeed_db") or ""
    current_name = os.path.basename(current_path) if current_path else ""
    need_save = False
    if not current_name and files:
        current_name = files[0]
        paper_cfg["refeed_db"] = os.path.join(directory, current_name)
        need_save = True
    elif current_name and files and current_name not in files:
        current_name = files[0]
        paper_cfg["refeed_db"] = os.path.join(directory, current_name)
        need_save = True
    if paper_cfg.get("refeed_db_dir") != directory:
        paper_cfg["refeed_db_dir"] = directory
        need_save = True
    if need_save:
        _save_config(cfg)
    return jsonify({"db_files": files, "current": current_name})


@app.route("/api/paper/set_refeed", methods=["POST"])
def paper_set_refeed():
    data = request.get_json(force=True)
    filename = str(data.get("filename") or "").strip()
    if not filename:
        return jsonify({"ok": False, "error": "filename required"}), 400
    cfg = _load_config()
    directory = _resolve_refeed_dir(cfg)
    target_path = os.path.abspath(os.path.join(directory, filename))
    safe_prefix = os.path.abspath(directory)
    if not target_path.startswith(safe_prefix):
        return jsonify({"ok": False, "error": "invalid path"}), 400
    if not os.path.isfile(target_path):
        return jsonify({"ok": False, "error": "file not found"}), 404
    paper_cfg = cfg.setdefault("paper", {})
    paper_cfg["refeed_db_dir"] = directory
    paper_cfg["refeed_db"] = target_path
    _save_config(cfg)
    return jsonify({"ok": True, "current": filename})


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_payload(row):
    payload_raw = None
    for key in ("payload", "Payload", "data_json", "data"):
        if key in row.keys():
            payload_raw = row[key]
            break
    if payload_raw is None:
        return {}
    if isinstance(payload_raw, (bytes, bytearray)):
        payload_raw = payload_raw.decode("utf-8", errors="ignore")
    if isinstance(payload_raw, str):
        try:
            return json.loads(payload_raw)
        except Exception:
            return {}
    if isinstance(payload_raw, dict):
        return payload_raw
    return {}


def _first_value(row, payload, candidates):
    for key in candidates:
        if key in row.keys():
            val = row[key]
            if val is not None:
                return val
        if payload and key in payload:
            val = payload[key]
            if val is not None:
                return val
    return None


def _row_to_board(row, time_col):
    payload = _parse_payload(row)
    ts_candidate = _coerce_float(row[time_col])
    if ts_candidate is None:
        ts_candidate = _coerce_float(payload.get("timestamp"))
    if ts_candidate is None:
        ts_candidate = time.time()
    if ts_candidate > 1e12:
        ts_candidate = ts_candidate / 1000.0
    bid = _coerce_float(
        _first_value(
            row,
            payload,
            ("bid", "bid1", "bid_price", "BidPrice", "best_bid"),
        )
    )
    ask = _coerce_float(
        _first_value(
            row,
            payload,
            ("ask", "ask1", "ask_price", "AskPrice", "best_ask"),
        )
    )
    last = _coerce_float(
        _first_value(
            row,
            payload,
            ("last", "last_price", "price", "close", "CurrentPrice"),
        )
    )
    if last is None:
        last = bid if bid is not None else ask
    volume = _coerce_int(
        _first_value(
            row,
            payload,
            ("volume", "turnover", "TradingVolume", "size"),
        )
    )
    return {
        "symbol": row["symbol"],
        "t": float(ts_candidate),
        "bid": bid,
        "ask": ask,
        "last": last,
        "volume": volume or 0,
    }


@app.route("/api/feed/board")
def api_feed_board():
    symbol = (request.args.get("symbol") or "").strip()
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400
    try:
        limit = int(request.args.get("limit", 1))
    except ValueError:
        return jsonify({"ok": False, "error": "invalid limit"}), 400
    limit = max(1, min(limit, 100))
    since_raw = request.args.get("since")
    if since_raw:
        try:
            since = float(since_raw)
        except ValueError:
            return jsonify({"ok": False, "error": "invalid since"}), 400
    else:
        since = None

    cfg = _load_config()
    db_path = _resolve_refeed_db(cfg)
    if not os.path.isfile(db_path):
        return jsonify({"ok": False, "error": "refeed db not found"}), 404

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return jsonify({"ok": False, "error": f"db open failed: {exc}"}), 500

    try:
        table = _pick_table(conn, symbol)
        if not table:
            return jsonify({"ok": True, "data": []})
        time_col = _pick_time_col(conn, table)
        if not time_col:
            return jsonify({"ok": False, "error": "time column not found"}), 500

        sql = f"SELECT * FROM {table} WHERE symbol=?"
        params = [symbol]
        if since is not None:
            since_value = since * 1000.0 if time_col == "ts_ms" else since
            sql += f" AND {time_col} > ?"
            params.append(since_value)
        sql += f" ORDER BY {time_col} DESC LIMIT ?"
        params.append(limit)
        try:
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        except sqlite3.Error as exc:
            return jsonify({"ok": False, "error": f"query failed: {exc}"}), 500
    finally:
        conn.close()

    if not rows:
        return jsonify({"ok": True, "data": []})

    boards = []
    for row in rows:
        board = _row_to_board(row, time_col)
        boards.append(board)
    if limit == 1:
        board = boards[0]
        board["ok"] = True
        return jsonify(board)
    return jsonify({"ok": True, "data": boards})


@app.route("/api/dev/config")
def dev_config():
    try:
        with open(SENTINEL_CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
            cfg = json.load(cfg_file)
        api_cfg = cfg.get("api", {})
        base = api_cfg.get("base_url", "").rstrip("/")
        token_path = api_cfg.get("token_path", "./config/kabu_token.json")
        if not os.path.isabs(token_path):
            token_path = os.path.normpath(os.path.join(BASE_DIR, token_path.lstrip("./\\")))
        password_set = bool(api_cfg.get("password"))
        return jsonify(
            {
                "base_url": base,
                "token_path": token_path,
                "password_set": password_set,
                "mode": current_mode["value"],
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/mode", methods=["GET"])
def get_mode():
    return jsonify(dict(current_mode))


@app.route("/api/mode", methods=["POST"])
def set_mode():
    data = request.get_json(force=True)
    want = (data.get("mode") or "").upper()
    if want not in ("LIVE", "LIVE-DRY", "PAPER"):
        return jsonify({"ok": False, "error": "invalid mode"}), 400

    if want in ("LIVE", "LIVE-DRY"):
        ok, info = _kabu_issue_token()
        if not ok:
            _set_mode("PAPER")
            return (
                jsonify(
                    {
                        "ok": False,
                        "mode": current_mode["value"],
                        "status": info.get("status"),
                        "code": info.get("error_code"),
                        "message": info.get("error_message"),
                    }
                ),
                503,
            )
        ok2, msg2 = _kabu_healthcheck()
        if not ok2:
            _set_mode("PAPER")
            return (
                jsonify(
                    {
                        "ok": False,
                        "mode": current_mode["value"],
                        "status": 503,
                        "code": -3,
                        "message": f"live check failed: {msg2}",
                    }
                ),
                503,
            )
        _set_mode(want)
        return jsonify({"ok": True, "mode": current_mode["value"], "status": 200})

    _set_mode(want)
    return jsonify({"ok": True, "mode": current_mode["value"]})


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
