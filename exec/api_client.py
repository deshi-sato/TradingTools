# exec/api_client.py
"""Minimal kabu station HTTP client with auto token issue/refresh."""

from __future__ import annotations
import os
import json
import requests
from typing import Any, Dict

# ---- Config (ENV可変) -------------------------------------------------------
BASE = os.environ.get("KABU_BASE_URL", "http://localhost:18080").rstrip("/")
AUTO_REFRESH = os.environ.get("KABU_AUTO_TOKEN", "1").lower() in ("1","true","yes","on")
TIMEOUT_GET  = float(os.environ.get("KABU_HTTP_TIMEOUT_GET",  "5"))
TIMEOUT_POST = float(os.environ.get("KABU_HTTP_TIMEOUT_POST", "8"))

# KABU_API_PW / KABU_API_PASSWORD のどちらでも可（API設定画面の「APIパスワード」）
PW_KEYS = ("KABU_API_PW", "KABU_API_PASSWORD")

# ---- Helpers ----------------------------------------------------------------
def _api_key() -> str:
    """優先: KABU_TOKEN → 互換: KABU_API_KEY"""
    return os.environ.get("KABU_TOKEN") or os.environ.get("KABU_API_KEY") or ""

def _hdr() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    tok = _api_key()
    if tok:
        h["X-API-KEY"] = tok
    return h

def _get_env_pw() -> str:
    for k in PW_KEYS:
        v = os.environ.get(k)
        if v:
            return v
    return ""

def issue_token_from_env_pw() -> str:
    """ENVの APIパスワードから /token を叩き、新トークンをENVに保存して返す。無ければ空文字。"""
    pw = _get_env_pw()
    if not pw:
        return ""
    url = f"{BASE}/kabusapi/token"
    r = requests.post(url, headers={"Content-Type": "application/json"},
                      data=json.dumps({"APIPassword": pw}), timeout=TIMEOUT_POST)
    r.raise_for_status()
    tok = r.json().get("Token") or ""
    if tok:
        # 互換性のため両方に入れる（既存コードがどちらを見ても通る）
        os.environ["KABU_TOKEN"] = tok
        os.environ["KABU_API_KEY"] = tok
    return tok

def ensure_token() -> None:
    """ヘッダに載せるキーが無ければ、ENVのAPIパスワードから発行を試みる。"""
    if _api_key():
        return
    issue_token_from_env_pw()

def _request(method: str, path: str, *, retry_on_401: bool = True, **kwargs) -> requests.Response:
    """共通リクエスト: 401なら一度だけトークン再発行→再送（AUTO_REFRESH時）。"""
    ensure_token()
    url = f"{BASE}{path}"
    timeout = kwargs.pop("timeout", TIMEOUT_GET if method.upper() == "GET" else TIMEOUT_POST)

    r = requests.request(method, url, headers=_hdr(), timeout=timeout, **kwargs)
    if r.status_code == 401 and AUTO_REFRESH and retry_on_401:
        if issue_token_from_env_pw():
            r = requests.request(method, url, headers=_hdr(), timeout=timeout, **kwargs)
    r.raise_for_status()
    return r

# ---- Public API -------------------------------------------------------------
def get_board(symbol: str, exchange: int = 1) -> Dict[str, Any]:
    """
    board/{symbol}@{exchange} を返す（東証=1, 名証=3, 福証=5, 札証=6）。
    symbol が既に "7011@1" 形式なら exchange は無視。
    """
    sym = symbol if "@" in symbol else f"{symbol}@{exchange}"
    r = _request("GET", f"/kabusapi/board/{sym}")
    return r.json()

def send_order(payload: Dict[str, Any]) -> Dict[str, Any]:
    """/sendorder へPOST（payloadはkabu REST仕様に準拠）。"""
    r = _request("POST", "/kabusapi/sendorder", data=json.dumps(payload))
    return r.json()

def get_token() -> Dict[str, Any]:
    """
    明示的に /token を叩いて発行＆ENVへ保存して返す。
    事前に KABU_API_PW か KABU_API_PASSWORD をセットしておくこと。
    """
    pw = _get_env_pw()
    if not pw:
        raise RuntimeError("KABU_API_PW (or KABU_API_PASSWORD) is not set")
    url = f"{BASE}/kabusapi/token"
    r = requests.post(url, headers={"Content-Type": "application/json"},
                      data=json.dumps({"APIPassword": pw}), timeout=TIMEOUT_POST)
    r.raise_for_status()
    jd = r.json()
    tok = jd.get("Token") or ""
    if tok:
        os.environ["KABU_TOKEN"] = tok
        os.environ["KABU_API_KEY"] = tok
    return jd

# ---- Ranking ---------------------------------------------------------------
def get_ranking(rtype: int, exchange_division: int = 1, count: int = 20):
    """
    kabuステーションのランキングAPI。
    Args:
        rtype: ランキング種別 (例: 1=売買代金, 2=出来高, 3=値上がり率, 4=値下がり率)
        exchange_division: 市場 (1=東証, 3=名証, 5=福証, 6=札証)
        count: 返す最大件数（APIの返却が多い場合もクライアント側で上限適用）
    Returns:
        List[Tuple[symbol:str, exchange:int]]
    """
    payload = {"Type": int(rtype), "ExchangeDivision": int(exchange_division)}
    r = _request("POST", "/kabusapi/ranking", data=json.dumps(payload))
    jd = r.json() or {}
    # 形は環境により差があるので、配列本体を頑健に拾う
    items = jd.get("Ranking") or jd.get("ranking") or jd
    out = []
    if isinstance(items, list):
        for it in items[:count]:
            sym = str(it.get("Symbol") or it.get("symbol") or "")
            ex  = it.get("Exchange") or it.get("exchange") or exchange_division
            try:
                ex = int(ex)
            except Exception:
                ex = exchange_division
            if sym:
                out.append((sym, ex))
    return out
