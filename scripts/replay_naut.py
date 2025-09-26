# scripts/replay_naut.py
from __future__ import annotations
import argparse, sqlite3, sys, os, logging
from datetime import datetime
from typing import Dict, Any, Iterable, Optional

# 1) 設定読み込み（BOM対応の共通関数）
try:
    from scripts.common_config import load_json_utf8
except Exception:
    # 最低限のローダ（BOM許容）
    import json, codecs

    def load_json_utf8(path: str) -> Dict[str, Any]:
        with codecs.open(path, "r", "utf-8-sig") as f:
            return json.load(f)


# 2) 可能なら naut_runner の判定関数を流用
_use_runner = False
try:
    from scripts.naut_runner import is_buy_signal as runner_is_buy
    from scripts.naut_runner import is_sell_signal as runner_is_sell
    from scripts.naut_runner import compute_base_score, apply_burst_and_decide

    _use_runner = True
except Exception:
    # フォールバック：コア条件のみ（burst/bonusは無視）
    def runner_is_buy(f: Dict[str, Any], thr: Dict[str, Any]) -> bool:
        ur = f.get("uptick_ratio")
        vol = f.get("vol_sum")
        sp = f.get("spread_bp")
        imb = f.get("depth_imbalance")
        if ur is None or vol is None or sp is None:
            return False
        cond_up = ur >= thr["BUY_UPTICK_THR"]
        cond_vol = vol > thr["BUY_VOL_MIN"]
        cond_sp = sp <= thr["BUY_SPREAD_MAX"]
        cond_imb = (imb is not None and imb > 0) if thr["BUY_REQUIRE_IMB"] else True
        return cond_up and cond_vol and cond_sp and cond_imb

    def runner_is_sell(f: Dict[str, Any], thr: Dict[str, Any]) -> bool:
        ur = f.get("uptick_ratio")
        imb = f.get("depth_imbalance")
        if ur is None:
            return False
        cond_up = ur <= thr["SELL_UPTICK_THR"]
        cond_imb = (imb is not None and imb < 0) if thr["SELL_REQUIRE_IMB"] else True
        return cond_up and cond_imb

    def compute_base_score(side: str, f: Dict[str, Any]) -> float:
        # 簡易スコア（チューニング専用の目安値）
        ur = f.get("uptick_ratio") or 0.0
        return (1.0 - ur) if side == "SELL" else ur

    def apply_burst_and_decide(
        side: str, base_score: float, f: Dict[str, Any]
    ) -> Dict[str, Any]:
        # フォールバック：そのまま返す
        return {"permit": True, "score": float(base_score), "why": "fallback-no-burst"}


def setup_logger(out_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    logger = logging.getLogger("replay")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(out_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def read_rows(
    conn: sqlite3.Connection, day: str, symbols: Optional[Iterable[str]]
) -> Iterable[Dict[str, Any]]:
    params = [day]
    sym_clause = ""
    if symbols:
        syms = tuple(str(s) for s in symbols)
        sym_clause = f" AND ticker IN {syms}"
    q = f"""
    SELECT *
    FROM features_stream
    WHERE substr(ts,1,10)=?
    {sym_clause}
    ORDER BY ts ASC
    """
    cur = conn.execute(q, params)
    cols = [d[0] for d in cur.description]
    for row in cur:
        yield dict(zip(cols, row))

# thr 読み込みのすぐ下に追加
def _vol_min_by_code(ticker: str) -> int:
    # 簡易実装：設定値をそのまま返す（必要なら銘柄コード別ルールに拡張）
    return int(thr["BUY_VOL_MIN"])


def main():
    ap = argparse.ArgumentParser(
        description="Replay BUY/SELL signals on historical features_stream"
    )
    ap.add_argument("-Config", required=True, help="path to stream_settings.json")
    ap.add_argument("-Date", required=True, help="YYYY-MM-DD to replay")
    ap.add_argument("-Symbols", default="", help="comma separated tickers (optional)")
    args = ap.parse_args()

    conf = load_json_utf8(args.Config)
    db_path = conf.get("db") or conf.get("db_path") or "rss_snapshot.db"
    symbols = [s.strip() for s in args.Symbols.split(",") if s.strip()] or None

    # 閾値を設定（SELL/BUYとも）
    naut = ((conf.get("settings") or {}).get("naut")) or {}
    thr = {
        "BUY_UPTICK_THR": float(naut.get("BUY_UPTICK_THR", 0.65)),
        "BUY_REQUIRE_IMB": bool(naut.get("BUY_REQUIRE_IMB", False)),
        "BUY_VOL_MIN": int(naut.get("BUY_VOL_MIN", 1200)),
        "BUY_SPREAD_MAX": float(naut.get("BUY_SPREAD_MAX", 5.0)),
        "SELL_UPTICK_THR": float(naut.get("SELL_UPTICK_THR", 0.50)),
        "SELL_REQUIRE_IMB": bool(naut.get("SELL_REQUIRE_IMB", False)),
    }

    log_path = os.path.join("logs", f"replay_{args.Date}.log")
    log = setup_logger(log_path)
    log.info(f"[MODE] replay date={args.Date} db={db_path} symbols={symbols or 'ALL'}")
    log.info(
        f"[RULE] BUY require_imb={thr['BUY_REQUIRE_IMB']} thr={thr['BUY_UPTICK_THR']} vol_min={thr['BUY_VOL_MIN']} spread_max={thr['BUY_SPREAD_MAX']}"
    )
    log.info(
        f"[RULE] SELL require_imb={thr['SELL_REQUIRE_IMB']} thr={thr['SELL_UPTICK_THR']}"
    )
    if _use_runner:
        log.info("[INFO] using naut_runner decision path (apply_burst_and_decide)")

    conn = sqlite3.connect(db_path)
    buy_cnt = sell_cnt = 0

    for f in read_rows(conn, args.Date, symbols):
        # --- BUY ---
        try:
            try:
                ok_buy = runner_is_buy(
                    f,
                    buy_uptick_thr = thr["BUY_UPTICK_THR"],
                    buy_require_imb= thr["BUY_REQUIRE_IMB"],
                    buy_vol_min    = thr["BUY_VOL_MIN"],
                    buy_spread_max = thr["BUY_SPREAD_MAX"],
                    vol_min_by_code= _vol_min_by_code,
                )
            except TypeError:
                # 古いシグネチャ互換
                ok_buy = runner_is_buy(f, thr)

            if ok_buy:
                base = compute_base_score("BUY", f)
                r = apply_burst_and_decide("BUY", base, f)
                if r.get("permit", False):
                    buy_cnt += 1
                    log.info(f"[BUY] {f['ticker']} ts={f['ts']} up={f.get('uptick_ratio')} "
                            f"vol={f.get('vol_sum')} sp={f.get('spread_bp')} score={r.get('score'):.4f}")
        except Exception as e:
            log.error(f"[EXC] BUY {f.get('ticker')} ts={f.get('ts')} err={e}")

        # --- SELL ---
        try:
            try:
                ok_sell = runner_is_sell(
                    f,
                    sell_uptick_thr = thr["SELL_UPTICK_THR"],
                    sell_require_imb= thr["SELL_REQUIRE_IMB"],
                    vol_min_by_code = _vol_min_by_code,
                )
            except TypeError:
                ok_sell = runner_is_sell(f, thr)

            if ok_sell:
                base = compute_base_score("SELL", f)
                r = apply_burst_and_decide("SELL", base, f)
                if r.get("permit", False):
                    sell_cnt += 1
                    log.info(f"[SELL] {f['ticker']} ts={f['ts']} up={f.get('uptick_ratio')} "
                            f"imb={f.get('depth_imbalance')} score={r.get('score'):.4f}")
        except Exception as e:
            log.error(f"[EXC] SELL {f.get('ticker')} ts={f.get('ts')} err={e}")

    log.info(f"[SUM] BUY={buy_cnt} SELL={sell_cnt} on {args.Date}")


if __name__ == "__main__":
    main()
