#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stream_microbatch_Offline.py
- kabu API 未接続で、既存 DB の raw_push を再生して features / orderbook を再生成
- 速度変更（-Speed）、無睡眠（-NoSleep）、銘柄絞り込み（-Symbols）対応
- BUY/SELL 特徴量の分布確認やスケーリング調整の検証に使う
"""

from __future__ import annotations
import argparse, json, sqlite3, time, queue, threading, sys
from pathlib import Path
from typing import Optional, Iterable

# 既存ロジックを再利用
try:
    import scripts.stream_microbatch as live
except ModuleNotFoundError:
    pkg_dir = Path(__file__).resolve().parent
    project_root = pkg_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import scripts.stream_microbatch as live


class OfflineFeeder(threading.Thread):
    def __init__(
        self,
        db_path: str,
        buf: "live.PushBuffer",
        stop_event: threading.Event,
        symbols: Optional[set[str]] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        speed: float = 5.0,
        nosleep: bool = False,
        batch_size: int = 1000,
    ):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.buf = buf
        self.stop_event = stop_event
        self.symbols = symbols
        self.tmin = tmin
        self.tmax = tmax
        self.speed = max(1e-6, float(speed))
        self.nosleep = bool(nosleep)
        self.batch_size = batch_size

    def _iter_rows(self) -> Iterable[tuple[float, Optional[str]]]:
        con = sqlite3.connect(self.db_path)
        try:
            params, where = [], []
            if self.tmin is not None:
                where.append("t_recv >= ?")
                params.append(self.tmin)
            if self.tmax is not None:
                where.append("t_recv <= ?")
                params.append(self.tmax)
            sql = "SELECT t_recv, payload FROM raw_push"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY t_recv ASC"
            cur = con.execute(sql, params)
            for row in cur:
                yield float(row[0]), row[1]
        finally:
            con.close()

    def run(self):
        prev_t = None
        count = 0
        for t_recv, payload in self._iter_rows():
            if self.stop_event.is_set():
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            # live 側と同じ正規化
            sym_raw = (
                obj.get("Symbol")
                or obj.get("IssueCode")
                or obj.get("SymbolCode")
                or obj.get("SymbolName")
                or ""
            )
            sym = live._normalize_sym(str(sym_raw))
            if not sym:
                continue
            if self.symbols and sym not in self.symbols:
                continue
            obj["Symbol"] = sym  # 明示的に格納
            self.buf.put(obj)
            count += 1

            if not self.nosleep:
                if prev_t is not None:
                    dt = max(0.0, (t_recv - prev_t) / self.speed)
                    # 軽いスロットリング（突発的な超短間隔をつぶしすぎない範囲）
                    time.sleep(dt if dt < 0.5 else 0.5)
                prev_t = t_recv
        # 終了
        # ここでは buf を閉じず、呼び出し側の stop_event で後続が drain する


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-DB", required=True, help="再生元 DB (naut_market_YYYYMMDD_refeed.db)"
    )
    ap.add_argument(
        "-DatasetId", help="新規登録する dataset_id。省略時はREFyyyymmdd_hhmm自動発行"
    )
    ap.add_argument("-Symbols", help="カンマ区切りで銘柄絞り込み（例: 7203,6758）")
    ap.add_argument("-From", type=float, help="t_recv 開始(UNIX秒)")
    ap.add_argument("-To", type=float, help="t_recv 終了(UNIX秒)")
    ap.add_argument(
        "-Speed", type=float, default=5.0, help="再生速度倍率（5.0で5倍速）"
    )
    ap.add_argument("-NoSleep", action="store_true", help="間引き無し（全速で流す）")
    ap.add_argument("-Verbose", type=int, default=1)
    ap.add_argument("-CodeVersion", default="offline_replay")
    args = ap.parse_args()

    db_path = str(Path(args.DB).resolve())
    # 1) DBテーブルを保証（不足していれば作成）
    live.ensure_tables(db_path)

    # 2) dataset_id を registry に登録
    cfg_snapshot = {"mode": "OFFLINE_REPLAY", "source": db_path}
    ds_id, inserted = live.register_dataset(
        Path(db_path), cfg_snapshot, args.CodeVersion, now_jst=live.get_jst_now()
    )
    if args.DatasetId:
        ds_id = args.DatasetId
    print(f"[offline] dataset_id: {ds_id} ({'registered' if inserted else 'exists'})")
    print(f"[offline] db: {db_path}")

    # 3) パイプ構築（WSは起動しない）
    stop_event = threading.Event()
    buf = live.PushBuffer(maxsize=2000)
    board_q: queue.Queue = queue.Queue(maxsize=10000)

    # orderbook saver & features worker は live の実装をそのまま使用
    saver = live.OrderbookSaver(db_path, board_q, stop_event)
    saver.start()
    feat_thread = threading.Thread(
        target=live.features_worker,
        args=(buf, db_path, board_q, stop_event, ds_id),
        daemon=True,
    )
    feat_thread.start()

    # 4) raw_push を時系列で再生
    symbols = set(s.strip() for s in args.Symbols.split(",")) if args.Symbols else None
    feeder = OfflineFeeder(
        db_path=db_path,
        buf=buf,
        stop_event=stop_event,
        symbols=symbols,
        tmin=args.From,
        tmax=args.To,
        speed=args.Speed,
        nosleep=args.NoSleep,
    )
    feeder.start()

    try:
        while feeder.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        feeder.join(2.0)
        feat_thread.join(2.0)
        saver.join(2.0)
        # ざっくり統計
        stats = buf.snapshot_stats(reset=False)
        print(
            f"[offline] pushbuf final: put={stats['put']} get={stats['get']} drop={stats['drop']} qsize={stats['qsize']}"
        )


if __name__ == "__main__":
    main()
