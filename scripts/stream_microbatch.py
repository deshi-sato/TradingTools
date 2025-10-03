#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stream_microbatch.py (板+ティック集約 + keepalive監視)
- kabu WebSocket PUSH を raw_q に貯める
- ParserWorker が decode/json して tick_q / board_q へ
- OrderbookSaver が orderbook_snapshot をバッチ保存
- tick_batch / features_stream に集約値を保存
"""

import argparse, json, logging, queue, sqlite3, sys, threading, time, socket
import ctypes, atexit, os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException
except Exception:
    create_connection=None
    WebSocketTimeoutException=type("WebSocketTimeoutException",(),{})
    WebSocketConnectionClosedException=type("WebSocketConnectionClosedException",(),{})

logger=logging.getLogger(__name__)

# helpers -------------------------------------------------
_singleton_handle = None
_pidfile_path = None

def _cleanup_pid():
    global _singleton_handle, _pidfile_path
    try:
        if _singleton_handle:
            ctypes.windll.kernel32.CloseHandle(_singleton_handle)
    except Exception:
        pass
    try:
        if _pidfile_path and _pidfile_path.exists():
            _pidfile_path.unlink()
    except Exception:
        pass

def _normalize_sym(s: str) -> str:
    return s.split("@", 1)[0].strip()

def singleton_guard(tag: str):
    """同名タスクを多重起動させないガード。"""
    global _singleton_handle, _pidfile_path
    name = f"Global\\{tag}"
    _singleton_handle = ctypes.windll.kernel32.CreateMutexW(None, False, name)
    if ctypes.GetLastError() == 183:
        print(f"[ERROR] {tag} already running", file=sys.stderr)
        sys.exit(1)
    pid_dir = Path("runtime/pids"); pid_dir.mkdir(parents=True, exist_ok=True)
    _pidfile_path = pid_dir / f"{tag}.pid"
    try: _pidfile_path.write_text(str(os.getpid()), encoding="utf-8")
    except Exception: pass
    atexit.register(_cleanup_pid)

def _to_float(x:Any)->Optional[float]:
    try: return float(x) if x is not None else None
    except Exception: return None

def normalize_timestamp(value:Any)->str:
    if not value:
        return datetime.now().isoformat(timespec="milliseconds")
    ts=str(value)
    if "T" not in ts and " " in ts: ts=ts.replace(" ","T",1)
    if len(ts)==8 and ts.count(":")==2:
        ts=f"{datetime.now().strftime('%Y-%m-%d')}T{ts}"
    return ts

def load_json_utf8(path):
    with open(path,"r",encoding="utf-8-sig") as f: return json.load(f)

EXPECTED_FEATURE_COLUMNS=(
    "ticker","ts","last_price","last_qty","ticks","upticks","downticks",
    "vol_sum","vwap","price_min","price_max",
)

DDL_TB="""
CREATE TABLE IF NOT EXISTS tick_batch(
  ticker TEXT,
  ts_window_start TEXT,
  ts_window_end TEXT,
  ticks INT,
  upticks INT,
  downticks INT,
  vol_sum REAL,
  last_price REAL
);
"""

DDL_OB="""
CREATE TABLE IF NOT EXISTS orderbook_snapshot(
  ticker TEXT,
  ts TEXT,
  bid1 REAL,
  ask1 REAL,
  over_sell_qty INT,
  under_buy_qty INT,
  sell_top3 INT,
  buy_top3 INT
);
"""

DDL_FEAT_CREATE="""
CREATE TABLE IF NOT EXISTS features_stream(
  ticker TEXT,
  ts TEXT,
  last_price REAL,
  last_qty INT,
  ticks INT,
  upticks INT,
  downticks INT,
  vol_sum REAL,
  vwap REAL,
  price_min REAL,
  price_max REAL
);
"""

def ensure_tables(db:str):
    conn=sqlite3.connect(db)
    try:
        with conn:
            conn.executescript(DDL_TB+DDL_OB)
            _ensure_features_table(conn)
    finally:
        conn.close()

def _ensure_features_table(conn:sqlite3.Connection):
    cur=conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='features_stream'")
    if cur.fetchone() is None:
        conn.execute(DDL_FEAT_CREATE); return
    cur=conn.execute("PRAGMA table_info(features_stream)")
    cols=[row[1] for row in cur.fetchall()]
    if cols!=list(EXPECTED_FEATURE_COLUMNS):
        conn.execute("DROP TABLE IF EXISTS features_stream")
        conn.execute(DDL_FEAT_CREATE)

def insert_tick_batch(conn,rows):
    if rows: conn.executemany("INSERT INTO tick_batch VALUES (?,?,?,?,?,?,?,?)",rows)

def insert_features(conn,rows):
    if rows: conn.executemany("INSERT INTO features_stream VALUES (?,?,?,?,?,?,?,?,?,?,?)",rows)

def insert_orderbook_snapshot(conn,rows):
    if rows:
        conn.executemany(
            "INSERT INTO orderbook_snapshot (ticker, ts, bid1, ask1, over_sell_qty, under_buy_qty, sell_top3, buy_top3) VALUES (?,?,?,?,?,?,?,?)",
            rows)

# RawWS ---------------------------------------------------
class RawWS(threading.Thread):
    def __init__(self,url,headers,stop_event,raw_q,recv_timeout=15,keepalive_sec=10,idle_reconnect_sec=180):
        super().__init__(daemon=True)
        self.url,self.headers,self.stop_event,self.raw_q=url,headers,stop_event,raw_q
        self.recv_timeout,self.keepalive_sec,self.idle_reconnect_sec=recv_timeout,keepalive_sec,idle_reconnect_sec
    def run(self):
        backoff=1.0
        while not self.stop_event.is_set():
            ws=None
            try:
                ws=create_connection(self.url,header=self.headers,timeout=6.0,
                    sockopt=((socket.SOL_SOCKET,socket.SO_KEEPALIVE,1),
                             (socket.SOL_SOCKET,socket.SO_RCVBUF,4*1024*1024)))
                ws.settimeout(self.recv_timeout)
                logger.info("connected raw")
                backoff=1.0
                last_ping=0.0; missed=0
                while not self.stop_event.is_set():
                    now=time.time()
                    if now-last_ping>=self.keepalive_sec:
                        try: ws.ping(); missed=0
                        except Exception: missed+=1
                        if missed>=2: raise RuntimeError("ping fail x2")
                        last_ping=now
                    try:
                        msg=ws.recv()
                    except WebSocketTimeoutException:
                        continue
                    except (WebSocketConnectionClosedException,OSError):
                        raise RuntimeError("ws closed")
                    if not msg: continue
                    try: self.raw_q.put(msg,timeout=0.001)
                    except queue.Full:
                        try: self.raw_q.get_nowait()
                        except Exception: pass
                        try: self.raw_q.put_nowait(msg)
                        except Exception: pass
            except Exception as e:
                if self.stop_event.is_set(): break
                logger.warning("raw ws error: %s (reconnect %.1fs)",e,backoff)
                time.sleep(backoff); backoff=min(backoff*2,30)
            finally:
                if ws:
                    try: ws.close()
                    except Exception: pass

# Parser --------------------------------------------------
class ParserWorker(threading.Thread):
    def __init__(self, raw_q, out_q, stop_event, symbols_set, price_guard):
        super().__init__(daemon=True)
        self.raw_q = raw_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.symbols_set = symbols_set
        # ガードは切り分けのため無効化（必要時に設定値を渡して有効化）
        self.price_guard = {}
        self.ob_q = None
        self.last_vol = {}

        # debug counters
        self.t_last = time.time()
        self.c_got=self.c_tick=self.c_board=0
        self.c_drop_sym=self.c_drop_watch=self.c_drop_price=self.c_drop_guard=0

    def _sym_of(self, obj):
        raw = str(
            obj.get("Symbol")
            or obj.get("IssueCode")
            or obj.get("SymbolCode")
            or obj.get("SymbolName")
            or ""
        ).strip()
        return _normalize_sym(raw)   # ← ここを追加

    def _price_of(self, obj):
        for k in ("CurrentPrice","Price","ExecutionPrice","Close","Last"):
            v = obj.get(k)
            if v is not None:
                try: return float(v)
                except: pass
        return None

    def run(self):
        while not self.stop_event.is_set():
            try:
                msg = self.raw_q.get(timeout=0.2)
            except queue.Empty:
                now=time.time()
                if now - self.t_last >= 3.0:
                    self.t_last = now
                    logging.info("DBG parser: got=%d tick=%d board=%d drop[sym=%d watch=%d price=%d guard=%d] raw_q=%d tick_q=%d",
                        self.c_got, self.c_tick, self.c_board,
                        self.c_drop_sym, self.c_drop_watch, self.c_drop_price, self.c_drop_guard,
                        getattr(self.raw_q,'qsize',lambda:0)(), getattr(self.out_q,'qsize',lambda:0)())
                    self.c_got=self.c_tick=self.c_board=self.c_drop_sym=self.c_drop_watch=self.c_drop_price=self.c_drop_guard=0
                continue

            s = msg.decode("utf-8","replace") if isinstance(msg,bytes) else str(msg)
            try:
                obj = json.loads(s)
            except Exception:
                continue
            self.c_got += 1

            # --- Board 判定 ---
            if ("OverSellQty" in obj) or ("UnderBuyQty" in obj) or ("Sell1Qty" in obj) or ("Buy1Qty" in obj) \
               or ("BidPrice" in obj and "AskPrice" in obj):
                self.c_board += 1
                if self.ob_q is not None:
                    sym = self._sym_of(obj)
                    if sym and (not self.symbols_set or sym in self.symbols_set):
                        bid = _to_float(obj.get("BidPrice"))
                        ask = _to_float(obj.get("AskPrice"))
                        over = int(float(obj.get("OverSellQty") or 0))
                        under = int(float(obj.get("UnderBuyQty") or 0))
                        sell3 = int(float(obj.get("Sell1Qty") or 0)) + int(float(obj.get("Sell2Qty") or 0)) + int(float(obj.get("Sell3Qty") or 0))
                        buy3  = int(float(obj.get("Buy1Qty")  or 0)) + int(float(obj.get("Buy2Qty")  or 0)) + int(float(obj.get("Buy3Qty")  or 0))
                        snap = (sym, datetime.now().isoformat(timespec="milliseconds"), bid, ask, over, under, sell3, buy3)
                        try: self.ob_q.put_nowait(snap)
                        except queue.Full: pass
                continue  # 板はここで終わり

            # --- Tick 判定（緩和）---
            sym = self._sym_of(obj)
            if not sym:
                self.c_drop_sym += 1; continue
            if self.symbols_set and sym not in self.symbols_set:
                self.c_drop_watch += 1; continue

            price = self._price_of(obj)
            if price is None:
                self.c_drop_price += 1; continue

            # 出来高差分
            qty = 0
            tv = obj.get("TradingVolume") or obj.get("Volume")
            try:
                tv = int(float(tv)) if tv is not None else None
                if tv is not None:
                    prev = self.last_vol.get(sym)
                    self.last_vol[sym] = tv
                    if prev is not None:
                        qty = max(0, tv - prev)
            except Exception:
                pass

            ts = obj.get("CurrentPriceTime") or obj.get("ExecutionDateTime") or obj.get("TradeTime") or obj.get("TransactTime") or obj.get("Time")
            ts = normalize_timestamp(ts)

            try:
                self.out_q.put_nowait((sym, float(price), int(qty), ts))
                self.c_tick += 1
            except queue.Full:
                pass

# Orderbook saver -----------------------------------------
class OrderbookSaver(threading.Thread):
    def __init__(self,db_path:str,board_q,stop_event,batch_size:int=100,flush_interval:float=0.5):
        super().__init__(daemon=True)
        self.db_path=db_path; self.board_q=board_q; self.stop_event=stop_event
        self.batch_size=batch_size; self.flush_interval=flush_interval
    def run(self):
        conn=sqlite3.connect(self.db_path,check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;"); conn.execute("PRAGMA synchronous=NORMAL;")
        batch=[]; last_flush=time.monotonic()
        try:
            while not self.stop_event.is_set() or not self.board_q.empty() or batch:
                try: item=self.board_q.get(timeout=0.1); batch.append(item)
                except queue.Empty: pass
                now=time.monotonic()
                if batch and (len(batch)>=self.batch_size or now-last_flush>=self.flush_interval or self.stop_event.is_set()):
                    insert_orderbook_snapshot(conn,batch); conn.commit(); batch.clear(); last_flush=now
        except Exception as exc:
            logger.exception("orderbook saver error: %s",exc)
        finally:
            if batch:
                try: insert_orderbook_snapshot(conn,batch); conn.commit()
                except Exception as exc: logger.exception("orderbook saver final flush failed: %s",exc)
            conn.close()

# main ----------------------------------------------------
def main():
    singleton_guard("stream_microbatch")
    ap=argparse.ArgumentParser()
    ap.add_argument("-Config",required=True)
    ap.add_argument("-Verbose",type=int,default=1)
    args=ap.parse_args()

    cfg=load_json_utf8(args.Config)
    symbols=list(cfg.get("symbols",[]))
    host=str(cfg.get("host","localhost")); port=int(cfg.get("port",18080))
    token=str(cfg.get("token") or "")
    if not symbols: sys.exit("ERROR: symbols empty")
    if not token: sys.exit("ERROR: token empty")

    price_guard=cfg.get("price_guard",{})
    window_ms=int(cfg.get("window_ms",300))
    db_path=cfg.get("db_path","rss_snapshot.db")
    log_path=cfg.get("log_path","logs/stream_microbatch.log")
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)

    logging.basicConfig(level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path,encoding="utf-8"),logging.StreamHandler(sys.stdout)])
    logger.info("boot symbols=%s db=%s",symbols,db_path)

    ensure_tables(db_path)
    conn=sqlite3.connect(db_path,check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;"); conn.execute("PRAGMA synchronous=NORMAL;")

    tick_q=queue.Queue(maxsize=int(cfg.get("tick_queue_max",20000)))
    board_q=queue.Queue(maxsize=int(cfg.get("orderbook_queue_max",10000)))
    raw_q=queue.Queue(maxsize=int(cfg.get("ws_raw_queue_max",50000)))
    stop_event=threading.Event()

    # 明示的に ALL を購読
    url=f"ws://{host}:{port}/kabusapi/websocket?filter=ALL"
    headers=[f"X-API-KEY: {token}"]
    rx=RawWS(url,headers,stop_event,raw_q); rx.start()

    # 置き換え前: parser = ParserWorker(raw_q, tick_q, stop_event, set(symbols), price_guard)
    norm_syms = { _normalize_sym(s) for s in symbols }
    parser = ParserWorker(raw_q, tick_q, stop_event, norm_syms, price_guard)
    parser.ob_q = board_q
    parser.start()

    orderbook_saver=OrderbookSaver(db_path,board_q,stop_event); orderbook_saver.start()

    last_price={s:None for s in symbols}
    window_s=window_ms/1000.0
    next_cut=time.monotonic()+window_s

    try:
        while True:
            ticks_buf={s:[] for s in symbols}
            while time.monotonic()<next_cut:
                try: s,price,qty,ts_iso=tick_q.get(timeout=0.01)
                except queue.Empty: continue
                if s in ticks_buf: ticks_buf[s].append((price,qty,ts_iso))

            ts_start_iso=datetime.now().isoformat(timespec="milliseconds")
            tick_rows=[]; feat_rows=[]
            for s in symbols:
                arr=ticks_buf[s]
                if not arr: continue
                upt=dwn=0; vol=0.0; vwap_num=0.0
                price_min=price_max=arr[0][0]
                prev_price=last_price[s] if last_price[s] is not None else arr[0][0]
                for price,qty,_ in arr:
                    if price>prev_price: upt+=1
                    elif price<prev_price: dwn+=1
                    prev_price=price
                    vol+=qty; vwap_num+=price*qty
                    if price<price_min: price_min=price
                    if price>price_max: price_max=price
                last_price[s]=arr[-1][0]; ts_end_iso=arr[-1][2]
                tick_rows.append((s,ts_start_iso,ts_end_iso,len(arr),upt,dwn,vol,last_price[s]))
                last_qty=int(arr[-1][1])
                vwap=vwap_num/vol if vol>0 else None
                feat_rows.append((s,ts_end_iso,last_price[s],last_qty,len(arr),upt,dwn,vol,vwap,price_min,price_max))

            with conn:
                insert_tick_batch(conn,tick_rows); insert_features(conn,feat_rows)
            if tick_rows:
                logger.info("batch ticks=%s feats=%s",sum(r[3] for r in tick_rows),len(feat_rows))

            now=time.monotonic(); next_cut+=window_s
            if next_cut<now: next_cut=now+window_s

    except KeyboardInterrupt:
        logger.info("stopped")
    finally:
        stop_event.set()
        rx.join(1.0); parser.join(1.0); orderbook_saver.join(1.0)
        conn.close()

if __name__=="__main__":
    main()
