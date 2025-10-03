#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stream_microbatch.py (一次バッファ化 + keepalive修正)
- kabu WebSocket PUSH 生データを raw_q にためる
- ParserWorker が decode/json → tick_q へ
- tick_batch / features_stream に保存
"""

import argparse, json, logging, queue, sqlite3, sys, threading, time, socket
import ctypes, atexit, os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException
except Exception:
    create_connection=None
    WebSocketTimeoutException=type("WebSocketTimeoutException",(),{})
    WebSocketConnectionClosedException=type("WebSocketConnectionClosedException",(),{})

logger=logging.getLogger(__name__)

# helpers -------------------------------------------------
# --- singleton guard: 二重起動防止（Windows Mutex + PIDファイル） ---
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

def singleton_guard(tag: str):
    """同名タスクが動作中なら即終了。"""
    global _singleton_handle, _pidfile_path
    # Mutex で重複検出
    name = f"Global\\{tag}"
    _singleton_handle = ctypes.windll.kernel32.CreateMutexW(None, False, name)
    if ctypes.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        print(f"[ERROR] {tag} already running", file=sys.stderr)
        sys.exit(1)

    # PIDファイルも置く（異常終了時の痕跡削除用）
    pid_dir = Path("runtime/pids"); pid_dir.mkdir(parents=True, exist_ok=True)
    _pidfile_path = pid_dir / f"{tag}.pid"
    try:
        _pidfile_path.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass

    atexit.register(_cleanup_pid)

def _to_float(x:Any)->Optional[float]:
    try: return float(x) if x is not None else None
    except: return None

def spread_bp(bid,ask):
    if not bid or not ask or bid<=0 or ask<=0: return None
    return (ask-bid)/((ask+bid)/2.0)*10000.0

def depth_imbalance_calc(buy3,sell3):
    tot=buy3+sell3
    return (buy3-sell3)/tot if tot>0 else 0.0

def uptick_ratio(u,d):
    tot=u+d
    return u/tot if tot>0 else 0.0

def load_json_utf8(path):
    with open(path,"r",encoding="utf-8-sig") as f: return json.load(f)

# DB ------------------------------------------------------
DDL_TB="""CREATE TABLE IF NOT EXISTS tick_batch(
  ticker TEXT, ts_window_start TEXT, ts_window_end TEXT,
  ticks INT, upticks INT, downticks INT, vol_sum REAL, last_price REAL);"""
DDL_FEAT="""CREATE TABLE IF NOT EXISTS features_stream(
  ticker TEXT, ts TEXT, uptick_ratio REAL, vol_sum REAL, spread_bp REAL,
  buy_top3 INT, sell_top3 INT, depth_imbalance REAL,
  burst_buy INT, burst_sell INT, burst_score REAL,
  streak_len INT, surge_vol_ratio REAL, last_signal_ts TEXT);"""

def ensure_tables(db):
    conn=sqlite3.connect(db)
    with conn: conn.executescript(DDL_TB+DDL_FEAT)
    conn.close()

def insert_tick_batch(conn,rows):
    if rows: conn.executemany("INSERT INTO tick_batch VALUES (?,?,?,?,?,?,?,?)",rows)
def insert_features(conn,rows):
    if not rows: return
    cols=["ticker","ts","uptick_ratio","vol_sum","spread_bp","buy_top3","sell_top3",
          "depth_imbalance","burst_buy","burst_sell","burst_score","streak_len",
          "surge_vol_ratio","last_signal_ts"]
    sql=f"INSERT INTO features_stream ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
    data=[tuple(r.get(c) for c in cols) for r in rows]
    conn.executemany(sql,data)

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
                last_rx=time.time(); last_ping=0.0; missed=0
                while not self.stop_event.is_set():
                    now=time.time()
                    if now-last_ping>=self.keepalive_sec:
                        try: ws.ping(); missed=0
                        except: missed+=1; 
                        if missed>=2: raise RuntimeError("ping fail x2")
                        last_ping=now
                    try:
                        msg=ws.recv(); last_rx=now
                    except WebSocketTimeoutException:
                        continue
                    except (WebSocketConnectionClosedException,OSError): raise RuntimeError("ws closed")
                    if not msg: continue
                    try: self.raw_q.put(msg,timeout=0.001)
                    except queue.Full:
                        try: self.raw_q.get_nowait()
                        except: pass
                        try: self.raw_q.put_nowait(msg)
                        except: pass
            except Exception as e:
                if self.stop_event.is_set(): break
                logger.warning("raw ws error: %s (reconnect %.1fs)",e,backoff)
                time.sleep(backoff); backoff=min(backoff*2,30)
            finally:
                if ws:
                    try: ws.close()
                    except: pass

# Parser --------------------------------------------------
class ParserWorker(threading.Thread):
    def __init__(self,raw_q,out_q,stop_event,symbols_set,price_guard):
        super().__init__(daemon=True)
        self.raw_q,self.out_q,self.stop_event=raw_q,out_q,stop_event
        self.symbols_set,self.price_guard=symbols_set,price_guard
        self.last_vol={}
    def run(self):
        while not self.stop_event.is_set():
            try: msg=self.raw_q.get(timeout=0.1)
            except queue.Empty: continue
            s=msg.decode("utf-8","replace") if isinstance(msg,bytes) else str(msg)
            if int(time.time())%5==0: print("RAW:",s[:120])
            try: obj=json.loads(s)
            except: continue
            sym=str(obj.get("Symbol") or obj.get("IssueCode") or "").strip()
            if not sym or (self.symbols_set and sym not in self.symbols_set): continue
            price=_to_float(obj.get("CurrentPrice") or obj.get("Price"))
            if price is None: continue
            lo_hi=self.price_guard.get(sym,(None,None))
            if isinstance(lo_hi,(list,tuple)) and len(lo_hi)==2:
                lo,hi=lo_hi
                if (lo and price<lo) or (hi and price>hi): continue
            qty=0; tv_raw=obj.get("TradingVolume") or obj.get("Volume")
            try:
                tv=int(float(tv_raw)) if tv_raw is not None else None
                if tv is not None:
                    prev=self.last_vol.get(sym); self.last_vol[sym]=tv
                    if prev is not None: qty=max(0,tv-prev)
            except: pass
            ts=obj.get("CurrentPriceTime") or obj.get("TradeTime") or obj.get("TransactTime") or obj.get("Time")
            if not ts: ts=datetime.now().isoformat(timespec="milliseconds")
            else:
                ts=str(ts)
                if "T" not in ts and " " in ts: ts=ts.replace(" ","T",1)
                if len(ts)==8 and ts.count(":")==2: ts=f"{datetime.now().strftime('%Y-%m-%d')}T{ts}"
            try: self.out_q.put((sym,float(price),int(qty),ts),timeout=0.01)
            except queue.Full: pass

# misc ----------------------------------------------------
def within_market_window(spec):
    if not spec: return True
    try:
        s,e=spec.split("-")
        sh=datetime.strptime(s,"%H:%M").time(); eh=datetime.strptime(e,"%H:%M").time()
        now=datetime.now().time()
        return sh<=now<=eh
    except: return True

# main ----------------------------------------------------
def main():
    singleton_guard("stream_microbatch")   # ← main 冒頭に追加
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
    market_window=cfg.get("market_window")
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
    raw_q=queue.Queue(maxsize=int(cfg.get("ws_raw_queue_max",50000)))
    stop_event=threading.Event()
    url=f"ws://{host}:{port}/kabusapi/websocket"; headers=[f"X-API-KEY: {token}"]
    rx=RawWS(url,headers,stop_event,raw_q); rx.start()
    parser=ParserWorker(raw_q,tick_q,stop_event,set(symbols),price_guard); parser.start()
    last_price={s:None for s in symbols}; window_s=window_ms/1000.0; next_cut=time.monotonic()+window_s
    try:
        while True:
            if market_window and not within_market_window(market_window):
                time.sleep(0.2); continue
            ticks_buf={s:[] for s in symbols}
            while time.monotonic()<next_cut:
                try: s,price,qty,ts_iso=tick_q.get(timeout=0.01); 
                except queue.Empty: continue
                if s in ticks_buf: ticks_buf[s].append((price,qty,ts_iso))
            ts_start_iso=datetime.now().isoformat(timespec="milliseconds")
            tick_rows=[]; feat_rows=[]
            for s in symbols:
                arr=ticks_buf[s]
                if arr:
                    upt=dwn=0; vol=0.0; prev=last_price[s] if last_price[s] else arr[0][0]
                    for price,qty,_ in arr:
                        if price>prev: upt+=1
                        elif price<prev: dwn+=1
                        prev=price; vol+=qty
                    last_price[s]=arr[-1][0]; ts_end_iso=arr[-1][2]
                    tick_rows.append((s,ts_start_iso,ts_end_iso,len(arr),upt,dwn,vol,last_price[s]))
                    try:
                        ob=conn.execute("""select bid1,ask1,buy_top3,sell_top3 from orderbook_snapshot
                                             where ticker=? and ts<=? order by ts desc limit 1""",(s,ts_end_iso)).fetchone()
                    except sqlite3.OperationalError: ob=None
                    if ob:
                        b1,a1,bt,st=ob
                        spr=spread_bp(b1,a1); imb=depth_imbalance_calc(int(bt),int(st))
                        feat_rows.append({"ticker":s,"ts":ts_end_iso,"uptick_ratio":uptick_ratio(upt,dwn),
                            "vol_sum":vol,"spread_bp":spr,"buy_top3":int(bt),"sell_top3":int(st),
                            "depth_imbalance":imb,"burst_buy":0,"burst_sell":0,"burst_score":0.0,
                            "streak_len":0,"surge_vol_ratio":1.0,"last_signal_ts":""})
            with conn: insert_tick_batch(conn,tick_rows); insert_features(conn,feat_rows)
            if tick_rows: logger.info("batch ticks=%s feats=%s",sum(r[3] for r in tick_rows),len(feat_rows))
            now=time.monotonic(); next_cut+=window_s
            if next_cut<now: next_cut=now+window_s
    except KeyboardInterrupt: logger.info("stopped")
    finally: stop_event.set(); rx.join(1.0); parser.join(1.0); conn.close()

if __name__=="__main__": main()
