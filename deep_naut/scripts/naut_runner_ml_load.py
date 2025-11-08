# -*- coding: utf-8 -*-
import argparse, time, numpy as np, torch, sys, os, datetime as dt, sqlite3
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.feature_window import make_window
from common.raw_push_stream import iter_features_from_rawpush

def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    from models.lstm_mini import MiniLSTM
    ncls = int(ckpt.get("ncls", 2))
    m = MiniLSTM(f=int(ckpt["f"]), ncls=ncls)
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m, int(ckpt.get("T", 48)), ncls

def _fmt_ts(ts):
    try:
        x = float(ts)
        if x > 1e12: x /= 1e9
        elif x > 1e10: x /= 1000.0
        return dt.datetime.fromtimestamp(x).strftime("%H:%M:%S.%f")[:-3]
    except Exception:
        return "??:??:??.???"

def _load_feat_names(path):
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n = line.strip()
            if n: names.append(n)
    return names

class PBoost:
    """安定化したブースト: ギャップで状態リセット + クリップ + tanh"""
    def __init__(self, alpha=0.9, clip=5.0, k=2.0):
        self.alpha = alpha
        self.clip = clip
        self.k = k
        self.s = 0.0
        self.prev_ts = None

    def update(self, base_prob, ts_sec):
        # 時間ギャップが大きければリセット
        if self.prev_ts is not None and (ts_sec - self.prev_ts) > 1.0:
            self.s = 0.0
        self.prev_ts = ts_sec

        # 平滑（0.5中心）
        self.s = self.alpha*self.s + (1.0-self.alpha)*(base_prob - 0.5)
        self.s = float(np.clip(self.s, -self.clip, self.clip))
        # 安定変換（オーバーフロー無し）
        p_star = 0.5*(np.tanh(self.k*self.s) + 1.0)
        return p_star

def _to_sec(ts):
    try:
        x = float(ts)
        if x > 1e12: return x/1e9
        if x > 1e10: return x/1000.0
        return x
    except Exception:
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="raw_push")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--feat-names", help="学習時に保存した特徴名ファイル（順番そのまま使用）", required=True)
    ap.add_argument("--sleep-ms", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-prob", action="store_true", help="確率をDBに書き戻す")
    a = ap.parse_args()

    # モデル読み込み
    model, T, ncls = load_model(a.model)
    F = model.lstm.input_size

    # 学習時の並び（正解リスト）
    want_names = _load_feat_names(a.feat_names)

    # ストリーム（全部入り + 名前一覧）
    rows, have_names = iter_features_from_rawpush(a.db, a.table, a.symbol)

    # 名前インデクスの対応表
    name_to_idx = {n:i for i,n in enumerate(have_names)}
    select_idx = []
    missing = []
    for n in want_names:
        if n in name_to_idx:
            select_idx.append(name_to_idx[n])
        else:
            missing.append(n)

    # 使う列を最終確定
    if missing:
        print(f"[WARN] missing features (not in stream): {missing}")
    if len(select_idx) < F:
        print(f"[WARN] want {F} features but only {len(select_idx)} found; will use first {len(select_idx)}")
    # モデルサイズに合わせてトリム/パディング（パディングは0）
    def trim_or_pad(feats):
        v = [feats[i] for i in select_idx] if select_idx else feats[:F]
        if len(v) < F:
            v = v + [0.0]*(F - len(v))
        elif len(v) > F:
            v = v[:F]
        return v

    print(f"[ML] have={len(have_names)} feats; want={len(want_names)}; model_F={F}")
    # p* ブースト（安定版）
    booster = PBoost(alpha=0.9, clip=5.0, k=2.0)

    prob_conn = None
    if a.write_prob:
        if a.dry_run:
            print("[WARN] --write-prob は dry-run では無効です")
        else:
            prob_conn = sqlite3.connect(a.db)
            prob_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_prob (
                    symbol TEXT NOT NULL,
                    t_exec INTEGER NOT NULL,
                    prob REAL,
                    PRIMARY KEY(symbol, t_exec)
                )
                """
            )
            prob_conn.commit()

    buf=[]; prev_pstar=None
    for feats_all, ts in rows:
        feats = trim_or_pad(feats_all)
        buf.append(feats)
        if len(buf) < T:
            continue

        x = torch.from_numpy(make_window(buf, T)).unsqueeze(0)
        logits = model(x)
        prob = (torch.softmax(logits, dim=1)[0,1] if ncls==2 else torch.sigmoid(logits)[0]).item()

        tsec = _to_sec(ts)
        pstar = booster.update(prob, tsec)
        dprob = None if prev_pstar is None else (pstar - prev_pstar)
        prev_pstar = pstar

        print(f"[ML] {_fmt_ts(ts)} prob={prob:.3f} p*={pstar:.3f}" + ("" if dprob is None else f" dprob={dprob:+.3f}"))
        if prob_conn is not None:
            prob_conn.execute(
                "INSERT OR REPLACE INTO ml_prob (symbol, t_exec, prob) VALUES (?, ?, ?)",
                (a.symbol, int(float(ts)), float(pstar)),
            )
            prob_conn.commit()
        time.sleep(a.sleep_ms/1000)

    if prob_conn is not None:
        prob_conn.close()

if __name__ == "__main__":
    main()
