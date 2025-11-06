# -*- coding: utf-8 -*-
import argparse, time, numpy as np, torch, sys, os, datetime as dt
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="raw_push")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--p-min", type=float, default=0.72)
    ap.add_argument("--sleep-ms", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    model, T, ncls = load_model(a.model)
    rows, feat_cols = iter_features_from_rawpush(a.db, a.table, a.symbol)
    F = model.lstm.input_size  # モデルが想定する入力次元
    print(f"[ML] feats={len(feat_cols)} (model uses first {F}); T={T}; ncls={ncls}")

    # 追加特徴のインデクス（存在しなければ -1）
    def idx(name):
        try: return feat_cols.index(name)
        except ValueError: return -1
    ix_ret5 = idx("ret_5s"); ix_ret10 = idx("ret_10s"); ix_retacc = idx("ret_accel_5s")
    ix_vratio = idx("vol_ratio"); ix_vacc = idx("vol_accel"); ix_nhi = idx("near_high_3m"); ix_st1 = idx("status_onehot")

    buf=[]; prev_prob=None
    for feats, ts in rows:
        # モデル入力は先頭F次元のみ（互換）
        feats_trim = feats[:F]
        buf.append(feats_trim)
        if len(buf) < T:
            continue

        x = torch.from_numpy(make_window(buf, T)).unsqueeze(0)
        logits = model(x)
        prob = (torch.softmax(logits, dim=1)[0,1] if ncls==2 else torch.sigmoid(logits)[0]).item()

        # ---- 早期反応ブースト（ルール合成）----
        def g(ix, default=0.0): return (feats[ix] if 0<=ix<len(feats) else default)
        ret5, ret10, retacc = g(ix_ret5), g(ix_ret10), g(ix_retacc)
        vratio, vacc, nhi = g(ix_vratio), g(ix_vacc), g(ix_nhi)
        st1 = g(ix_st1)

        # 軽量スコア（経験則）: 上昇寄与をやや強め
        s = ( 3.2*ret5 + 1.8*retacc + 0.8*ret10
            + 0.6*(vratio-1.0) + 0.4*vacc
            + 0.8*nhi + 0.5*st1 )
        s = 1/(1+np.exp(-(4.0*s)))  # squash

        p_blend = 0.7*prob + 0.3*float(s)

        dprob = None if prev_prob is None else (p_blend - prev_prob)
        prev_prob = p_blend

        ts_str = _fmt_ts(ts)
        if dprob is None:
            print(f"[ML] {ts_str} prob={prob:.3f} p*={p_blend:.3f}")
        else:
            print(f"[ML] {ts_str} prob={prob:.3f} p*={p_blend:.3f} dprob={dprob:+.3f}")

        time.sleep(a.sleep_ms/1000)

if __name__ == "__main__":
    main()
