# -*- coding: utf-8 -*-

import argparse, time, numpy as np, torch
from models.lstm_mini import MiniLSTM
from common.feature_window import make_window
from common.raw_push_stream import iter_features_from_rawpush

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lstm"], default="lstm")
    ap.add_argument("--db", default=r".\db\naut_market_20251031_refeed.db")
    ap.add_argument("--table", default="features_stream")
    ap.add_argument("--symbols", default="6501")
    ap.add_argument("--p-min", type=float, default=0.72)
    ap.add_argument("--ev-floor", type=float, default=0.02)
    ap.add_argument("--spread-max", type=float, default=0.0048)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep-ms", type=int, default=400, help="リプレイ間隔（ms）")
    args = ap.parse_args()

    symbol = args.symbols.split(",")[0].strip()
    rows, feat_cols = iter_features_from_rawpush(args.db, args.table, symbol)
    print(f"[ML] DB={args.db} table={args.table} symbol={symbol} feats={len(feat_cols)} -> {feat_cols[:6]}...")

    model = MiniLSTM(f=len(feat_cols)).eval()
    buf: list[list[float]] = []

    for v in rows:
        buf.append(v)
        if len(buf) < 48:
            continue
        x = torch.from_numpy(make_window(buf, 48)).unsqueeze(0)  # (1,48,F)
        prob = float(model(x).item())
        # ここに EV / スプレッドの既存条件をANDで合流予定
        print(f"[ML] prob={prob:.3f}")
        # オフラインDBの再生：400msペースで進める
        time.sleep(args.sleep_ms / 1000.0)

if __name__ == "__main__":
    main()
