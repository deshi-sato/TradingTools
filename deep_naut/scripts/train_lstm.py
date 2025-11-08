# -*- coding: utf-8 -*-

import numpy as np, torch, torch.nn as nn, argparse, math, random

FEATURE_KEYS = [
    "price","mid","spread","imb","vwap_dev","vol_rate","val_rate",
    "depth_sell5","depth_buy5","microprice",
    "st_no","st_up","st_down",
    "d_bid_qty","d_ask_qty","is_trade","trade_size","imb_rate","spread_chg","mid_chg",
    "ret_5s","ret_10s","ret_accel_5s",
    "vol_ratio","vol_accel",
    "near_high_3m","status_onehot",
    "price_ma3","vol_ma3","imb_ma3","candle_up",
]
INPUT_DIM = len(FEATURE_KEYS)
MODEL_PATH = "models/lstm_3905_v2.pt"

def row_to_features(row):
    """dict(row) -> list[float], 欠損は0埋め"""
    vec = []
    for key in FEATURE_KEYS:
        val = row.get(key)
        vec.append(0.0 if val is None else float(val))
    return vec

class MiniLSTM(nn.Module):
    def __init__(self, f, h=64, l=2, ncls=2):
        super().__init__()
        self.lstm = nn.LSTM(f, h, num_layers=l, batch_first=True)
        self.fc = nn.Linear(h, ncls)
    def forward(self, x):
        y,_ = self.lstm(x)
        return self.fc(y[:,-1,:])

def sliding(X, y, T=48, stride=1):
    xs, ys = [], []
    for i in range(0, len(X)-T-1, stride):
        yy = y[i+T]
        if yy == -1:
            continue
        xs.append(X[i:i+T]); ys.append(yy)
    return np.array(xs, np.float32), np.array(ys, np.int64)

def train_eval(X, y, T=48, epochs=5, bs=256, lr=1e-3):
    if X.shape[1] != INPUT_DIM:
        raise ValueError(f"FEATURE_KEYS({INPUT_DIM}) と入力次元({X.shape[1]})が一致しません。")
    Xs, ys = sliding(X, y, T)
    n = len(ys); idx = np.arange(n); np.random.shuffle(idx)
    cut = int(n*0.8)
    tr, va = idx[:cut], idx[cut:]
    Xt, Xv = Xs[tr], Xs[va]; yt, yv = ys[tr], ys[va]

    model = MiniLSTM(f=INPUT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def loop(Xb, yb, train=True):
        model.train(train)
        tot=0; correct=0
        for i in range(0, len(yb), bs):
            xb = torch.from_numpy(Xb[i:i+bs])
            yb2= torch.from_numpy(yb[i:i+bs])
            logits = model(xb)
            loss = loss_fn(logits, yb2)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            pred = logits.argmax(1)
            tot += len(yb2); correct += int((pred==yb2).sum())
        return correct/tot if tot else 0.0

    for ep in range(1, epochs+1):
        acc_tr = loop(Xt, yt, True)
        acc_va = loop(Xv, yv, False)
        print(f"epoch {ep}: acc_tr={acc_tr:.3f} acc_va={acc_va:.3f}")

    torch.save({"state_dict": model.state_dict(), "f": INPUT_DIM, "T": T, "ncls": 2}, MODEL_PATH)
    print(f"saved: {MODEL_PATH}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--X", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=5)
    a=ap.parse_args()
    X=np.load(a.X); y=np.load(a.y)
    train_eval(X,y,T=a.T,epochs=a.epochs)
