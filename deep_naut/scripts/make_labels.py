import numpy as np, argparse

def label_future(mid, spread, H=32, fee=0.0, band=1.2):
    y = np.full_like(mid, -1, dtype=np.int8)  # -1:中立, 1:上, 0:下
    eps = spread * band + fee
    up = np.where(mid[H:] - mid[:-H] >= eps[:-H])[0]
    dn = np.where(mid[:-H] - mid[H:] >= eps[:-H])[0]
    y[up] = 1; y[dn] = 0
    return y

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--band", type=float, default=1.2)
    ap.add_argument("--out", default="exports/labels.npy")
    a=ap.parse_args()
    m=np.load(a.meta)
    y=label_future(m["mid"], m["spread"], a.H, a.fee, a.band)
    np.save(a.out, y)
    print("OK:", a.out, "counts:", {k:int((y==k).sum()) for k in (-1,0,1)})
