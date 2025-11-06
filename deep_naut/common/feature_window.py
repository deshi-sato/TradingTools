import numpy as np

def make_window(buf, T=48):
    x = np.asarray(buf[-T:], dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
