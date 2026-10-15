import numpy as np

def smooth(y, sigma=1.2):

    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    x = np.arange(len(y))
    out = np.empty(len(y))
    for i in x:
        w = np.exp(-0.5 * ((x - i) / sigma) ** 2)
        out[i] = float((w * y).sum() / w.sum())
    return out
