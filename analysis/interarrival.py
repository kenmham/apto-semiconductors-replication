"""Complete inter-advance intervals and annual event-age statistics."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
N_NULL = 2000
SEED = 0
MIN_GAPS_WITHIN = 2
NEG_INF = -10**9
def blocks_of(labels, years):
    d = {}
    for i, lab in enumerate(labels):
        d.setdefault(lab, []).append(i)
    return [np.array(v)[np.argsort(years[np.array(v)])] for v in d.values()]

def stats(t, blocks, years):
    gaps_all, gaps_feat, u_all, y_all, acf_a, acf_b = ([], [], [], [], [], [])
    for idx in blocks:
        yy, tt = (years[idx], t[idx])
        marked = np.where(tt == 1, yy, NEG_INF)
        prev = np.concatenate(([NEG_INF], np.maximum.accumulate(marked)[:-1]))
        ok = prev > NEG_INF
        if ok.any():
            u_all.append((yy[ok] - prev[ok]).astype(float))
            y_all.append(tt[ok].astype(int))
        jy = yy[tt == 1]
        if len(jy) >= 2:
            g = np.diff(jy).astype(float)
            gaps_all.append(g)
            if len(g) >= MIN_GAPS_WITHIN:
                gaps_feat.append(g)
            if len(g) >= 2:
                acf_a.append(g[:-1])
                acf_b.append(g[1:])
    gaps = np.concatenate(gaps_all) if gaps_all else np.array([])
    u = np.concatenate(u_all) if u_all else np.array([])
    y = np.concatenate(y_all) if y_all else np.array([])
    s1 = float(gaps.std(ddof=1) / gaps.mean()) if len(gaps) > 2 else np.nan
    cvs = [g.std(ddof=1) / g.mean() for g in gaps_feat if g.mean() > 0]
    s2 = float(np.mean(cvs)) if cvs else np.nan
    s3 = np.nan
    if len(y) > 10 and 0 < y.sum() < len(y):
        lr = LogisticRegression(max_iter=1000, C=1000000.0)
        lr.fit(u.reshape(-1, 1), y)
        s3 = float(lr.coef_[0][0])
    s4 = np.nan
    if acf_a:
        a, b = (np.concatenate(acf_a), np.concatenate(acf_b))
        if len(a) > 3 and a.std() > 0 and (b.std() > 0):
            s4 = float(spearmanr(a, b).correlation)
    return dict(s1=s1, s2=s2, s3=s3, s4=s4, gaps=gaps, u=u, y=y, n_gaps=len(gaps), n_feat_gaps=len(gaps_feat), n_rows_u=len(u), n_acf=int(sum((len(x) for x in acf_a))))
