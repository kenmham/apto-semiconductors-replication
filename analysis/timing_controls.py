"""Full-window, eight-year-block, and post-2006 permutation comparisons."""
import numpy as np
import interarrival as P
NAMES = dict(s1='cv_pooled', s2='cv_within', s3='hazard_slope', s4='gap_acf1')
def run(tag, years, targ, blocks, block_ids=None):
    obs = P.stats(targ, blocks, years)
    rng = np.random.default_rng(P.SEED)
    null = {k: np.full(P.N_NULL, np.nan) for k in NAMES}
    for b in range(P.N_NULL):
        t = targ.copy()
        for idx in blocks:
            if block_ids is None:
                t[idx] = rng.permutation(t[idx])
            else:
                bid = block_ids[idx]
                for u in np.unique(bid):
                    m = bid == u
                    t[idx[m]] = rng.permutation(t[idx[m]])
        r = P.stats(t, blocks, years)
        for k in null:
            null[k][b] = r[k]
    out = []
    print(f"\n=== {tag} ===  rows {len(years)}  advances {int(targ.sum())}  complete gaps {obs['n_gaps']}  features in S2 {obs['n_feat_gaps']}", flush=True)
    for k in NAMES:
        d = null[k][~np.isnan(null[k])]
        lo, hi = np.percentile(d, [2.5, 97.5])
        o, med = (obs[k], float(np.median(d)))
        p = float(min(1.0, 2 * min((d <= o).mean(), (d >= o).mean())))
        v = 'inside' if lo <= o <= hi else 'above' if o > hi else 'below'
        out.append(f'{tag},{k},{NAMES[k]},{o:.6f},{med:.6f},{lo:.6f},{hi:.6f},{p:.4f},{v}')
        print(f'{NAMES[k]:>13}: obs {o:+.4f}  null [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  {v.upper()}', flush=True)
    return out
