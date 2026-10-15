import os
import numpy as np
import polars as pl
from scipy import stats

import hero_style as H

DATA = os.environ["APTO_DATA"]
FEATURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "feature_selection.csv")
CROP = (0.5, 99.5)
MIN_PARTS = 500
NU = 60
TROUGH_MAX_U = 0.88
KDE_CAP = 20000
N_BOOT = 1000
N_NULL = 1000
SEED = 0
UU = np.linspace(0, 1, NU)


def smooth_density(g):

    lo, hi = np.percentile(g, CROP)
    if hi <= lo:
        return None, None
    NB = 400
    edges = np.linspace(lo, hi, NB + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(g, bins=edges)
    iqr = np.subtract(*np.percentile(g, [75, 25]))
    h = 0.9 * min(g.std(), iqr / 1.34 + 1e-9) * g.size ** (-1 / 5)
    if h <= 0:
        return None, None
    sig = max(1.0, h / (grid[1] - grid[0]))
    r = int(max(1, round(4 * sig)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    d = np.convolve(hist.astype(float), k / k.sum(), mode="same")
    return grid, d


def depth_of(g):

    grid, d = smooth_density(g)
    if grid is None:
        return None
    mi = int(np.argmax(d))
    if mi >= len(grid) - 5 or d[mi] <= 0:
        return None
    gt, dt = grid[mi:], d[mi:] / d[mi]
    span = gt[-1] - gt[0]
    if span <= 0:
        return None
    u = (gt - gt[0]) / span
    prof = np.interp(UU, u, dt)
    seg = UU <= TROUGH_MAX_U
    ti = int(np.argmin(np.where(seg, prof, np.inf)))
    depth = float(prof[ti:].max() - prof[ti])
    return depth, float(UU[ti]), prof


def load_features():
    sel = pl.read_csv(FEATURES).filter(pl.col("selected"))
    feats = list(zip(sel["column"], sel["subcategory"], sel["direction"], sel["label"]))
    by_sub = {}
    for sub in sorted(set(s for _, s, _, _ in feats)):
        cols = sorted(set(c for c, s, _, _ in feats if s == sub))
        by_sub[sub] = pl.scan_parquet(DATA).filter(pl.col("subcategory") == sub).select(cols).collect()
    return feats, by_sub


def main():
    rng = np.random.default_rng(SEED)
    feats, by_sub = load_features()
    rows = []
    for i, (c, sub, d, label) in enumerate(feats):
        v = by_sub[sub][c].to_numpy().astype(float)
        v = v[np.isfinite(v)]
        v = v[v > 0]
        if v.size < MIN_PARTS:
            continue
        g_full = np.log10(v)
        if d == "lo":
            g_full = -g_full

        if g_full.size > KDE_CAP:
            g = g_full[rng.choice(g_full.size, KDE_CAP, replace=False)]
        else:
            g = g_full
        obs = depth_of(g)
        if obs is None:
            continue
        depth, tu, _ = obs
        n = g.size


        bd = np.empty(N_BOOT)
        for b in range(N_BOOT):
            r = depth_of(g[rng.integers(0, n, n)])
            bd[b] = r[0] if r is not None else np.nan
        bd = bd[np.isfinite(bd)]
        ci_lo, ci_hi = np.percentile(bd, [2.5, 97.5])


        a, loc, scale = stats.skewnorm.fit(g)
        nd = np.empty(N_NULL)
        for b in range(N_NULL):
            samp = stats.skewnorm.rvs(a, loc=loc, scale=scale, size=n, random_state=rng)
            r = depth_of(samp)
            nd[b] = r[0] if r is not None else np.nan
        nd = nd[np.isfinite(nd)]
        null_med = float(np.median(nd))
        null_ref = float(np.percentile(nd, 97.5))

        clears = bool(ci_lo > null_ref)
        rows.append(dict(column=c, subcategory=sub, direction=d, label=label,
                         n_parts=int(v.size), n_kde=int(n), trough_u=round(tu, 4),
                         depth=round(depth, 5), depth_lo=round(float(ci_lo), 5),
                         depth_hi=round(float(ci_hi), 5), null_median=round(null_med, 5),
                         null_ref=round(null_ref, 5), clears=clears))
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(feats)} features processed")

    df = pl.DataFrame(rows).sort("depth", descending=True)
    df.write_csv("valley_depth.csv")
    n_feat = df.height
    n_clear = int(df["clears"].sum())
    print(f"\nfeatures with a usable upper tail: {n_feat}")
    print(f"clear the unimodal null (boot floor > null ceiling): {n_clear} ({n_clear / n_feat:.1%})")

    print("wrote valley_depth.csv")

    _plot(df)


def _plot(df):
    d = df.sort("depth", descending=True)
    depth = d["depth"].to_numpy()
    lo = d["depth_lo"].to_numpy()
    hi = d["depth_hi"].to_numpy()
    null_ref = d["null_ref"].to_numpy()
    clears = d["clears"].to_numpy()
    n = len(depth)
    n_clear = int(clears.sum())
    x = np.arange(n)

    def build(size):
        s = H.apply(size)
        fig, ax = H.new(size, 13, 6.4)

        ax.vlines(x, lo, hi, color=H.FAINT, lw=1.4 * s["scale"], zorder=1)
        ax.scatter(x[clears], depth[clears], s=16 * s["scale"] ** 2, color=H.SIGNAL,
                   zorder=3, label=f"clears the unimodal null ({n_clear})")
        ax.scatter(x[~clears], depth[~clears], s=12 * s["scale"] ** 2, color=H.NULL,
                   zorder=2, label=f"does not clear ({n - n_clear})")

        ax.plot(x, null_ref, color=H.FRONTIER, lw=1.6 * s["scale"], ls="--",
                label="unimodal null ceiling (97.5th pct)", zorder=2)
        ax.axhline(0, color=H.INK, lw=0.8 * s["scale"])
        ax.set_xlabel("features, ordered by valley depth")
        ax.set_ylabel("valley depth\n(ceiling shelf − dead-zone floor, mode-relative)")
        ax.set_title(f"How many frontier shoulders survive their null\n"
                     f"{n_clear} of {n} features ({n_clear / n:.0%}) show a valley deeper than a "
                     f"unimodal reference of the same support")
        ax.legend(loc="upper right")
        H.despine(ax)
        fig.tight_layout()
        return fig

    H.render(build, "valley_depth")
    print("wrote images/valley_depth_poster.png and images/valley_depth_slide.png")


if __name__ == "__main__":
    main()
