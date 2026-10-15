import os
import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import specs
from specs import SPECS, scan, frontier_series, step_gains

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
specs.SRC = os.environ["APTO_DATA"]
CSV = os.path.join(os.path.dirname(HERE), "data")
IMG = os.path.join(os.path.dirname(HERE), "outputs", "figures")


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def episodes_for_feature(step_years, steps, T):


    out = []
    run = []
    for y, st in zip(step_years.tolist(), steps.tolist()):
        hi = st > T
        contiguous = run and (y - run[-1][0] == 1)
        if hi and (not run or contiguous):
            run.append((y, st))
        elif hi:
            out.append(run); run = [(y, st)]
        else:
            if run:
                out.append(run); run = []
    if run:
        out.append(run)
    eps = []
    for r in out:
        ys = [y for y, _ in r]; ss = [s for _, s in r]
        eps.append(dict(length=len(r), peak=max(ss), total=float(sum(ss)),
                        year_start=min(ys), year_end=max(ys)))
    return eps


def main():
    lf = scan()

    series = []
    for col, sub, label, direction in SPECS:
        s = frontier_series(lf, col, sub, direction)
        if s is None:
            continue
        step_years, steps, total = step_gains(s)
        series.append((label, step_years, steps))
    print(f"features with a usable frontier: {len(series)}/{len(SPECS)}")

    ep_rows, corr_rows = [], []
    for T in THRESHOLDS:
        eps = []
        for label, step_years, steps in series:
            for e in episodes_for_feature(step_years, steps, T):
                e = {**e, "feature": label, "threshold": T}
                eps.append(e)
                ep_rows.append(e)
        n = len(eps)
        length = np.array([e["length"] for e in eps], float)
        peak = np.array([e["peak"] for e in eps], float)
        tot = np.array([e["total"] for e in eps], float)
        share1 = float(np.mean(length == 1)) if n else float("nan")
        maxlen = int(length.max()) if n else 0


        def corr(a, b):
            if n < 3 or len(np.unique(b)) < 2 or len(np.unique(a)) < 2:
                return (float("nan"), float("nan"))
            pr, pp = pearsonr(a, b)
            sr, sp = spearmanr(a, b)
            return (pr, pp, sr, sp)

        cp = corr(peak, length)
        ct = corr(tot, length)
        row = dict(threshold=T, n_episodes=n, share_len1=share1, max_len=maxlen,
                   peak_pearson_r=cp[0], peak_pearson_p=cp[1] if len(cp) > 1 else float("nan"),
                   peak_spearman_r=cp[2] if len(cp) > 2 else float("nan"),
                   peak_spearman_p=cp[3] if len(cp) > 3 else float("nan"),
                   total_spearman_r=ct[2] if len(ct) > 2 else float("nan"),
                   total_spearman_p=ct[3] if len(ct) > 3 else float("nan"))
        corr_rows.append(row)
        print(f"T={T:.2f} (+{(10**T-1)*100:3.0f}%/yr) | eps={n:3d} "
              f"len1={share1*100:3.0f}% maxlen={maxlen} | "
              f"peak~len: Pearson r={cp[0]:+.2f} p={cp[1] if len(cp)>1 else float('nan'):.3f} "
              f"Spearman r={cp[2] if len(cp)>2 else float('nan'):+.2f} "
              f"p={cp[3] if len(cp)>3 else float('nan'):.3f}")

    ep_df = pl.DataFrame(ep_rows)
    corr_df = pl.DataFrame(corr_rows)
    ep_df.write_csv(os.path.join(CSV, "episodes.csv"))
    corr_df.write_csv(os.path.join(CSV, "episode_corr.csv"))


    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    T = corr_df["threshold"].to_numpy()
    axL.axhline(0, color="k", lw=.8)
    axL.plot(T, corr_df["peak_spearman_r"].to_numpy(), "o-", color="#4575b4",
             label="peak size ~ length (Spearman)")
    axL.plot(T, corr_df["total_spearman_r"].to_numpy(), "s--", color="#d73027",
             label="summed gain ~ length (Spearman)")
    for x, y, nn in zip(T, corr_df["peak_spearman_r"].to_numpy(), corr_df["n_episodes"].to_numpy()):
        if np.isfinite(y):
            axL.annotate(f"n={nn}", (x, y), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=7, color="#333")
    axL.set_xlabel("high-improvement threshold T (log10 step/yr)")
    axL.set_ylabel("rank correlation with episode length")
    axL.set_ylim(-1, 1)
    axL.legend(fontsize=8, loc="upper right")
    axL.grid(True, alpha=.25)
    axL.set_title("Does jump size track episode length?\nacross thresholds", fontsize=11)


    T0 = 0.10
    sub = ep_df.filter(pl.col("threshold") == T0)
    L = sub["length"].to_numpy().astype(float); P = sub["peak"].to_numpy()
    rng = np.linspace(-0.15, 0.15, len(L)) if len(L) else np.array([])
    axR.scatter(L + rng, P, s=22, alpha=.5, color="#4575b4", edgecolor="none")
    axR.set_xlabel("episode length (years, jittered)")
    axR.set_ylabel("peak jump size in episode (log10 step)")
    axR.set_xticks(range(1, (int(L.max()) if len(L) else 1) + 1))
    share1_0 = float(np.mean(L == 1)) if len(L) else float("nan")
    axR.set_title(f"Peak jump size vs episode length  (T={T0}, +26%/yr)\n"
                  f"n={len(L)} episodes, {share1_0*100:.0f}% are single-year", fontsize=11)
    axR.grid(True, alpha=.25)

    fig.suptitle(f"Jump size vs high-improvement episode length  [{specs.Y0}-{specs.Y1}]",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(IMG, "episode_length.png"), dpi=130)
    print(f"\nwrote {os.path.join(CSV, 'episodes.csv')}")
    print(f"wrote {os.path.join(CSV, 'episode_corr.csv')}")
    print(f"wrote {os.path.join(IMG, 'episode_length.png')}")


if __name__ == "__main__":
    main()
