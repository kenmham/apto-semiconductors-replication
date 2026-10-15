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


def jumps_for_feature(step_years, steps, T):


    sm = {int(y): float(s) for y, s in zip(step_years.tolist(), steps.tolist())}
    ymax = max(sm) if sm else None
    out = []
    for y, g1 in sm.items():
        if g1 <= T:
            continue
        if y + 2 > ymax or (y + 1) not in sm or (y + 2) not in sm:
            continue
        out.append((y, g1, sm[y + 1], sm[y + 2]))
    return out


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

    jr_rows, corr_rows = [], []
    for T in THRESHOLDS:
        rows = []
        for label, step_years, steps in series:
            for y, g1, g2, g3 in jumps_for_feature(step_years, steps, T):
                den3 = g1 + g2 + g3
                den2 = g1 + g2
                r = dict(feature=label, threshold=T, year=y, g1=g1, g2=g2, g3=g3,
                         pct_yr2_of3=(g2 / den3 if den3 > 0 else 0.0),
                         pct_yr3_of3=(g3 / den3 if den3 > 0 else 0.0),
                         pct_yr2_of2=(g2 / den2 if den2 > 0 else 0.0))
                rows.append(r); jr_rows.append(r)
        n = len(rows)
        size = np.array([r["g1"] for r in rows], float)
        p2 = np.array([r["pct_yr2_of3"] for r in rows], float)
        p3 = np.array([r["pct_yr3_of3"] for r in rows], float)
        p23 = p2 + p3
        p2of2 = np.array([r["pct_yr2_of2"] for r in rows], float)

        def corr(a, b):
            if n < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
                return (float("nan"),) * 4
            pr, pp = pearsonr(a, b); sr, sp = spearmanr(a, b)
            return (pr, pp, sr, sp)

        c2 = corr(size, p2); c3 = corr(size, p3); c23 = corr(size, p23)
        cof2 = corr(size, p2of2)
        mean2 = float(np.mean(p2)) if n else float("nan")
        mean3 = float(np.mean(p3)) if n else float("nan")
        zero23 = float(np.mean(p23 == 0)) if n else float("nan")
        corr_rows.append(dict(threshold=T, n_jumps=n,
                              mean_pct_yr2=mean2, mean_pct_yr3=mean3, share_zero_followthrough=zero23,
                              yr2_spearman_r=c2[2], yr2_spearman_p=c2[3],
                              yr3_spearman_r=c3[2], yr3_spearman_p=c3[3],
                              yr23_spearman_r=c23[2], yr23_spearman_p=c23[3],
                              yr2_pearson_r=c2[0], yr2_pearson_p=c2[1],
                              of2_spearman_r=cof2[2], of2_spearman_p=cof2[3]))
        print(f"T={T:.2f} (+{(10**T-1)*100:3.0f}%/yr) | jumps={n:3d} "
              f"mean%yr2={mean2*100:4.1f}% yr3={mean3*100:4.1f}% zero-FT={zero23*100:3.0f}% | "
              f"size~%yr2 Sp r={c2[2]:+.2f} p={c2[3]:.3f} | "
              f"size~%yr3 Sp r={c3[2]:+.2f} p={c3[3]:.3f} | "
              f"size~%(yr2+3) Sp r={c23[2]:+.2f} p={c23[3]:.3f}")

    pl.DataFrame(jr_rows).write_csv(os.path.join(CSV, "second_year.csv"))
    corr_df = pl.DataFrame(corr_rows)
    corr_df.write_csv(os.path.join(CSV, "second_year_corr.csv"))


    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    T = corr_df["threshold"].to_numpy()
    axL.axhline(0, color="k", lw=.8)
    axL.plot(T, corr_df["yr2_spearman_r"].to_numpy(), "o-", color="#4575b4",
             label="size ~ %gain in yr2")
    axL.plot(T, corr_df["yr3_spearman_r"].to_numpy(), "^-", color="#91bfdb",
             label="size ~ %gain in yr3")
    axL.plot(T, corr_df["yr23_spearman_r"].to_numpy(), "s--", color="#d73027",
             label="size ~ %gain in yr2+yr3 (follow-through)")
    for x, y, nn in zip(T, corr_df["yr23_spearman_r"].to_numpy(), corr_df["n_jumps"].to_numpy()):
        if np.isfinite(y):
            axL.annotate(f"n={nn}", (x, y), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=7, color="#333")
    axL.set_xlabel("jump threshold T (log10 step/yr)")
    axL.set_ylabel("rank corr(jump size, year-2 gain share)")
    axL.set_ylim(-1, 1)
    axL.legend(fontsize=8, loc="upper right")
    axL.grid(True, alpha=.25)
    axL.set_title("Do bigger jumps sustain into year 2?\nacross thresholds", fontsize=11)

    T0 = 0.10
    sub = corr_df.filter(pl.col("threshold") == T0)
    dj = pl.read_csv(os.path.join(CSV, "second_year.csv")).filter(pl.col("threshold") == T0)
    S = dj["g1"].to_numpy(); P = (dj["pct_yr2_of3"] + dj["pct_yr3_of3"]).to_numpy()
    axR.scatter(S, P, s=24, alpha=.55, color="#4575b4", edgecolor="none")
    axR.set_xlabel("jump size, year-1 gain (log10 step)")
    axR.set_ylabel("share of 3-yr gain in years 2+3 (follow-through)")
    r0 = float(sub["yr23_spearman_r"][0]); p0 = float(sub["yr23_spearman_p"][0])
    n0 = int(sub["n_jumps"][0]); z0 = float(sub["share_zero_followthrough"][0])
    axR.set_title(f"Jump size vs yr2+yr3 follow-through  (T={T0}, +26%/yr)\n"
                  f"n={n0}, {z0*100:.0f}% pure single-year spikes, Spearman r={r0:+.2f} (p={p0:.2f})",
                  fontsize=11)
    axR.grid(True, alpha=.25)

    fig.suptitle(f"Jump size vs second-year gain share  [{specs.Y0}-{specs.Y1}]", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(IMG, "second_year.png"), dpi=130)
    print(f"\nwrote {os.path.join(CSV, 'second_year.csv')}")
    print(f"wrote {os.path.join(CSV, 'second_year_corr.csv')}")
    print(f"wrote {os.path.join(IMG, 'second_year.png')}")


if __name__ == "__main__":
    main()
