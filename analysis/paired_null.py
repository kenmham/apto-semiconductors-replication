import os, sys
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

import jump_cv as J
from specs import Y0, Y1, MIN_PER_YEAR, MIN_YEARS

IMG = os.path.join(os.path.dirname(HERE), "outputs", "figures")
os.makedirs(IMG, exist_ok=True)


HORIZON    = J.HORIZON
JUMP_FLOOR = J.JUMP_FLOOR
SEEDS      = [0, 1, 2]
N_BOOT     = 2000


ARM_A = list(J.OWN_COLS)


FAMILIES = {
    "competitive-structure (COMP)": list(J.COMP_COLS),
    "cross-feature spillover (SPILL)": list(J.SPILL_COLS),
    "catalog dynamics (CAT)": list(J.CAT_COLS),
    "market motion (MKT)": list(J.MKT_COLS),
    "challenger trajectory (CHAL)": list(J.CHAL_COLS),
}
ARM_B = list(J.PRED_COLS)


MIN_TRAIN_ROWS = 30


def logistic():

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000, C=1.0))


def gbm(seed):

    return make_pipeline(SimpleImputer(strategy="median"),
                         GradientBoostingClassifier(random_state=seed))


def forward_folds(df, cols, make_est):


    X = df.select(cols).to_numpy()
    y = df["target"].to_numpy().astype(int)
    years = df["year"].to_numpy()
    out = {}
    for Y in np.unique(years):
        tr, te = years < Y, years == Y
        if tr.sum() < MIN_TRAIN_ROWS or y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
            continue
        if y[te].sum() == 0 or (1 - y[te]).sum() == 0:
            continue
        est = make_est()
        est.fit(X[tr], y[tr])
        out[int(Y)] = (y[te], est.predict_proba(X[te])[:, 1])
    return out


def fold_metrics(folds):

    rows = {}
    for Y, (yt, p) in folds.items():
        rows[Y] = dict(auc=roc_auc_score(yt, p), ap=average_precision_score(yt, p),
                       n_pos=int(yt.sum()), n_total=int(len(yt)))
    return rows


def boot_ci(vals, weights=None, n=N_BOOT, seed=0):

    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    idx = np.arange(len(vals))
    means = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if weights is None:
            means.append(vals[b].mean())
        else:
            w = np.asarray(weights, float)[b]
            means.append(np.average(vals[b], weights=w))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def calibration(folds, n_bins=10):

    yt = np.concatenate([f[0] for f in folds.values()])
    p = np.concatenate([f[1] for f in folds.values()])
    brier = brier_score_loss(yt, p)

    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    pred_c, obs_c, cnt = [], [], []
    for b in range(n_bins):
        m = which == b
        if m.sum() == 0:
            continue
        pred_c.append(p[m].mean()); obs_c.append(yt[m].mean()); cnt.append(int(m.sum()))

    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000, C=1e6)
    lr.fit(z, yt)
    slope = float(lr.coef_[0, 0]); intercept = float(lr.intercept_[0])
    return dict(brier=brier, pred=pred_c, obs=obs_c, cnt=cnt, slope=slope, intercept=intercept)


def main():
    print(f"window {Y0}-{Y1}  MIN_PER_YEAR={MIN_PER_YEAR}  MIN_YEARS={MIN_YEARS}  "
          f"HORIZON={HORIZON}  JUMP_FLOOR={JUMP_FLOOR}")
    print(f"Arm A ({len(ARM_A)}): {ARM_A}")
    print(f"Arm B ({len(ARM_B)}): A + 5 families")
    for k, v in FAMILIES.items():
        print(f"    {k}: {v}")

    df = J.build_panel()
    print(f"panel: {df.height} rows, {df['label'].n_unique()} features, "
          f"{int(df['target'].sum())} jumps (base {df['target'].mean():.3f})")


    fA = forward_folds(df, ARM_A, logistic)
    fB = forward_folds(df, ARM_B, logistic)
    common = sorted(set(fA) & set(fB))
    assert set(fA) == set(fB), "fold mask must be identical across arms"
    mA, mB = fold_metrics(fA), fold_metrics(fB)


    rows, deltas, ntot = [], [], []
    for Y in common:
        d = mB[Y]["auc"] - mA[Y]["auc"]
        deltas.append(d); ntot.append(mA[Y]["n_total"])
        for arm, m in (("A_frontier_only", mA[Y]), ("B_selection_enriched", mB[Y])):
            rows.append(dict(fold=Y, arm=arm, auc=round(m["auc"], 4), pr_auc=round(m["ap"], 4),
                             n_pos=m["n_pos"], n_total=m["n_total"], delta_B_minus_A=round(d, 4)))
    pl.DataFrame(rows).write_csv(os.path.join(HERE, "paired_null_summary.csv"))

    deltas = np.array(deltas); ntot = np.array(ntot)
    mean_d = float(deltas.mean())
    wmean_d = float(np.average(deltas, weights=ntot))
    n_pos_folds = int((deltas > 0).sum())


    ci_by_seed = {s: boot_ci(deltas, seed=s) for s in SEEDS}
    wci_by_seed = {s: boot_ci(deltas, weights=ntot, seed=s) for s in SEEDS}

    print(f"\n[logistic] folds={len(common)} ({common[0]}..{common[-1]})")
    print(f"  mean delta (B-A) = {mean_d:+.4f}   n_total-weighted = {wmean_d:+.4f}")
    print(f"  sign consistency: {n_pos_folds}/{len(common)} folds positive")
    for s in SEEDS:
        print(f"  seed {s}: 95% CI over folds = [{ci_by_seed[s][0]:+.4f}, {ci_by_seed[s][1]:+.4f}]  "
              f"(weighted [{wci_by_seed[s][0]:+.4f}, {wci_by_seed[s][1]:+.4f}])")


    print("\n[gbm seed sweep] mean delta (B-A) per seed:")
    gbm_means = []
    for s in SEEDS:
        gA = forward_folds(df, ARM_A, lambda s=s: gbm(s))
        gB = forward_folds(df, ARM_B, lambda s=s: gbm(s))
        gmA, gmB = fold_metrics(gA), fold_metrics(gB)
        gd = np.array([gmB[Y]["auc"] - gmA[Y]["auc"] for Y in sorted(set(gA) & set(gB))])
        gbm_means.append(float(gd.mean()))
        print(f"  seed {s}: mean delta = {gd.mean():+.4f}   {int((gd>0).sum())}/{len(gd)} folds positive")


    cA, cB = calibration(fA), calibration(fB)
    print(f"\n[calibration] frontier-only : Brier {cA['brier']:.4f}  slope {cA['slope']:.2f}  intercept {cA['intercept']:+.2f}")
    print(f"              selection-enr.: Brier {cB['brier']:.4f}  slope {cB['slope']:.2f}  intercept {cB['intercept']:+.2f}")


    make_delta_fig(common, deltas, mean_d, ci_by_seed[0])
    make_roc_fig(fA, fB)
    make_calib_fig(cA, cB)


    summ = dict(folds=common, mean_d=mean_d, wmean_d=wmean_d, ci=ci_by_seed[0], wci=wci_by_seed[0],
                ci_by_seed=ci_by_seed, n_pos_folds=n_pos_folds, deltas=deltas.round(4).tolist(),
                gbm_means=[round(x, 4) for x in gbm_means],
                aucA={Y: round(mA[Y]["auc"], 3) for Y in common},
                aucB={Y: round(mB[Y]["auc"], 3) for Y in common},
                calibA=cA, calibB=cB)
    import json
    with open(os.path.join(HERE, "_headline.json"), "w") as fh:
        json.dump({k: (v if not isinstance(v, dict) or k not in ("calibA", "calibB")
                       else {kk: vv for kk, vv in v.items() if kk != "cnt"})
                   for k, v in summ.items()}, fh, indent=2, default=str)
    print("\nwrote paired_null_summary.csv, images/*.png, _headline.json")


C_A = "#0072B2"
C_B = "#D55E00"
C_D = "#000000"


def make_delta_fig(folds, deltas, mean_d, ci):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=220)
    x = np.arange(len(folds))
    colors = [C_B if d > 0 else C_A for d in deltas]
    ax.bar(x, deltas, color=colors, width=0.68, zorder=3)
    ax.axhline(0, color="black", lw=2.2, zorder=4)

    ax.axhspan(ci[0], ci[1], color="0.75", alpha=0.45, zorder=1)
    ax.axhline(mean_d, color=C_D, lw=1.6, ls="--", zorder=5,
               label=f"mean delta {mean_d:+.3f}  (95% CI {ci[0]:+.3f}, {ci[1]:+.3f})")
    ax.set_xticks(x); ax.set_xticklabels([str(f) for f in folds], rotation=0)
    ax.set_xlabel("Forward-validation fold (test year)")
    ax.set_ylabel("AUC(selection-enriched) minus AUC(frontier-only)")
    ax.set_title("Paired per-fold delta in next-year frontier-jump AUC")
    lim = max(0.16, np.abs(deltas).max() * 1.15)
    ax.set_ylim(-lim, lim)
    ax.legend(loc="upper right", frameon=False)
    ax.text(0.012, 0.97, "above the line: enrichment helps",
            transform=ax.transAxes, va="top", ha="left", color=C_B, fontsize=9)
    ax.text(0.012, 0.03, "below the line: enrichment hurts",
            transform=ax.transAxes, va="bottom", ha="left", color=C_A, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "paired_delta.png"))
    plt.close(fig)


def make_roc_fig(fA, fB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=220, sharex=True, sharey=True)
    for ax, folds, col, title in ((axes[0], fA, C_A, "Frontier-only (Arm A)"),
                                  (axes[1], fB, C_B, "Selection-enriched (Arm B)")):
        yt_all, p_all = [], []
        for Y, (yt, p) in sorted(folds.items()):
            fpr, tpr, _ = roc_curve(yt, p)
            ax.plot(fpr, tpr, color=col, alpha=0.28, lw=1.0)
            yt_all.append(yt); p_all.append(p)
        yt_all = np.concatenate(yt_all); p_all = np.concatenate(p_all)
        fpr, tpr, _ = roc_curve(yt_all, p_all)
        pooled = roc_auc_score(yt_all, p_all)
        ax.plot(fpr, tpr, color=col, lw=3.0, label=f"pooled AUC {pooled:.3f}")
        ax.plot([0, 1], [0, 1], color="0.4", ls=":", lw=1.4)
        ax.set_title(title); ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("True positive rate")
    fig.suptitle("Forward-fold ROC, both arms (faint = per fold, bold = pooled)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "roc_by_fold.png"))
    plt.close(fig)


def make_calib_fig(cA, cB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=220)
    ax.plot([0, 1], [0, 1], color="0.4", ls=":", lw=1.5, label="perfect calibration")
    for c, col, lab in ((cA, C_A, "frontier-only"), (cB, C_B, "selection-enriched")):
        ax.plot(c["pred"], c["obs"], "o-", color=col, lw=1.8, ms=6,
                label=f"{lab}  (Brier {c['brier']:.3f})")
    ax.set_xlabel("Mean predicted jump probability")
    ax.set_ylabel("Observed jump frequency")
    ax.set_title("Calibration, pooled forward-fold predictions")
    ax.set_xlim(0, max(0.5, max(max(cA["pred"]), max(cB["pred"])) * 1.1))
    ax.set_ylim(0, ax.get_xlim()[1])
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "calibration.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
