import os
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from specs import scan, frontier_series, step_gains, norm, MIN_PER_YEAR

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(os.path.dirname(HERE), "data")

HORIZON    = int(os.environ.get("WK3_HORIZON", "3"))
JUMP_FLOOR = float(os.environ.get("WK3_JUMP_FLOOR", "0.10"))
N_BOOT     = 500
SEED       = 0


OWN_COLS = ["recent_change", "praw_change", "yrs_since_jump", "maturity"]

COMP_COLS = ["leader_margin", "dispersion", "eff_mfrs", "leader_held"]

SPILL_COLS = ["sib_jumped_last", "sib_jumps_recent"]

CAT_COLS = ["cat_inflow", "cat_growth"]

WIN = 3
MKT_A = ["new_mfr_rate_3y", "top_churn_3y", "n_leaders_3y"]
MKT_B = ["gap_trend_3y", "runnerup_slope_3y", "near_frontier_trend"]
MKT_C = ["median_slope_3y", "pack_slope_3y", "inflow_accel"]
MKT_COLS = MKT_A + MKT_B + MKT_C

CHAL_COLS = ["challenger_gap", "challenger_climb_3y", "challenger_gapclose_3y"]
PRED_COLS = OWN_COLS + COMP_COLS + SPILL_COLS + CAT_COLS + MKT_COLS + CHAL_COLS

MODELS = {
    "logistic":  make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(max_iter=1000, C=1.0)),
    "gbm":       make_pipeline(SimpleImputer(strategy="median"),
                               GradientBoostingClassifier(random_state=SEED)),
}


def load_selected():
    d = pl.read_csv(os.path.join(CSV, "feature_selection.csv")).filter(pl.col("selected"))
    return [(r["column"], r["subcategory"], f'{r["column"]}|{r["subcategory"]}', r["direction"])
            for r in d.iter_rows(named=True)]


def market_series(lf, col, sub, hi):


    q_pack, q_edge = (0.90, 0.95) if hi else (0.10, 0.05)
    d = 1 if hi else -1
    df = (lf.filter(pl.col("subcategory") == sub)
            .select(["release_year", col, pl.coalesce(["manufacturer", "mfr"]).alias("mfr")])
            .drop_nulls(subset=[col]).filter(pl.col(col) > 0).collect())
    if df.height == 0:
        return {"comp": {}, "by_year": {}, "chal": {}}
    near = ((pl.col(col) >= pl.col(col).max() / 1.25) if hi
            else (pl.col(col) <= pl.col(col).min() * 1.25)).mean()
    g = (df.group_by("release_year").agg(
            n=pl.len(),
            best=(pl.col(col).max() if hi else pl.col(col).min()),
            pack=pl.col(col).quantile(q_pack),
            edge=pl.col(col).quantile(q_edge),
            p50=pl.col(col).quantile(0.5),
            near=near.alias("near"),

            leader=pl.col("mfr").sort_by([col, "mfr"], descending=[hi, False]).first())
           .filter(pl.col("n") >= MIN_PER_YEAR).sort("release_year"))
    if g.height == 0:
        return {"comp": {}, "by_year": {}, "chal": {}}

    thr = g.select(["release_year", "pack"])
    top = df.join(thr, on="release_year").filter(
        (pl.col(col) >= pl.col("pack")) if hi else (pl.col(col) <= pl.col("pack")))
    tops = top.group_by("release_year").agg(makers=pl.col("mfr").unique())
    eff = (top.group_by(["release_year", "mfr"]).agg(c=pl.len())
              .group_by("release_year")
              .agg(eff=(pl.col("c").sum() ** 2) / (pl.col("c") ** 2).sum()))
    g = g.join(eff, on="release_year", how="left").join(tops, on="release_year", how="left").sort("release_year")

    yr = g["release_year"].to_numpy()
    best = g["best"].to_numpy(); pack = g["pack"].to_numpy(); edge = g["edge"].to_numpy()
    p50 = g["p50"].to_numpy(); nearv = g["near"].to_numpy(); nn = g["n"].to_numpy()
    leaders = [norm(x) for x in g["leader"].to_list()]
    makers = [set(norm(m) for m in (ms or [])) for ms in g["makers"].to_list()]
    effv = g["eff"].to_numpy()
    held = np.ones(len(yr))
    for k in range(1, len(yr)):
        held[k] = held[k - 1] + 1 if leaders[k] == leaders[k - 1] else 1

    comp, by_year = {}, {}
    for k, y in enumerate(yr):
        y = int(y)
        margin = float(np.log10(best[k] / pack[k]) * d) if pack[k] > 0 else np.nan
        disp = float(np.log10(edge[k] / p50[k]) * d) if p50[k] > 0 else np.nan
        comp[y] = (margin, disp, float(effv[k]) if effv[k] is not None else np.nan, float(held[k]))
        by_year[y] = dict(
            disp=disp, near=float(nearv[k]),
            logedge=float(np.log10(edge[k]) * d), logp50=float(np.log10(p50[k]) * d),
            logpack=float(np.log10(pack[k]) * d), logn=float(np.log10(nn[k])),
            leader=leaders[k], makers=makers[k])


    gated = set(int(y) for y in yr)
    mb = (df.filter(pl.col("mfr").is_not_null())
            .group_by(["release_year", "mfr"])
            .agg(v=(pl.col(col).max() if hi else pl.col(col).min())))
    mbest = {}
    lead_v = {}
    runnerup = {}
    for y, m, v in zip(mb["release_year"].to_list(), mb["mfr"].to_list(), mb["v"].to_list()):
        y = int(y)
        if y not in gated or v is None or v <= 0:
            continue
        mm = norm(m)
        cur = mbest.setdefault(y, {})
        cur[mm] = (max(cur[mm], v) if hi else min(cur[mm], v)) if mm in cur else v
    for y, mkrs in mbest.items():


        ranked = sorted(mkrs.items(), key=lambda kv: ((-kv[1] if hi else kv[1]), kv[0]))
        lead_v[y] = ranked[0][1]
        if len(ranked) >= 2:
            cname, cval = ranked[1]
            runnerup[y] = (cname, float(np.log10(lead_v[y] / cval) * d))
    chal = {}
    for y in mbest:
        if y not in runnerup:
            continue
        C, gap = runnerup[y]
        yb = y - WIN
        climb = gapclose = np.nan
        if yb in mbest and C in mbest[yb] and yb in lead_v:
            climb = float(np.log10(mbest[y][C] / mbest[yb][C]) * d)
            gap_then = float(np.log10(lead_v[yb] / mbest[yb][C]) * d)
            gapclose = gap_then - gap
        chal[y] = (gap, climb, gapclose)
    return {"comp": comp, "by_year": by_year, "chal": chal}


def market_dynamics(by_year, Y):

    nan9 = (np.nan,) * 9
    b = by_year.get(Y)
    if b is None:
        return nan9
    bw = by_year.get(Y - WIN)
    b2w = by_year.get(Y - 2 * WIN)

    recent_yrs = [y for y in range(Y - WIN + 1, Y + 1) if y in by_year]
    prior_yrs = [y for y in by_year if y <= Y - WIN]
    prev_yrs = [y for y in range(Y - 2 * WIN + 1, Y - WIN + 1) if y in by_year]
    recent_top = set().union(*[by_year[y]["makers"] for y in recent_yrs]) if recent_yrs else set()
    prior_top = set().union(*[by_year[y]["makers"] for y in prior_yrs]) if prior_yrs else set()
    prev_top = set().union(*[by_year[y]["makers"] for y in prev_yrs]) if prev_yrs else set()
    new_mfr = float(len(recent_top - prior_top)) if prior_yrs else np.nan
    if recent_top and prev_top:
        churn = 1.0 - len(recent_top & prev_top) / len(recent_top | prev_top)
    else:
        churn = np.nan
    n_leaders = float(len({by_year[y]["leader"] for y in recent_yrs})) if recent_yrs else np.nan

    gap_trend = (b["disp"] - bw["disp"]) if bw else np.nan
    runnerup = (b["logedge"] - bw["logedge"]) if bw else np.nan
    near_trend = (b["near"] - bw["near"]) if bw else np.nan

    med_slope = (b["logp50"] - bw["logp50"]) if bw else np.nan
    pack_slope = (b["logpack"] - bw["logpack"]) if bw else np.nan
    inflow_accel = ((b["logn"] - bw["logn"]) - (bw["logn"] - b2w["logn"])) if (bw and b2w) else np.nan
    return (new_mfr, churn, n_leaders, gap_trend, runnerup, near_trend,
            med_slope, pack_slope, inflow_accel)


def build_panel():


    lf = scan()
    feats = []
    sub_year_jumps = {}
    for col, sub, label, direction in load_selected():
        s = frontier_series(lf, col, sub, direction)
        if s is None:
            continue
        step_years, steps, total = step_gains(s)
        if total <= 0:
            continue
        d = 1 if s["hi"] else -1
        yr = s["yr"]
        prog = np.log10(s["cB"]) * d
        lpraw = np.log10(s["praw"]) * d

        jmap = {int(step_years[k]): int(steps[k] > JUMP_FLOOR) for k in range(len(steps))}
        for y, j in jmap.items():
            sub_year_jumps[(sub, y)] = sub_year_jumps.get((sub, y), 0) + j
        mkt = market_series(lf, col, sub, s["hi"])
        feats.append(dict(col=col, sub=sub, label=label, hi=s["hi"], yr=yr, prog=prog,
                          lpraw=lpraw, steps=steps, jmap=jmap, n=s["n"],
                          comp=mkt["comp"], by_year=mkt["by_year"], chal=mkt["chal"]))

    rows = []
    for f in feats:
        sub, yr, prog, lpraw, steps = f["sub"], f["yr"], f["prog"], f["lpraw"], f["steps"]
        jmap, comp, label, n = f["jmap"], f["comp"], f["label"], f["n"]
        by_year, chal = f["by_year"], f["chal"]
        M = len(yr)
        jump_years = sorted(y for y, j in jmap.items() if j)
        for i in range(HORIZON + 1, M):
            step_i = float(steps[i - 1])
            ycand = int(yr[i])
            target = 1 if step_i > JUMP_FLOOR else 0
            recent_change = float(prog[i - 1] - prog[i - 1 - HORIZON])
            praw_change = float(lpraw[i - 1] - lpraw[i - 1 - HORIZON])
            maturity = float(prog[i - 1])
            prev = [jy for jy in jump_years if jy < ycand]
            yrs_since = float(ycand - prev[-1]) if prev else np.nan
            margin, disp, eff_m, held = comp.get(int(yr[i - 1]), (np.nan,) * 4)

            yprev = ycand - 1
            sib_last = sub_year_jumps.get((sub, yprev), 0) - jmap.get(yprev, 0)
            sib_recent = sum(sub_year_jumps.get((sub, yy), 0) - jmap.get(yy, 0)
                             for yy in range(ycand - HORIZON, ycand))

            cat_inflow = float(np.log10(n[i - 1])) if n[i - 1] > 0 else np.nan
            cat_growth = (float(np.log10(n[i - 1] / n[i - 2]))
                          if n[i - 1] > 0 and n[i - 2] > 0 else np.nan)

            mkt = market_dynamics(by_year, ycand - 1)
            chl = chal.get(ycand - 1, (np.nan,) * 3)
            rows.append((label, sub, ycand, target,
                         recent_change, praw_change, yrs_since, maturity,
                         margin, disp, eff_m, held,
                         1.0 if sib_last > 0 else 0.0, float(sib_recent),
                         cat_inflow, cat_growth, *mkt, *chl))
    cols = ["label", "subcategory", "year", "target",
            "recent_change", "praw_change", "yrs_since_jump", "maturity",
            "leader_margin", "dispersion", "eff_mfrs", "leader_held",
            "sib_jumped_last", "sib_jumps_recent", "cat_inflow", "cat_growth"] + MKT_COLS + CHAL_COLS
    return pl.DataFrame(rows, schema=cols, orient="row")


FEATURE_SETS = {"own": OWN_COLS,
                "challenger": CHAL_COLS, "own+challenger": OWN_COLS + CHAL_COLS,
                "market": MKT_COLS, "own+market": OWN_COLS + MKT_COLS, "all": PRED_COLS}
GBM_SETS = {"own+challenger", "own+market", "all"}


def oof_predictions(df, est, scheme, cols):

    X = df.select(cols).to_numpy()
    y = df["target"].to_numpy().astype(int)
    oof = np.full(len(y), np.nan)

    if scheme == "lofo":
        labels = df["label"].to_numpy()
        for held in np.unique(labels):
            tr, te = labels != held, labels == held
            if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
                continue
            est.fit(X[tr], y[tr])
            oof[te] = est.predict_proba(X[te])[:, 1]
    elif scheme == "forward":
        years = df["year"].to_numpy()
        for Y in np.unique(years):
            tr, te = years < Y, years == Y
            if tr.sum() < 30 or y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
                continue
            est.fit(X[tr], y[tr])
            oof[te] = est.predict_proba(X[te])[:, 1]
    return oof, y


def boot_auc_ci(y, p, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0 or (1 - y[b]).sum() == 0:
            continue
        vals.append(roc_auc_score(y[b], p[b]))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    os.makedirs(CSV, exist_ok=True)
    df = build_panel()
    n, npos = df.height, int(df["target"].sum())
    print(f"panel: {n} rows, {df['label'].n_unique()} features, "
          f"{npos} jumps (base rate {npos/n:.3f}), "
          f"HORIZON={HORIZON}, JUMP_FLOOR={JUMP_FLOOR}")

    results = []
    for scheme in ("lofo", "forward"):
        print(f"\n[{scheme}]")
        for fs_name, cols in FEATURE_SETS.items():
            for name, est in MODELS.items():

                if name == "gbm" and fs_name not in GBM_SETS:
                    continue
                oof, y = oof_predictions(df, est, scheme, cols)
                m = ~np.isnan(oof)
                if m.sum() == 0 or y[m].sum() == 0:
                    print(f"  {fs_name:4s} {name:9s}  (no scorable folds)")
                    continue
                yy, pp = y[m], oof[m]
                auc = roc_auc_score(yy, pp)
                lo, hi = boot_auc_ci(yy, pp)
                ap = average_precision_score(yy, pp)
                brier = brier_score_loss(yy, pp)
                base = float(yy.mean())
                results.append(dict(feature_set=fs_name, model=name, scheme=scheme,
                                    n_scored=int(m.sum()), base_rate=round(base, 4),
                                    auc=round(auc, 4), auc_lo=round(lo, 4), auc_hi=round(hi, 4),
                                    ap=round(ap, 4), ap_lift=round(ap / base, 2),
                                    brier=round(brier, 4)))
                print(f"  {fs_name:4s} {name:9s}  AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  "
                      f"AP {ap:.3f} (x{ap/base:.2f} base)  Brier {brier:.3f}")

    pl.DataFrame(results).write_csv(os.path.join(CSV, "jump_cv_results.csv"))
    print(f"\nwrote {os.path.join(CSV, 'jump_cv_results.csv')}")
    print("AUC 0.5 and AP≈base_rate = no signal. Beat these with Move-2 predictors.")


if __name__ == "__main__":
    main()
