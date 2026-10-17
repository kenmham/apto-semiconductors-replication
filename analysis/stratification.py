"""Forward predictions regrouped by shoulder depth and product strata."""
from pathlib import Path
import os
import numpy as np
import polars as pl
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import jump_cv as J
ARM_A = list(J.OWN_COLS)
ARM_B = list(J.PRED_COLS)
VALLEY_CSV = Path(__file__).resolve().parents[1]/'data/shoulder_fits.csv'
N_QUINTILES = 5
MIN_CELL_N = 40
MIN_CELL_POS = 5
MIN_FEAT_POS = 1
MIN_FEAT_NEG = 1
N_BOOT = 2000
N_PERM_FEATURE = 5000
N_PERM_MAX = 2000
BOOT_SEED = 0
PERM_SEED = 0
CUTS = ['category', 'subcategory', 'volume']

def roc_auc_score(y, p):
    """Rank-sum AUC, including half credit for ties."""
    y = np.asarray(y, dtype=bool)
    n1 = int(y.sum()); n0 = len(y)-n1
    return float((rankdata(p)[y].sum()-n1*(n1+1)/2)/(n1*n0))

def logistic():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))

def auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, p)

def build_scored(panel):
    df = panel.clone()
    pA, y = J.oof_predictions(df, logistic(), 'forward', ARM_A)
    pB, _ = J.oof_predictions(df, logistic(), 'forward', ARM_B)
    m = ~np.isnan(pA)
    assert np.array_equal(m, ~np.isnan(pB)), 'arms must score the identical row set'
    df = df.with_columns([pl.Series('pA', pA), pl.Series('pB', pB), pl.Series('scored', m)]).filter(pl.col('scored'))
    df = df.with_columns(pl.col('label').str.split('|').list.first().alias('column'))
    vd = pl.read_csv(VALLEY_CSV).select(['column', 'subcategory', 'depth', 'n_parts'])
    df = df.join(vd, on=['column', 'subcategory'], how='left')
    assert df['depth'].null_count() == 0, 'every scored feature must have a valley depth'
    parts = pl.scan_parquet(os.environ['APTO_DATA']).select(['subcategory', 'category']).drop_nulls().group_by(['subcategory', 'category']).len().collect()
    cmap = {}
    for sub, grp in parts.sort(['subcategory', 'len'], descending=[False, True]).group_by('subcategory', maintain_order=True):
        cmap[sub[0]] = grp['category'][0]
    df = df.with_columns(pl.col('subcategory').replace_strict(cmap, default='Unknown').alias('category'))
    feat_parts = df.group_by('label').agg(pl.col('n_parts').first())
    med = float(feat_parts['n_parts'].median())
    df = df.with_columns(pl.when(pl.col('n_parts') >= med).then(pl.lit('high-volume (consumer-like)')).otherwise(pl.lit('low-volume (industrial-like)')).alias('volume'))
    return (df, med)

def boot_group(y, pA, pB, seed=BOOT_SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pA = np.asarray(pA)
    pB = np.asarray(pB)
    idx = np.arange(len(y))
    aA, aB, dd = ([], [], [])
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        vy = y[b]
        if vy.sum() == 0 or vy.sum() == len(vy):
            continue
        ra, rb = (roc_auc_score(vy, pA[b]), roc_auc_score(vy, pB[b]))
        aA.append(ra)
        aB.append(rb)
        dd.append(rb - ra)
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a else (np.nan, np.nan)
    return (q(aA), q(aB), q(dd))

def group_row(name, sub, extra=None):
    y = sub['target'].to_numpy()
    pA = sub['pA'].to_numpy()
    pB = sub['pB'].to_numpy()
    aA, aB = (auc(y, pA), auc(y, pB))
    ciA, ciB, ciD = boot_group(y, pA, pB)
    row = dict(stratum=name, n=int(len(y)), npos=int(y.sum()), aucA=round(aA, 4) if np.isfinite(aA) else None, aucB=round(aB, 4) if np.isfinite(aB) else None, delta=round(aB - aA, 4) if np.isfinite(aA) and np.isfinite(aB) else None, aucA_lo=round(ciA[0], 4), aucA_hi=round(ciA[1], 4), aucB_lo=round(ciB[0], 4), aucB_hi=round(ciB[1], 4), delta_lo=round(ciD[0], 4), delta_hi=round(ciD[1], 4))
    if extra:
        row.update(extra)
    return row

def quintile_bins(depth_by_feat):
    labels = sorted(depth_by_feat.keys())
    depths = np.array([depth_by_feat[l] for l in labels])
    order = np.argsort(depths, kind='stable')
    q = np.empty(len(labels), dtype=int)
    edges = np.floor(np.arange(len(labels)) * N_QUINTILES / len(labels)).astype(int)
    q[order] = edges
    return {labels[i]: int(q[i]) for i in range(len(labels))}

def primary_valley(df):
    depth_by_feat = {r['label']: r['depth'] for r in df.group_by('label', maintain_order=True).agg(pl.col('depth').first()).iter_rows(named=True)}
    qmap = quintile_bins(depth_by_feat)
    df = df.with_columns(pl.col('label').replace_strict(qmap).alias('quintile'))
    qrows = []
    for qi in range(N_QUINTILES):
        sub = df.filter(pl.col('quintile') == qi)
        drange = sub.group_by('label').agg(pl.col('depth').first())['n_parts' if False else 'depth']
        row = group_row(f'Q{qi + 1}', sub, extra=dict(quintile=qi + 1, depth_lo=round(float(drange.min()), 5), depth_hi=round(float(drange.max()), 5), n_features=sub['label'].n_unique()))
        qrows.append(row)
    frows = []
    for lab, grp in df.group_by('label', maintain_order=True):
        y = grp['target'].to_numpy()
        if y.sum() < MIN_FEAT_POS or len(y) - y.sum() < MIN_FEAT_NEG:
            continue
        aA = auc(y, grp['pA'].to_numpy())
        aB = auc(y, grp['pB'].to_numpy())
        frows.append(dict(label=lab[0], subcategory=grp['subcategory'][0], depth=float(grp['depth'][0]), n=int(len(y)), npos=int(y.sum()), aucA=round(aA, 4), aucB=round(aB, 4), delta=round(aB - aA, 4)))
    fdf = pl.DataFrame(frows)
    depth = fdf['depth'].to_numpy()
    rng = np.random.default_rng(PERM_SEED)
    spear = {}
    for stat in ('aucA', 'aucB', 'delta'):
        v = fdf[stat].to_numpy()
        rho = spearmanr(depth, v).correlation
        perm = np.empty(N_PERM_FEATURE)
        for i in range(N_PERM_FEATURE):
            perm[i] = spearmanr(rng.permutation(depth), v).correlation
        p_two = float((np.abs(perm) >= abs(rho)).mean())
        p_greater = float((perm >= rho).mean())
        spear[stat] = dict(rho=round(float(rho), 4), p_two_sided=round(p_two, 4), p_one_sided_greater=round(p_greater, 4), null_lo=round(float(np.percentile(perm, 2.5)), 4), null_hi=round(float(np.percentile(perm, 97.5)), 4))
    rng2 = np.random.default_rng(PERM_SEED + 1)
    labels = sorted(depth_by_feat.keys())
    depvals = np.array([depth_by_feat[l] for l in labels])
    lab_arr = df['label'].to_numpy()
    y_arr = df['target'].to_numpy()
    pA_arr = df['pA'].to_numpy()
    idx_by_lab = {l: np.where(lab_arr == l)[0] for l in labels}
    null_band = {qi: [] for qi in range(N_QUINTILES)}
    for _ in range(N_PERM_MAX):
        shuf = rng2.permutation(depvals)
        dbf = {labels[i]: shuf[i] for i in range(len(labels))}
        qm = quintile_bins(dbf)
        for qi in range(N_QUINTILES):
            sel = np.concatenate([idx_by_lab[l] for l in labels if qm[l] == qi])
            a = auc(y_arr[sel], pA_arr[sel])
            if np.isfinite(a):
                null_band[qi].append(a)
    band = {qi: (float(np.percentile(null_band[qi], 2.5)), float(np.percentile(null_band[qi], 97.5)), float(np.median(null_band[qi]))) for qi in range(N_QUINTILES)}
    return (df, qrows, fdf, spear, band)

def secondary_strata(df):
    all_rows, dropped, maxstat = ([], [], {})
    for cut in CUTS:
        cells = []
        for name, grp in df.group_by(cut, maintain_order=True):
            nm = name[0]
            n, npos = (grp.height, int(grp['target'].sum()))
            keep = n >= MIN_CELL_N and npos >= MIN_CELL_POS and (0 < npos < n)
            row = group_row(nm, grp, extra=dict(cut=cut, kept=keep))
            all_rows.append(row)
            if keep:
                cells.append((nm, grp))
            else:
                reason = []
                if n < MIN_CELL_N:
                    reason.append(f'n={n}<{MIN_CELL_N}')
                if npos < MIN_CELL_POS:
                    reason.append(f'npos={npos}<{MIN_CELL_POS}')
                if not 0 < npos < n:
                    reason.append('single-class')
                dropped.append(dict(cut=cut, stratum=nm, n=n, npos=npos, reason=';'.join(reason)))
        if len(cells) >= 1:
            sizes = [g.height for _, g in cells]
            pool = pl.concat([g for _, g in cells])
            y = pool['target'].to_numpy()
            pA = pool['pA'].to_numpy()
            pB = pool['pB'].to_numpy()
            obs_aucA = [auc(g['target'].to_numpy(), g['pA'].to_numpy()) for _, g in cells]
            obs_delta = [auc(g['target'].to_numpy(), g['pB'].to_numpy()) - auc(g['target'].to_numpy(), g['pA'].to_numpy()) for _, g in cells]
            obs_max_auc = float(np.nanmax(obs_aucA))
            obs_max_delta = float(np.nanmax(obs_delta))
            rng = np.random.default_rng(PERM_SEED + 7)
            null_auc, null_delta = ([], [])
            for _ in range(N_PERM_MAX):
                perm = rng.permutation(len(y))
                start, ma, md = (0, [], [])
                for sz in sizes:
                    sel = perm[start:start + sz]
                    start += sz
                    a = auc(y[sel], pA[sel])
                    b = auc(y[sel], pB[sel])
                    if np.isfinite(a):
                        ma.append(a)
                        if np.isfinite(b):
                            md.append(b - a)
                if ma:
                    null_auc.append(max(ma))
                if md:
                    null_delta.append(max(md))
            maxstat[cut] = dict(n_strata=len(cells), obs_max_auc=round(obs_max_auc, 4), obs_max_auc_stratum=cells[int(np.nanargmax(obs_aucA))][0], p_max_auc=round(float((np.asarray(null_auc) >= obs_max_auc).mean()), 4), null_auc_95=round(float(np.percentile(null_auc, 95)), 4), obs_max_delta=round(obs_max_delta, 4), obs_max_delta_stratum=cells[int(np.nanargmax(obs_delta))][0], p_max_delta=round(float((np.asarray(null_delta) >= obs_max_delta).mean()), 4), null_delta_95=round(float(np.percentile(null_delta, 95)), 4))
        else:
            maxstat[cut] = dict(n_strata=0)
    return (all_rows, dropped, maxstat)
