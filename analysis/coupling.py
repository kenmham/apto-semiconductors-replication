"""Detrended cohort medians and lower-cohort changes during frontier stalls."""
import os
import numpy as np
import polars as pl
from specs import scan, frontier_series
SRC = os.environ['APTO_DATA']
MIN_PER_YEAR = 40
def detrend_residuals(yrs, xr, yr_):
    lx, ly = (np.log10(xr), np.log10(yr_))
    x = lx - np.polyval(np.polyfit(yrs, lx, 1), yrs)
    y = ly - np.polyval(np.polyfit(yrs, ly, 1), yrs)
    return (x, y)

def build(col, sub, direction, frac, y0, y1):
    hi = direction == 'hi'
    df = pl.scan_parquet(SRC).filter(pl.col('release_year').is_between(y0, y1)).filter(pl.col('subcategory') == sub).select(['release_year', col]).drop_nulls().filter(pl.col(col) > 0).collect()
    rows = []
    for yr, g in df.group_by('release_year', maintain_order=False):
        yr = yr[0]
        v = g[col].to_numpy()
        if v.size < MIN_PER_YEAR:
            continue
        q = np.quantile(v, 1 - frac if hi else frac)
        front = v[v >= q] if hi else v[v <= q]
        rest = v[v < q] if hi else v[v > q]
        if front.size == 0 or rest.size == 0:
            continue
        rows.append((int(yr), float(np.median(rest)), float(np.median(front)), v.size))
    rows.sort()
    return rows

def feature_changes(col, sub, direction, frac, y0, y1, lf):
    rows = build(col, sub, direction, frac, y0, y1)
    fs = frontier_series(lf, col, sub, direction)
    if not rows or fs is None:
        return []
    sign = 1.0 if direction == 'hi' else -1.0
    typ = {int(r[0]): r[1] for r in rows}
    cB = {int(y): b for y, b in zip(fs['yr'].tolist(), fs['cB'].tolist())}
    yrs = sorted(set(typ) & set(cB))
    out = []
    for a, b in zip(yrs[:-1], yrs[1:]):
        advanced = sign * (np.log10(cB[b]) - np.log10(cB[a])) > 1e-09
        d_typ = sign * (np.log10(typ[b]) - np.log10(typ[a]))
        out.append((advanced, float(d_typ)))
    return out
