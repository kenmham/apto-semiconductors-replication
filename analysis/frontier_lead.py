import numpy as np
import polars as pl
from specs import MIN_PER_YEAR
from smoothing import smooth

def median_line(lf, col, sub, yr):

    g = (lf.filter(pl.col("subcategory") == sub).select(["release_year", col])
           .drop_nulls().filter(pl.col(col) > 0)
           .group_by("release_year")
           .agg(pl.col(col).median().alias("med"), pl.len().alias("n"))
           .filter(pl.col("n") >= MIN_PER_YEAR).sort("release_year").collect())
    d = dict(zip(g["release_year"].to_list(), g["med"].to_list()))
    med = np.array([d.get(int(y), np.nan) for y in yr], dtype=float)
    if np.isnan(med).any():
        idx = np.arange(len(med)); good = ~np.isnan(med)
        med[~good] = np.interp(idx[~good], idx[good], med[good])
    return 10.0 ** smooth(np.log10(med))
