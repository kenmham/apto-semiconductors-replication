"""Reproduce manuscript timing and product-stratification results."""
from pathlib import Path
import json
import numpy as np
import polars as pl
import interarrival as T
import timing_controls as C
import stratification as S
HERE = Path(__file__).resolve().parents[1]/'outputs'
OUT = HERE/'data/timing_stratification'
OUT.mkdir(parents=True, exist_ok=True)
panel = pl.read_parquet(HERE / 'data/retained_forecast_panel.parquet')
assert panel['label'].n_unique() == 204 and panel.height == 2449
assert panel['target'].sum() == 270
years = panel['year'].to_numpy().astype(int)
labels = panel['label'].to_numpy()
target = panel['target'].to_numpy().astype(int)
blocks = T.blocks_of(labels, years)
obs = T.stats(target, blocks, years)
support = {k: obs[k] for k in ('n_gaps', 'n_feat_gaps', 'n_rows_u', 'n_acf')}
(OUT / 'timing_support.json').write_text(json.dumps(support, indent=2))
rows = ['arm,stat,label,observed,null_median,null_lo95,null_hi95,p_two_sided,verdict']
rows += C.run('full', years, target, blocks)
rows += C.run('blockperm', years, target, blocks, (years - 2000) // 8)
m = years >= 2006
rows += C.run('post2006', years[m], target[m], T.blocks_of(labels[m], years[m]))
(OUT / 'timing_statistics.csv').write_text('\n'.join(rows) + '\n')
print('Timing complete', support, flush=True)
df, median = S.build_scored(panel)
df = df.sort(['label', 'year'])
df.write_parquet(OUT / 'h3_scored.parquet')
print('H3 scored', df.height, df['label'].n_unique(), int(df['target'].sum()), flush=True)
df, qrows, fdf, spear, band = S.primary_valley(df)
pl.DataFrame(qrows).write_csv(OUT / 'valley_quintiles.csv')
fdf.write_csv(OUT / 'per_feature.csv')
print('H3 primary', spear, flush=True)
all_rows, dropped, maxstat = S.secondary_strata(df)
pl.DataFrame(all_rows).write_csv(OUT / 'strata_results.csv')
pl.DataFrame(dropped).write_csv(OUT / 'strata_dropped.csv')
result = dict(capabilities=df['label'].n_unique(), scored_rows=df.height, events=int(df['target'].sum()), years=sorted(df['year'].unique().to_list()), eligible_features=fdf.height, volume_median=median, spearman=spear, maxstat=maxstat, quintile_null_band=band)
(OUT / 'h3_statistics.json').write_text(json.dumps(result, indent=2))
