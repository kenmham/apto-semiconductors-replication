from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import sys

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parents[1] / 'outputs'
ROOT = HERE.parent
ANALYSIS = ROOT / 'analysis'
SUPP = ANALYSIS
import specs
from episode_length import episodes_for_feature
from second_year_share import jumps_for_feature
from smoothing import smooth
from frontier_lead import median_line

OUT = HERE / 'data'
OUT.mkdir(exist_ok=True)
specs.SRC = __import__('os').environ['APTO_DATA']
selection_path = ROOT / 'data/feature_selection.csv'
selection = pl.read_csv(selection_path)
selected = selection.filter(pl.col('selected'))
lf = specs.scan()

catalog = pl.scan_parquet(specs.SRC)
summary = {
    'catalog': catalog.select(pl.len().alias('records'),
        pl.col('subcategory').drop_nulls().n_unique().alias('subcategories'),
        pl.col('release_year').null_count().alias('missing_release_year'),
        pl.col('release_year').is_between(2000, 2024).sum().alias('records_2000_2024')).collect().to_dicts()[0],
    'selection': {'candidates': selection.height,
        'coverage_pass': int(selection['pass_coverage'].sum()),
        'change_pass': int(selection['has_change'].sum()),
        'both_pass': selection.filter(pl.col('pass_coverage') & pl.col('has_change')).height,
        'selected': selected.height, 'subcategories': selected['subcategory'].n_unique(),
        'attribute_columns': selected['column'].n_unique(),
        'higher': selected.filter(pl.col('direction') == 'hi').height,
        'lower': selected.filter(pl.col('direction') == 'lo').height},
}


near, episodes, windows = [], [], []
for col, sub, label, direction in specs.SPECS:
    s = specs.frontier_series(lf, col, sub, direction)
    if s is None:
        continue
    years, gains, _ = specs.step_gains(s)
    episodes.extend(dict(feature=label, **e) for e in episodes_for_feature(years, gains, .10))
    for year, g1, g2, g3 in jumps_for_feature(years, gains, .10):
        pos = list(s['yr']).index(year)
        windows.append(dict(feature=label, year=year, initial_share=g1/(g1+g2+g3),
            initial_interval_years=int(year-s['yr'][pos-1])))
    p = (lf.filter(pl.col('subcategory') == sub).select('release_year', col)
         .drop_nulls().filter(pl.col(col) > 0).collect())
    total, nnear = 0, 0
    for y, frontier in zip(s['yr'], s['cB']):
        v = p.filter(pl.col('release_year') == y)[col].to_numpy()
        ratio = v / frontier if direction == 'hi' else frontier / v
        total += len(v)
        nnear += int(np.sum(ratio >= .75))
    near.append(dict(feature=label, n=total, near_share=nnear/total))
summary['anchors'] = {'eligible': len(near), 'median_near_percent': float(100*np.median([r['near_share'] for r in near])),
    'episodes': len(episodes), 'singletons': sum(r['length'] == 1 for r in episodes),
    'windows': len(windows), 'mean_initial_share': float(np.mean([r['initial_share'] for r in windows])),
    'windows_with_initial_gap': sum(r['initial_interval_years'] > 1 for r in windows)}
pl.DataFrame(near).write_csv(OUT / 'anchor_near.csv')
pl.DataFrame(episodes).write_csv(OUT / 'anchor_episodes.csv')
pl.DataFrame(windows).write_csv(OUT / 'anchor_windows.csv')


lead_rows, excluded = [], []
for row in selected.iter_rows(named=True):
    col, sub, direction = row['column'], row['subcategory'], row['direction']
    s = specs.frontier_series(lf, col, sub, direction)
    if s is None:
        excluded.append(dict(column=col, subcategory=sub, reason='fewer than eight usable years'))
        continue
    median = median_line(lf, col, sub, s['yr'])
    raw = (lf.filter(pl.col('subcategory') == sub).select('release_year', col)
        .drop_nulls().filter(pl.col(col) > 0).group_by('release_year')
        .agg(pl.col(col).median().alias('median'), pl.len().alias('n'))
        .filter(pl.col('n') >= 40).sort('release_year').collect())
    assert raw['release_year'].to_list() == list(s['yr'])
    rawmed = raw['median'].to_numpy()
    d = 1 if direction == 'hi' else -1
    lead = (s['cA'] / median) ** d
    lead_raw = (s['cA'] / rawmed) ** d
    gain = float(d*np.log10(s['cA'][-1]/s['cA'][0]))
    lead_rows.append(dict(column=col, subcategory=sub, direction=direction,
        start_year=int(s['yr'][0]), end_year=int(s['yr'][-1]), usable_years=len(s['yr']),
        listing_observations=s['N'], p95_total_log_gain=gain,
        gap_start=float(lead[0]), gap_end=float(lead[-1]), widening=float(lead[-1]/lead[0]),
        raw_gap_start=float(lead_raw[0]), raw_gap_end=float(lead_raw[-1]),
        raw_widening=float(lead_raw[-1]/lead_raw[0])))
leads = pl.DataFrame(lead_rows).sort('gap_end', descending=True)
leads.write_csv(OUT / 'all_capability_leads.csv')
summary['lead_analysis'] = {'eligible': leads.height, 'excluded': excluded,
    'zero_p95_growth': int((leads['p95_total_log_gain'] == 0).sum()),
    'widening': int((leads['widening'] > 1+1e-10).sum()),
    'narrowing': int((leads['widening'] < 1-1e-10).sum()),
    'median_start': float(leads['gap_start'].median()), 'median_end': float(leads['gap_end'].median()),
    'median_widening': float(leads['widening'].median()),
    'smoothed_endpoint_below_one': leads.filter((pl.col('gap_start') < 1) | (pl.col('gap_end') < 1)).height,
    'raw_widening': int((leads['raw_widening'] > 1+1e-10).sum()),
    'raw_median_widening': float(leads['raw_widening'].median())}


valley = pl.read_csv(ROOT / 'data/shoulder_fits.csv')
summary['saved_shoulders'] = {'rows':valley.height,
    'depth_ge_004': int((valley['depth'] >= .04).sum()),
    'lower_ge_004': int((valley['depth_lo'] >= .04).sum()),
    'clears':int(valley['clears'].sum())}
(OUT / 'audit_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'sources'}, indent=2))
