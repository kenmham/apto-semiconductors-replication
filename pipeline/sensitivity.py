"""Recompute the original, retained, and additional current-field exclusions."""
from pathlib import Path
import importlib.util, json, warnings
import numpy as np
import polars as pl
HERE = Path(__file__).resolve().parents[1] / 'outputs'
ROOT = HERE.parent
source = ROOT / 'analysis/paired_null.py'
spec = importlib.util.spec_from_file_location('paired', source)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
selected = P.J.load_selected()
rf = 'currentRatingAmps|RF FETs, MOSFETs'
ldo = 'operatingSupplyCurrent|LDO Voltage Regulators'
pmic = 'currentSupply|Power Management - Specialized'
scenarios = {'original_206': [], 'retained_204': [rf, ldo], 'retained_203': [rf, ldo, pmic]}
leads = pl.read_csv(HERE / 'data/all_capability_leads.csv').with_columns((pl.col('column') + '|' + pl.col('subcategory')).alias('key'))
results = []
foldrows = []
for name, excluded in scenarios.items():
    assert set(excluded) <= set((r[2] for r in selected))
    P.J.load_selected = lambda: [r for r in selected if r[2] not in excluded]
    panel = P.J.build_panel()
    assert not set(excluded) & set(panel['label'].unique().to_list())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a = P.fold_metrics(P.forward_folds(panel, P.ARM_A, P.logistic))
        b = P.fold_metrics(P.forward_folds(panel, P.ARM_B, P.logistic))
    assert set(a) == set(b)
    years = sorted(a)
    d = np.array([b[y]['auc'] - a[y]['auc'] for y in years])
    w = [a[y]['n_total'] for y in years]
    l = leads.filter(~pl.col('key').is_in(excluded))
    r = dict(scenario=name, excluded=excluded, panel_rows=panel.height, panel_events=int(panel['target'].sum()), scored_rows=sum(w), scored_events=sum((a[y]['n_pos'] for y in years)), folds=years, mean_delta=float(d.mean()), ci=P.boot_ci(d), weighted_delta=float(np.average(d, weights=w)), weighted_ci=P.boot_ci(d, w), negative_folds=int((d < 0).sum()), lead_series=l.height, widening=int((l['widening'] > 1 + 1e-10).sum()), median_widening=float(l['widening'].median()), warnings=sorted(set((str(x.message) for x in caught))))
    results.append(r)
    foldrows.extend((dict(scenario=name, year=y, auc_baseline=a[y]['auc'], auc_enriched=b[y]['auc'], delta=b[y]['auc'] - a[y]['auc'], rows=a[y]['n_total'], events=a[y]['n_pos']) for y in years))
    print(name, r['mean_delta'], r['ci'], flush=True)
    (HERE / 'data/sample_sensitivities.json').write_text(json.dumps(dict(scenarios=results), indent=2) + '\n')
    pl.DataFrame(foldrows).write_csv(HERE / 'data/sample_sensitivity_folds.csv')
