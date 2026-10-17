"""Recompute Appendix G scores from saved responses; no model calls."""
from pathlib import Path
import json
import numpy as np
import polars as pl
from scipy.stats import spearmanr, pointbiserialr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import jump_cv as J
from pilot_prompts import REAL_FEATURES
R = Path(__file__).resolve().parents[1]
D = R/'data/pilots'
O = R/'outputs/data'
forecast = []
for version, score in [('v1', 'jump_likelihood'), ('v2', 'score')]:
    d = pl.read_csv(D/f'forecast_{version}.csv', infer_schema_length=10000)
    d = d.filter(pl.col('ok') & pl.col(score).is_not_null())
    for rung in sorted(d['rung'].unique()):
        s = d.filter(pl.col('rung') == rung)
        forecast.append(dict(version=version, rung=rung, rows=s.height, events=int(s['target'].sum()),
                             auc=float(roc_auc_score(s['target'], s[score]))))
a = pl.read_csv(D/'ablation.csv').filter(pl.col('ok'))
w = a.pivot(values='propensity', index=['base_id','pass_i','feature'], on='level',
            aggregate_function='first').with_columns((pl.col('hi')-pl.col('lo')).alias('sensitivity'))
p = w.group_by('feature').agg(policy=pl.col('sensitivity').mean(),
                            policy_se=pl.col('sensitivity').std()/pl.len().sqrt()).sort('feature')
rel = w.pivot(values='sensitivity',index=['base_id','feature'],on='pass_i',aggregate_function='first').drop_nulls()
reliability = float(np.corrcoef(rel['1'],rel['2'])[0,1])
# Historical truth associations used the original 206-capability panel.
panel = J.build_panel()
y = panel['target'].to_numpy().astype(int)
X = panel.select(REAL_FEATURES).to_numpy().astype(float)
Xi = np.where(np.isnan(X), np.nanmedian(X,axis=0), X)
uni = dict(zip(REAL_FEATURES,[float(pointbiserialr(y,Xi[:,i]).statistic)
                             if Xi[:,i].std()>0 else 0.0 for i in range(len(REAL_FEATURES))]))
multi = dict(zip(REAL_FEATURES,LogisticRegression(max_iter=1000,C=1.0).fit(StandardScaler().fit_transform(Xi),y).coef_[0]))
policy = {r['feature']:r for r in p.iter_rows(named=True)}
rows = [dict(feature=f, policy=round(r['policy'],1), policy_se=round(r['policy_se'],1),
             truth_uni=round(uni.get(f,0.0),3), truth_multi=round(multi.get(f,0.0),3))
        for f,r in sorted(policy.items())]
pl.DataFrame(rows).write_csv(O/'pilot_recomputed_associations.csv')
historical = pl.read_csv(D/'policy_vs_truth.csv').sort('feature')
for record in historical.iter_rows(named=True):
    actual = policy[record['feature']]
    assert round(actual['policy'],1) == record['policy']
    assert round(actual['policy_se'],1) == record['policy_se']
historical.write_csv(O/'pilot_policy_vs_truth.csv')
t = [r for r in historical.iter_rows(named=True) if r['feature']!='packaging']
recomputed = [r for r in rows if r['feature']!='packaging']
recomputed_correlation = float(spearmanr([r['policy'] for r in recomputed],[r['truth_uni'] for r in recomputed]).statistic)
correlation = float(spearmanr([r['policy'] for r in t],[r['truth_uni'] for r in t]).statistic)
at = pl.read_csv(D/'attribution.csv',infer_schema_length=10000).filter(pl.col('ok'))
at = at.with_columns((pl.col('occurred').str.to_lowercase()=='yes').alias('affirmed'),
    ((~pl.col('driver').str.to_lowercase().is_in(['','none','n/a','unknown'])) &
     (pl.col('driver_type').str.to_lowercase()!='none')).alias('specific'),
    pl.col('confidence').str.to_lowercase().replace_strict({'low':0.0,'medium':1.0,'high':2.0},default=0.0).alias('conf'))
at = at.with_columns((pl.col('affirmed').cast(pl.Float64)+pl.col('specific').cast(pl.Float64)+pl.col('conf')/2).alias('score'))
at.write_csv(O/'pilot_attribution_scores.csv')
att = []
for sub in sorted(at['subcategory'].unique()):
    s=at.filter(pl.col('subcategory')==sub);real=s.filter(pl.col('condition')=='real')
    att.append(dict(subcategory=sub,queries=s.height,real=real.height,affirmed_real=int(real['affirmed'].sum()),
                    auc=float(roc_auc_score(s['condition']=='real',s['score']))))
batch=pl.read_csv(D/'batch_bias.csv')
result=dict(forecast=forecast,ablation=dict(base_cases=a['base_id'].n_unique(),passes=a['pass_i'].n_unique(),
    facts_including_placebo=a['feature'].n_unique(),pass_reliability=reliability,
    packaging_points=policy['packaging']['policy'],truth_panel_rows=panel.height,truth_panel_events=int(y.sum()),
    policy_truth_spearman=correlation,recomputed_policy_truth_spearman=recomputed_correlation),attribution=att,
    batch_retest=dict(rows=batch.height,original_retest_correlation=float(np.corrcoef(batch['orig'],batch['retest'])[0,1]),
                      original_solo_correlation=float(np.corrcoef(batch['orig'],batch['solo'])[0,1])))
(O/'pilots.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
