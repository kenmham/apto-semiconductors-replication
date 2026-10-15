from pathlib import Path
import importlib.util,json,warnings
import numpy as np,polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parents[1]/'outputs';R=H.parent;O=H/'data';F=H/'figures'
spec=importlib.util.spec_from_file_location('paired',R/'analysis/paired_null.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
excluded={'currentRatingAmps|RF FETs, MOSFETs','operatingSupplyCurrent|LDO Voltage Regulators'}
selected=P.J.load_selected();P.J.load_selected=lambda:[r for r in selected if r[2] not in excluded]
panel=P.J.build_panel();panel.write_parquet(H/'data/retained_forecast_panel.parquet')
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 a=P.forward_folds(panel,P.ARM_A,P.logistic);b=P.forward_folds(panel,P.ARM_B,P.logistic)
 ca=P.calibration(a);cb=P.calibration(b)
 boosts=[]
 for seed in P.SEEDS:
  x=P.fold_metrics(P.forward_folds(panel,P.ARM_A,lambda:P.gbm(seed)));y=P.fold_metrics(P.forward_folds(panel,P.ARM_B,lambda:P.gbm(seed)))
  boosts.append(float(np.mean([y[k]['auc']-x[k]['auc'] for k in x])))
v=pl.read_csv(R/'data/shoulder_fits.csv').filter(~(pl.col('column')+'|'+pl.col('subcategory')).is_in(excluded))
v.write_csv(H/'data/retained_shoulders.csv')
sel=pl.read_csv(R/'data/feature_selection.csv').filter(pl.col('selected')).filter(~(pl.col('column')+'|'+pl.col('subcategory')).is_in(excluded))
stats=dict(panel_rows=panel.height,events=int(panel['target'].sum()),subcategories=sel['subcategory'].n_unique(),attributes=sel['column'].n_unique(),higher=int((sel['direction']=='hi').sum()),lower=int((sel['direction']=='lo').sum()),shoulder_n=v.height,depth=int((v['depth']>=.04).sum()),lower_depth=int((v['depth_lo']>=.04).sum()),null_clear=int(v['clears'].sum()),baseline=ca,enriched=cb,boosting_mean_deltas=boosts)
(H/'data/retained_summary.json').write_text(json.dumps(stats,indent=2))

ma=P.fold_metrics(a);mb=P.fold_metrics(b);years=sorted(a);delta=np.array([mb[y]['auc']-ma[y]['auc'] for y in years]);ci=P.boot_ci(delta)
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');ax=axs[0]
ax.bar(years,delta,color='#176b87');ax.axhline(0,color='black',lw=.8);ax.axhline(delta.mean(),color='#b45a30',ls='--');ax.axhspan(*ci,color='#b45a30',alpha=.12);ax.set(title='A  Paired annual AUC differences',xlabel='Test year',ylabel='Enriched − baseline AUC');ax.text(.03,.05,f'Mean {delta.mean():.4f}\n95% fold-bootstrap interval [{ci[0]:.4f}, {ci[1]:.4f}]',transform=ax.transAxes,fontsize=9)
ax=axs[1]
for obj,label,col in [(ca,'Frontier history','#176b87'),(cb,'Catalog enriched','#b45a30')]:ax.plot(obj['pred'],obj['obs'],'o-',label=label,color=col)
ax.plot([0,1],[0,1],':',color='gray');ax.set(title='B  Retained-panel calibration',xlabel='Mean predicted probability',ylabel='Observed event fraction',xlim=(0,1),ylim=(0,1));ax.legend()
fig.savefig(F/'figure_04.png',dpi=180);plt.close(fig)

near=pl.read_csv(H/'data/anchor_near.csv').sort('near_share');lead=pl.read_csv(O/'lead_ratios.csv').head(6)
fig,axs=plt.subplots(1,2,figsize=(13,5.3),gridspec_kw={'width_ratios':[.8,1.45]},layout='constrained');ax=axs[0]
ax.scatter(np.arange(29)+1,near['near_share']*100,color='#176b87');ax.axhline(near['near_share'].median()*100,color='#b45a30',ls='--',label='Median 1.46%');ax.set(yscale='log',title='A  Near-frontier listing shares',xlabel='Anchor rank (29 capabilities)',ylabel='Listings with r ≥ 0.75 (%)');ax.legend(fontsize=9)
ax=axs[1]
labels=[]
for i,r in enumerate(lead.iter_rows(named=True)):
 ax.plot([r['gap_start'],r['gap_end']],[i,i],color='#adb8bd');ax.scatter(r['gap_start'],i,color='#176b87',label='Start' if i==0 else None);ax.scatter(r['gap_end'],i,color='#b45a30',label='End' if i==0 else None)
 labels.append(r['column']+'\n'+r['subcategory']+f" ({r['start_year']}–{r['end_year']})")
ax.set_yticks(range(6),labels,fontsize=8);ax.invert_yaxis();ax.set(xscale='log',title='B  Six largest retained end ratios',xlabel='Reported-specification lead ratio (log scale)');ax.legend(loc='lower right',fontsize=9);ax.grid(axis='x',alpha=.15)
fig.savefig(F/'figure_02.png',dpi=180);plt.close(fig)
lead.write_csv(O/'figure2_selected.csv')

import sys
from episode_length import episodes_for_feature
import specs as sp

counts=[]
for threshold in [.05,.1,.15,.2,.25,.3,.4]:
 es=[]
 for col,sub,label,direction in sp.SPECS:
  z=sp.frontier_series(sp.scan(),col,sub,direction)
  if z is not None:
   yr,g,total=sp.step_gains(z);es+=episodes_for_feature(yr,g,threshold)
 counts.append((threshold,len(es),sum(e['length']==1 for e in es)))
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');ax=axs[0];ax.plot([x[0] for x in counts],[100*x[2]/x[1] for x in counts],'o-',color='#176b87')
for t,n,k in counts:ax.annotate(str(n),(t,100*k/n),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
ax.set(title='A  Single-year advance episodes',xlabel='Advance threshold (log₁₀ units)',ylabel='Singleton episodes (%)',ylim=(60,110))
ax=axs[1];vals=[stats['depth'],stats['lower_depth'],stats['null_clear']];ax.bar(range(3),100*np.array(vals)/204,color=['#176b87','#4d8d98','#b45a30']);ax.set_xticks(range(3),['Depth ≥ .04','Lower bound ≥ .04','Clears fitted null'],fontsize=9)
for i,n in enumerate(vals):ax.text(i,100*n/204+1,f'{n}/204',ha='center')
ax.set(title='B  Distinct shoulder criteria',ylabel='Retained capabilities (%)',ylim=(0,60));fig.savefig(F/'figure_03.png',dpi=180);plt.close(fig)
print(json.dumps({k:v for k,v in stats.items() if k not in ['baseline','enriched']},indent=2))
print('Calibration',ca['slope'],ca['brier'],cb['slope'],cb['brier'])
