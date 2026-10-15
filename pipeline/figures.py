from pathlib import Path
import importlib.util
import json
import os
import warnings
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

OUT = ROOT / 'outputs/diagnostics'
FIG = ROOT / 'outputs/figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

P = module('paired_revision', ROOT / 'analysis/paired_null.py')
panel_path = ROOT / 'outputs/data/retained_forecast_panel.parquet'
panel = pl.read_parquet(panel_path)
assert panel.height == 2449 and panel['label'].n_unique() == 204
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    folds_a = P.forward_folds(panel, P.ARM_A, P.logistic)
    folds_b = P.forward_folds(panel, P.ARM_B, P.logistic)
    ca, cb = P.calibration(folds_a), P.calibration(folds_b)
ma, mb = P.fold_metrics(folds_a), P.fold_metrics(folds_b)
years = sorted(folds_a)
assert years == sorted(folds_b) == list(range(2005, 2024))
delta = np.array([mb[y]['auc'] - ma[y]['auc'] for y in years])
assert abs(delta.mean() - (-0.04167)) < 0.00001
summary_path = ROOT / 'outputs/data/retained_summary.json'
saved = json.loads(summary_path.read_text())
for current, original in [(ca, saved['baseline']), (cb, saved['enriched'])]:
    for key in ['pred', 'obs', 'cnt', 'brier', 'slope']:
        assert np.allclose(current[key], original[key], rtol=1e-6, atol=1e-8), key

added = [c for c in P.ARM_B if c not in P.ARM_A]
records, predictions, influences, subgroups = [], [], [], []
for year in years:
    tr = panel.filter(pl.col('year') < year)
    te = panel.filter(pl.col('year') == year)
    y, pa = folds_a[year]
    _, pb = folds_b[year]
    labels = te['label'].to_list()
    prev = panel.filter(pl.col('year') == year - 1)
    def dist(frame):
        return {r['subcategory']: r['len'] / frame.height for r in frame.group_by('subcategory').len().iter_rows(named=True)}
    now, before = dist(te), dist(prev)
    tv = .5 * sum(abs(now.get(s, 0) - before.get(s, 0)) for s in set(now) | set(before))
    rec = dict(year=year, baseline_auc=ma[year]['auc'], enriched_auc=mb[year]['auc'],
               delta=mb[year]['auc']-ma[year]['auc'], train_n=tr.height,
               train_events=int(tr['target'].sum()), test_n=te.height, test_events=int(y.sum()),
               test_event_rate=float(y.mean()), subcategories=te['subcategory'].n_unique(),
               added_train_missing=float(np.isnan(tr.select(added).to_numpy()).mean()),
               added_test_missing=float(np.isnan(te.select(added).to_numpy()).mean()),
               added_all_missing_train=sum(np.isnan(tr[c].to_numpy()).all() for c in added),
               test_new_capability_share=len(set(labels)-set(tr['label']))/te.height,
               subcategory_mix_change=tv)
    records.append(rec)
    for row, a, b in zip(te.select('label','subcategory','target').iter_rows(named=True), pa, pb):
        predictions.append(dict(year=year, **row, baseline_probability=float(a), enriched_probability=float(b)))
    if year in (2009, 2016):

        for i in np.flatnonzero(y == 1):
            aa = ((pa[i] > pa[y == 0]) + .5 * (pa[i] == pa[y == 0])).mean()
            bb = ((pb[i] > pb[y == 0]) + .5 * (pb[i] == pb[y == 0])).mean()
            influences.append(dict(year=year, label=labels[i], baseline_rank=float(aa),
                                   enriched_rank=float(bb), delta_contribution=float((bb-aa)/y.sum())))
        for sub in te['subcategory'].unique().to_list():
            mask = te['subcategory'].to_numpy() != sub
            if len(np.unique(y[mask])) == 2:
                subgroups.append(dict(year=year, omitted_subcategory=sub, remaining_n=int(mask.sum()),
                    remaining_events=int(y[mask].sum()), fixed_prediction_delta=float(
                    roc_auc_score(y[mask],pb[mask])-roc_auc_score(y[mask],pa[mask]))))

pl.DataFrame(records).write_csv(OUT / 'annual_diagnostics.csv')
pl.DataFrame(predictions).write_csv(OUT / 'paired_predictions.csv')
pl.DataFrame(influences).sort('year','delta_contribution').write_csv(OUT / 'event_rank_contributions.csv')
pl.DataFrame(subgroups).sort('year','fixed_prediction_delta').write_csv(OUT / 'subcategory_omission_diagnostics.csv')
correlations = {k: float(spearmanr([r[k] for r in records],delta).statistic)
                for k in ['train_n','test_n','test_events','test_event_rate',
                          'added_train_missing','added_test_missing','subcategory_mix_change']}
sensitivities = {}
for label, mask in [('all', np.ones(len(years), bool)), ('without_2009', np.array(years)!=2009),
                    ('without_2009_and_2016', ~np.isin(years,[2009,2016])),
                    ('2010_onward',np.array(years)>=2010)]:
    d = delta[mask]
    sensitivities[label] = dict(folds=int(mask.sum()), mean=float(d.mean()),
                               negative=int((d<0).sum()), ci=P.boot_ci(d))
summary = dict(mean_delta=float(delta.mean()), ci=P.boot_ci(delta), negative_folds=int((delta<0).sum()),
               baseline=ca, enriched=cb, sensitivities=sensitivities,
               leave_one_year_out_mean_range=[float(min(np.delete(delta,i).mean() for i in range(len(delta)))),
                                             float(max(np.delete(delta,i).mean() for i in range(len(delta))))],
               descriptive_spearman=correlations,
               warnings=sorted(set(str(w.message) for w in caught)),
               panel_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest())


bins = []
for arm, folds in [('baseline',folds_a),('enriched',folds_b)]:
    y = np.concatenate([folds[t][0] for t in years])
    prob = np.concatenate([folds[t][1] for t in years])
    which = np.clip(np.digitize(prob,np.linspace(0,1,11))-1,0,9)
    for b in range(10):
        mask = which == b
        if mask.any():
            bins.append(dict(model=arm, bin=b, n=int(mask.sum()), events=int(y[mask].sum()),
                             predicted=float(prob[mask].mean()), observed=float(y[mask].mean())))
pl.DataFrame(bins).write_csv(OUT/'calibration_bins.csv')
(OUT/'diagnostics_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})
BLUE, ORANGE, PURPLE = '#176b87', '#b45a30', '#715594'
fig = plt.figure(figsize=(9,8.5))
gs = fig.add_gridspec(2,2,height_ratios=[1,1.30],hspace=.65,wspace=.38)
ax = fig.add_subplot(gs[0,:])
ax.bar(years,delta,color=[BLUE if d>0 else ORANGE for d in delta],width=.7,zorder=3)
ax.axhline(0,color='#343b40',lw=1)
ax.axhline(delta.mean(),color='#343b40',lw=1,ls='--')
ax.set(xlim=(2004.3,2023.7),ylim=(-.285,.075),xticks=years[::2],
       ylabel='AUC change from adding indicators',xlabel='Test year')
ax.tick_params(axis='x',labelrotation=45,labelsize=10)
ax.set_title('A  Lower ranking accuracy in 15 of 19 test years',loc='left',pad=15,fontweight='bold',fontsize=12)
ax.text(.01,.96,'Above zero: prediction improves',transform=ax.transAxes,color=BLUE,fontsize=9,va='top')
ax.text(.60,.33,'Below zero: prediction worsens',transform=ax.transAxes,color=ORANGE,fontsize=9)
ax.text(.46,.10,'Mean change: -0.042 AUC\n95% interval for the mean: [-0.074, -0.015]',
        transform=ax.transAxes,fontsize=10,linespacing=1.6)
for year in [2009,2016]:
    d=delta[years.index(year)]
    ax.annotate(f'{d:.3f}',(year,d),xytext=(0,-12),textcoords='offset points',ha='center',fontsize=9)
ax.grid(axis='y',alpha=.13,zorder=0)

fig.text(.10,.55,'B  Predicted chances versus observed advance frequencies',fontsize=12,fontweight='bold')
handles=[Line2D([],[],marker='o',markerfacecolor='white',markeredgecolor=BLUE,markeredgewidth=1.5,linestyle='',label='Predicted chance'),
         Line2D([],[],marker='D',color=ORANGE,linestyle='',label='Observed frequency')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.50,.54),ncol=2,frameon=False,fontsize=10)
for j,(arm,title) in enumerate([('baseline','Frontier history\nonly'),('enriched','Frontier history\n+ product indicators')]):
    ax=fig.add_subplot(gs[1,j])
    rows={r['bin']:r for r in bins if r['model']==arm}
    for b in range(10):
        if b not in rows:
            ax.text(.5,b,'No predictions',ha='center',va='center',color='#899096',fontsize=8)
            continue
        r=rows[b]
        if r['n']<20: ax.axhspan(b-.43,b+.43,color='#f0f1f2',zorder=0)
        ax.plot([r['predicted'],r['observed']],[b,b],color='#adb8bd',lw=1.5,zorder=1)
        ax.scatter(r['predicted'],b,facecolor='white',edgecolor=BLUE,s=34,lw=1.5,zorder=3)
        ax.scatter(r['observed'],b,color=ORANGE,s=25,marker='D',zorder=4)
        ax.text(1.04,b,f"{r['n']:,}",ha='left',va='center',fontsize=9,transform=ax.get_yaxis_transform())
    ax.text(1.04,1.04,'n',transform=ax.transAxes,fontweight='bold',fontsize=10)
    ax.set(ylim=(9.6,-.6),xlim=(-.025,1.025),xticks=[0,.25,.5,.75,1],
           yticks=range(10),yticklabels=[f'{b*10}-{(b+1)*10}%' for b in range(10)],
           xlabel='Probability / frequency')
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(labelsize=9)
    ax.set_title(title,fontsize=10,pad=12,fontweight='bold')
    if j==0:ax.set_ylabel('Predicted-probability band')
    ax.grid(axis='x',alpha=.15)
fig.text(.1,.022,'Gray rows: fewer than 20 predictions. n = capability-years in each model-specific band.',fontsize=9)
fig.subplots_adjust(left=.1,right=.92,top=.93,bottom=.10)
fig.savefig(FIG/'figure_04.png',dpi=240)
fig.savefig(FIG/'figure_04.pdf')
plt.close(fig)


S = module('anchor_specs_revision',ROOT/'analysis/specs.py')
lf=S.scan()
series,annual=[] ,[]
for col,sub,label,direction in S.SPECS:
    if col=='gain' and sub=='RF Amplifiers':continue
    s=S.frontier_series(lf,col,sub,direction)
    assert s is not None
    med=(lf.filter(pl.col('subcategory')==sub).select('release_year',col).drop_nulls()
         .filter(pl.col(col)>0).group_by('release_year')
         .agg(pl.col(col).median().alias('median'),pl.len().alias('n'))
         .filter(pl.col('n')>=S.MIN_PER_YEAR).sort('release_year').collect())
    assert np.array_equal(med['release_year'].to_numpy(),s['yr'])
    assert np.array_equal(med['n'].to_numpy(),s['n'])
    sign=1 if direction=='hi' else -1
    median=med['median'].to_numpy()
    p95=sign*np.log10(s['cA']/s['cA'][0]);p99=sign*np.log10(s['cB']/s['cB'][0])
    p50=sign*np.log10(median/median[0])
    series.append((label,sub,s,p95,p99,p50))
    for i,y in enumerate(s['yr']):
        annual.append(dict(column=col,subcategory=sub,direction=direction,year=int(y),n=int(s['n'][i]),
                     cumulative_p95_or_p05=float(s['cA'][i]),cumulative_p99_or_p01=float(s['cB'][i]),
                     annual_median=float(median[i]),frontier_p95_change=float(p95[i]),
                     frontier_p99_change=float(p99[i]),median_change=float(p50[i])))
assert len(series)==28
pl.DataFrame(annual).write_csv(OUT/'trajectory_annual_values.csv')
for page in range(7):
    fig,axs=plt.subplots(2,2,figsize=(8.5,7.3))
    for ax,item in zip(axs.flat,series[page*4:page*4+4]):
        label,sub,s,p95,p99,p50=item
        ax.plot(s['yr'],p95,'o-',ms=3,lw=1.4,label='Cumulative p95 / p05',color=BLUE)
        ax.plot(s['yr'],p99,'s--',ms=3,lw=1.3,label='Cumulative p99 / p01',color=ORANGE)
        ax.plot(s['yr'],p50,'D-.',ms=3,lw=1.4,label='Annual product-population median',color=PURPLE,zorder=4)
        ax.axhline(0,color='gray',lw=.5,alpha=.4)
        import textwrap
        ax.set_title(label+'\n'+textwrap.fill(sub,40),fontsize=9.5,pad=10)
        ax.set_xlabel('Recorded product year',fontsize=9.5)
        ax.set_ylabel('Oriented log10 change from own start',fontsize=9.5)
        ax.tick_params(labelsize=9);ax.grid(alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    handles,labels=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.10,right=.98,top=.9,bottom=.14,hspace=.68,wspace=.34)
    fig.savefig(FIG/f'figure_{page+5:02d}.pdf')
    fig.savefig(FIG/f'figure_{page+5:02d}.png',dpi=160)
    plt.close(fig)

print(json.dumps({k:summary[k] for k in ['mean_delta','ci','sensitivities','leave_one_year_out_mean_range','descriptive_spearman']},indent=2))
print(pl.DataFrame(records).filter(pl.col('year').is_in([2005,2009,2016,2023])))
print('Saved 28 median trajectories and the redesigned forecast figure.')
