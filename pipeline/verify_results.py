"""Check regenerated outputs against the manuscript's reported empirical values."""
from pathlib import Path
import json
import math
import polars as pl

O = Path(__file__).resolve().parents[1]/'outputs'
checks = []
def read(name):
    return json.loads((O/name).read_text())
def exact(name, actual, expected):
    if actual != expected:
        raise ValueError(f'{name}: {actual!r} != {expected!r}')
    checks.append(name)
def displayed(name, actual, expected, digits):
    if not math.isfinite(actual) or abs(actual-expected) > .5*10**(-digits)+1e-12:
        raise ValueError(f'{name}: {actual!r} does not match reported {expected}')
    checks.append(name)

a = read('data/audit_summary.json')
exact('catalog records',a['catalog']['records'],1739040)
exact('anchor capabilities',a['anchors']['eligible'],29)
displayed('median near-frontier share (%)',a['anchors']['median_near_percent'],1.46,2)
c = read('data/concentration.json')
t = next(r for r in c['thresholds'] if r['threshold']==.1)
exact('substantial-advance episodes',t['episodes'],63)
exact('single-year episodes',t['singletons'],49)
exact('eligible concentration windows',c['windows'],73)
displayed('initial gain share (%)',100*c['initial_share_mean'],79.3,1)
displayed('peak versus duration rho',c['peak_duration']['rho'],.215,3)
displayed('peak versus duration p',c['peak_duration']['p'],.090,3)
displayed('initial versus continuation rho',c['initial_continuation']['rho'],-.153,3)
displayed('initial versus continuation p',c['initial_continuation']['p'],.197,3)
exact('lead-ratio table rows',pl.read_csv(O/'data/lead_ratios.csv').height,204)
r = read('data/retained_summary.json')
for key,value in [('panel_rows',2449),('events',270),('subcategories',95),('attributes',121),('higher',138),('lower',66),('shoulder_n',204),('depth',51),('lower_depth',35),('null_clear',93)]:
    exact('retained '+key,r[key],value)
for arm,slope,brier in [('baseline',.436,.0980),('enriched',.057,.1048)]:
    displayed(arm+' calibration slope',r[arm]['slope'],slope,3)
    displayed(arm+' Brier score',r[arm]['brier'],brier,4)
for i,value in enumerate([-.04043,-.03875,-.04639]):
    displayed('boosting seed '+str(i),r['boosting_mean_deltas'][i],value,5)
s = {r['scenario']:r for r in read('data/sample_sensitivities.json')['scenarios']}
for name,mean,weighted in [('original_206',-.04105,-.03539),('retained_204',-.04167,-.03610),('retained_203',-.04405,None)]:
    displayed(name+' mean AUC difference',s[name]['mean_delta'],mean,5)
    if weighted is not None:
        displayed(name+' weighted AUC difference',s[name]['weighted_delta'],weighted,5)
for actual,expected in zip(s['retained_203']['ci'],[-.08020,-.01542]):
    displayed('additional exclusion interval',actual,expected,5)
timing = pl.read_csv(O/'data/timing_stratification/timing_statistics.csv')
for arm,stat,column,value,digits in [('full','s2','observed',.793,3),('full','s2','p_two_sided',.006,3),('blockperm','s2','p_two_sided',.032,3),('full','s3','observed',-.098,3),('full','s3','p_two_sided',.048,3),('blockperm','s3','p_two_sided',.716,3),('post2006','s2','observed',.818,3),('post2006','s2','p_two_sided',.002,3)]:
    row=timing.filter((pl.col('arm')==arm)&(pl.col('stat')==stat)).row(0,named=True)
    displayed(arm+' '+stat+' '+column,row[column],value,digits)
h = read('data/timing_stratification/h3_statistics.json')
for key,value in [('capabilities',204),('scored_rows',2394),('events',262),('eligible_features',160)]:
    exact('stratification '+key,h[key],value)
for stat,rho,p in [('aucA',.099,.218),('delta',.015,.847),('aucB',.144,.072)]:
    displayed(stat+' depth correlation',h['spearman'][stat]['rho'],rho,3)
    displayed(stat+' depth permutation p',h['spearman'][stat]['p_two_sided'],p,3)
for cut,pa,pd in [('category',.984,.804),('subcategory',.775,.516),('volume',.703,.105)]:
    displayed(cut+' max AUC p',h['maxstat'][cut]['p_max_auc'],pa,3)
    displayed(cut+' max enrichment p',h['maxstat'][cut]['p_max_delta'],pd,3)
k = read('data/coupling.json')
for case,value in zip(k['cases'],[.70,.76]):
    displayed(case['column']+' coupling',case['pearson_r'],value,2)
exact('FPGA stall years',k['cases'][0]['stall']['n'],9)
displayed('FPGA stall mean',k['cases'][0]['stall']['mean'],-.195,3)
exact('pooled stall observations',k['pooled_stall']['n'],494)
displayed('pooled declining share (%)',100*k['pooled_stall']['negative_share'],37,0)
p = read('data/pilots.json')
f = [r for r in p['forecast'] if r['version']=='v1']
for row,value in zip(f,[.535,.605]):
    exact(row['rung']+' pilot rows',row['rows'],162)
    exact(row['rung']+' pilot advances',row['events'],30)
    displayed(row['rung']+' pilot AUC',row['auc'],value,3)
for key,value in [('base_cases',12),('passes',2),('facts_including_placebo',9)]:
    exact('ablation '+key,p['ablation'][key],value)
displayed('pilot policy association',p['ablation']['policy_truth_spearman'],-.07,2)
displayed('packaging effect (points)',p['ablation']['packaging_points'],9,0)
for row,real,affirmed,auc in zip(p['attribution'],[19,11],[15,1],[.66,.49]):
    exact(row['subcategory']+' actual advances',row['real'],real)
    exact(row['subcategory']+' affirmations',row['affirmed_real'],affirmed)
    displayed(row['subcategory']+' attribution AUC',row['auc'],auc,2)
for i in range(1,12):
    exact('Figure '+str(i)+' PNG exists',(O/'figures'/f'figure_{i:02d}.png').is_file(),True)
print(f'{len(checks)} manuscript-value and output checks passed.')
