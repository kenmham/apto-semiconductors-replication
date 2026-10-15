from pathlib import Path
import sys,json
import numpy as np
import polars as pl
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1]; O=R/'outputs/data'
import specs,episode_length as E,second_year_share as W
lf=specs.scan(); episodes=[]; windows=[]
for col,sub,label,direction in specs.SPECS:
    s=specs.frontier_series(lf,col,sub,direction)
    if s is not None:
        years,gains,_=specs.step_gains(s)
        for threshold in E.THRESHOLDS:
            episodes.extend(dict(feature=label,threshold=threshold,**e) for e in E.episodes_for_feature(years,gains,threshold))
        windows.extend(dict(feature=label,year=y,g1=g1,g2=g2,g3=g3,initial=g1/(g1+g2+g3)) for y,g1,g2,g3 in W.jumps_for_feature(years,gains,.1))
e=pl.DataFrame(episodes);w=pl.DataFrame(windows)
e.write_csv(O/'concentration_episodes.csv');w.write_csv(O/'concentration_windows.csv')
e0=e.filter(pl.col('threshold')==.1)
r1=spearmanr(e0['peak'],e0['length']);r2=spearmanr(w['g1'],1-w['initial'])
summary={'thresholds':[{'threshold':t,'episodes':len(g:=e.filter(pl.col('threshold')==t)),'singletons':g.filter(pl.col('length')==1).height} for t in E.THRESHOLDS], 'windows':len(w),'initial_share_mean':w['initial'].mean(),'peak_duration':{'rho':float(r1.statistic),'p':float(r1.pvalue)},'initial_continuation':{'rho':float(r2.statistic),'p':float(r2.pvalue)}}
(O/'concentration.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
