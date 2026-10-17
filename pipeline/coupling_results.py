"""Reproduce the two coupling cases and pooled stall comparison."""
from pathlib import Path
import json
import numpy as np
import polars as pl
import specs
import coupling as C
O = Path(__file__).resolve().parents[1]/'outputs/data'
lf = specs.scan()
changes = [dict(feature=label, advanced=bool(a), typical_change=g)
           for col,sub,label,direction in specs.SPECS
           for a,g in C.feature_changes(col,sub,direction,.25,2000,2024,lf)]
c = pl.DataFrame(changes)
c.write_csv(O/'coupling_changes.csv')
def summ(g):
    a=g['typical_change'].to_numpy()
    return {'n':len(a),'mean':float(np.mean(a)),'median':float(np.median(a)),'negative_share':float(np.mean(a<0))}
cases=[]
for col,sub in [('numberOfLogicElementsCells','FPGAs (Field Programmable Gate Array)'),('frequencySwitching','Voltage Regulators - DC DC Switching Regulators')]:
    rows=C.build(col,sub,'hi',.25,2000,2024); a=np.array(rows);x,y=C.detrend_residuals(a[:,0],a[:,1],a[:,2]); ch=pl.DataFrame([{'advanced':bool(adv),'typical_change':g} for adv,g in C.feature_changes(col,sub,'hi',.25,2000,2024,lf)])
    cases.append({'column':col,'subcategory':sub,'n_years':len(rows),'pearson_r':float(np.corrcoef(x,y)[0,1]),'stall':summ(ch.filter(~pl.col('advanced'))),'advance':summ(ch.filter(pl.col('advanced')))})
(O/'coupling.json').write_text(json.dumps({'cases':cases,'pooled_stall':summ(c.filter(~pl.col('advanced'))),'pooled_advance':summ(c.filter(pl.col('advanced')))},indent=2)+'\n')
print((O/'coupling.json').read_text())
