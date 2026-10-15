from pathlib import Path
import sys
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE=Path(__file__).resolve().parents[1]/'outputs'
import specs
OUT=HERE/'figures'
plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
fig=plt.figure(figsize=(14,12),facecolor='white')
gs=fig.add_gridspec(2,2,left=.07,right=.97,bottom=.09,top=.94,hspace=.39,wspace=.24)
a,b,c,d=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
blue='#176b87';orange='#b45a30';gray='#52616b'
lf=specs.scan();s=specs.frontier_series(lf,'clockFrequency','Memory','hi')
p=lf.filter(pl.col('subcategory')=='Memory').select('release_year','clockFrequency').drop_nulls().filter(pl.col('clockFrequency')>0).collect()
p=p.filter(pl.col('release_year').is_in(s['yr'].tolist()))

a.scatter(p['release_year'],np.log10(p['clockFrequency']),s=6,alpha=.08,color=gray,rasterized=True,label='Listed specifications')
a.plot(s['yr'],np.log10(s['praw']),'o--',lw=1.4,ms=3,color=orange,label='Annual p99: f')
a.step(s['yr'],np.log10(s['cB']),where='post',lw=2,color=blue,label='Cumulative p99: F')
a.set(title='A  Observed catalog trajectory',xlabel='Recorded catalog year',ylabel='log₁₀ memory clock frequency (Hz)')
a.legend(fontsize=8,loc='lower right');a.grid(alpha=.15)
a.text(.02,.94,f'Memory • {p.height:,} listings in {len(s["yr"])} usable years',transform=a.transAxes,fontsize=9,color=gray)
pl.DataFrame({'year':s['yr'],'n':s['n'],'annual_p99':s['praw'],'cumulative_p99':s['cB']}).write_csv(HERE/'data/figure1_empirical_series.csv')

y=np.arange(8);raw=np.array([0,.05,.3,.46,.4,.44,.72,.7]);F=np.maximum.accumulate(raw)
b.plot(y,raw,'o--',color=orange,label='Annual quantile f',lw=1.3)
b.step(y,F,where='post',color=blue,lw=2,label='Cumulative frontier F')
b.annotate('',(2,.30),(2,.05),arrowprops={'arrowstyle':'<->','color':'black'})
b.text(2.12,.16,'g',fontsize=12)
b.annotate('',(2,.83),(3,.83),arrowprops={'arrowstyle':'|-|','color':gray});b.text(2.5,.87,'L = 2',ha='center',fontsize=9)
b.annotate('',(3,.60),(6,.60),arrowprops={'arrowstyle':'<->','color':gray});b.text(4.5,.64,'T = 3 years',ha='center',fontsize=9)
b.annotate('',(3,-.10),(5,-.10),arrowprops={'arrowstyle':'<->','color':gray});b.text(4,-.17,'a at year 5 = 2 years',ha='center',fontsize=9)
b.set(xlim=(-.3,7.4),ylim=(-.23,1),title='B  Advance timing — schematic',xlabel='Illustrative calendar year',ylabel='Oriented log₁₀ capability')
b.legend(fontsize=8,loc='upper left');b.grid(alpha=.12)
b.text(0,-.26,'g = F(t) − F(previous usable year); advance if g > 0.10\nL: consecutive advance years; T: interval between advances\na: years since prior advance; c = g(t) / [g(t)+g(t+1)+g(t+2)]',transform=b.transAxes,fontsize=9,linespacing=1.6)
c.set_axis_off();c.set_title('C  Population position and shape — schematic',loc='left')
c.text(.01,.91,'Direction d = +1 (higher is better), −1 (lower is better)',fontsize=9,transform=c.transAxes)
c.text(.01,.80,r'Listing position: $r=(x/B)^d$; $B=10^{dF}$',fontsize=11,transform=c.transAxes)
c.text(.01,.70,'Near-frontier share: fraction of listings with r ≥ 0.75',fontsize=10,transform=c.transAxes)
c.text(.01,.60,'Lead ratio: (cumulative p95 / smoothed median)ᵈ',fontsize=10,transform=c.transAxes)
ins=c.inset_axes([.10,.13,.80,.35]);u=np.linspace(0,1,160);density=.8*np.exp(-5*u)+.31*np.exp(-((u-.8)/.13)**2);density/=density[0]
ins.plot(u,density,color=blue,lw=2);v=np.argmin(np.where(u<=.88,density,np.inf));peak=v+np.argmax(density[v:]);ins.plot([u[v],u[peak]],[density[v]]*2,':',color=gray);ins.annotate('',(u[peak],density[peak]),(u[peak],density[v]),arrowprops={'arrowstyle':'<->'});ins.text(u[peak]+.04,(density[peak]+density[v])/2,'D');ins.set(xlabel='Normalized position beyond mode, u',ylabel='Density');ins.tick_params(labelsize=8)
c.text(.01,-.03,'D: maximum rebound from trough, including the trough.\nPooled density boundary ≠ annual frontier; D is not a buyer threshold.',fontsize=9,transform=c.transAxes,linespacing=1.5)
d.set_axis_off();d.set_title('D  What enters the forecast?',loc='left')
def box(y,h,text,color):
 d.add_patch(FancyBboxPatch((.01,y),.97,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='#cad4d9',transform=d.transAxes));d.text(.04,y+h-.035,text,va='top',fontsize=10,transform=d.transAxes,linespacing=1.6)
box(.64,.29,'BASELINE  •  4 frontier-history variables\nRecent frontier change • raw-quantile change\nAge since prior advance • maturity','#e9f2f5')
box(.12,.46,'ENRICHED  •  baseline + 20 catalog variables\nCompetitive structure                         4\nCross-feature spillovers                       2\nCatalog dynamics                               2\nPopulation motion                              9\nChallenger trajectories                        3','#f8eee6')
d.text(.02,.015,'Predictors use information preceding the outcome year.\nManufacturer identities and listing history are required.',fontsize=9,transform=d.transAxes,linespacing=1.5)
fig.text(.07,.025,'Catalog measurements describe listed specifications, not sales or demand. B–D explain definitions; they are not additional empirical results.',fontsize=10,color=gray)
for ax in (a,b,ins):
 ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'figure_01.png',dpi=180)
fig.savefig(OUT/'figure_01.svg')
plt.close(fig)
print('Wrote editable SVG, PNG, and empirical series CSV.')
