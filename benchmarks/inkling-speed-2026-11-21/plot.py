"""Render the served before/after comparison without treating rejected requests as measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt
import numpy as np
p=Path(__file__).resolve().parent
base=json.loads((p/'baseline.json').read_text())
new=json.loads((p/'optimized.json').read_text())
fresh=json.loads((p/'fresh-pairs.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(14,5.2))
fig.subplots_adjust(top=.75,bottom=.19,wspace=.32)
x=np.arange(3);w=.35
for ax,(users,metric,title) in zip(axes,[(1,'median_decode_tps','Generation · one user'),(2,'median_decode_tps','Generation · two users, each'),(1,'median_prompt_tps','Prompt processing · one user')]):
 for shift,data,label,color in [(-w/2,base,'Before','#1687d9'),(w/2,new,'After','#ff8b3d')]:
  cells=fresh['profiles']['baseline' if label=='Before' else 'optimized'] if users==2 else [c for c in data['cells'] if c['users']==users]
  vals=[c.get(metric) if c['admitted']==users else np.nan for c in cells]
  bars=ax.bar(x+shift,vals,w,label=label,color=color)
  for b,v in zip(bars,vals):
   if np.isfinite(v):ax.text(b.get_x()+b.get_width()/2,v,'{:.0f}'.format(v) if metric=='median_prompt_tps' else f'{v:.1f}',ha='center',va='bottom',fontsize=10)
   else:ax.text(b.get_x()+b.get_width()/2,.04,'RAM\nlimited',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=8,color='#777777')
 ax.set_title(title,fontsize=12);ax.set_xticks(x,['4K','8K','32K']);ax.set_xlabel('Prompt tokens');ax.set_ylabel('Tokens / second');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 ax.set_ylim(0,2000 if metric=='median_prompt_tps' else 60 if users==1 else 35)
fig.suptitle('Inkling 2 Mini · faster streaming on M5 Max',fontsize=19,fontweight='bold',y=.97)
fig.text(.5,.88,'Same 8-bit checkpoint · three passes · MLX 0.32.4 · 256 output tokens',ha='center',fontsize=11,color='#555555')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.85),ncol=2,frameon=False)
fig.text(.5,.035,'One measured attempt per cell; fresh KV. Hashes match for admitted requests.\nTwo-user panel: fresh backend per cell. Earlier warm-model RAM refusals remain in the raw results.',ha='center',fontsize=9,color='#555555')
fig.savefig(p/'speed.png',dpi=150,facecolor='white');fig.savefig(p/'speed.svg',facecolor='white')
f=p/'speed.svg';f.write_text('\n'.join(line.rstrip() for line in f.read_text().splitlines())+'\n')
