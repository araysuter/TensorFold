import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import numpy as np
p=Path(__file__).resolve().parent
d=json.loads((p/'dspark-followup.json').read_text())
keys=['spec-followup-fixed8-verify4','spec-followup-width4-16','spec-followup-width8-5bit','spec-followup-native-width4']
labels=['5-bit: 8 drafted / 4 verified','5-bit: 4-token block','5-bit: 8-token block','Original FP4/FP8: 4-token block']
colors=['#1687d9','#ff8b3d','#b564b3','#49a890']
fig,axes=plt.subplots(1,2,figsize=(13,5.6));fig.subplots_adjust(top=.79,bottom=.26,wspace=.24)
for key,label,color in zip(keys,labels,colors):
 c=d[key]['cases']; x=[v['input_tokens']/1000 for v in c]
 axes[0].plot(x,[v['candidate']['decode_tps'] for v in c],'-o',label=label,color=color,lw=2)
 axes[1].plot(x,[100*(1-v['candidate']['accepted']/v['candidate']['proposed']) for v in c],'-o',color=color,lw=2)
base=d['spec-followup-width8-5bit']['cases']
axes[0].plot([x['input_tokens']/1000 for x in base],[x['plain']['decode_tps'] for x in base],':o',color='#444444',label='5-bit plain reference',lw=2)
axes[0].axhline(80,color='#999999',ls='--',lw=1);axes[0].text(31.8,81.5,'80 tok/s target',ha='right',color='#777777',fontsize=9)
for ax in axes:
 ax.set_xticks([4,8,32],['4K','8K','32K']);ax.set_xlabel('Prompt tokens');ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylim(0,90);axes[0].set_ylabel('Generated tokens / second');axes[0].set_title('Generation speed')
axes[1].set_ylim(0,105);axes[1].set_ylabel('Proposals discarded (%)');axes[1].set_title('Drafts not committed')
fig.suptitle('Inkling 2 Mini · DSpark follow-up on M5 Max',fontsize=19,fontweight='bold',y=.97)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False,fontsize=9)
fig.text(.5,.015,'One user · fresh KV · 256 output tokens · one attempt/cell · all drafted outputs equal their paired plain outputs.\nLines interpolate measured points; 4-token 5-bit block was not measured at 8K. Discarded includes unused proposal tails.',ha='center',fontsize=8,color='#555555')
fig.savefig(p/'speed-and-rejection.png',dpi=160,facecolor='white');fig.savefig(p/'speed-and-rejection.svg',facecolor='white')

svg = p / 'speed-and-rejection.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
