import argparse,csv,json,pathlib,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator
p=argparse.ArgumentParser();p.add_argument('report',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args();r=json.loads(a.report.read_text());a.output.mkdir(parents=True,exist_ok=True)
levels=[1,2,4,8];lengths=[4000,8000,32000];colors=['#246CC0','#E68A35','#B55D93','#4D7682'];rows=[]
for c in r['cells']:
 successful=[x for x in c['rows'] if 'error' not in x]
 rows.append(dict(input_tokens=c['input_tokens'],users=c['users'],admitted=c.get('admitted',len(successful)),median_effective_prompt_tps=statistics.median(x['usage']['prompt_tokens']/(x['first']-x['sent']) for x in successful) if successful else None,median_active_prompt_tps=c.get('median_prompt_tps'),median_decode_tps=c.get('median_decode_tps'),aggregate_prompt_tps=c.get('aggregate_prompt_tps'),aggregate_decode_tps=c.get('aggregate_decode_tps'),median_ttft_s=c.get('median_ttft')))
lookup={(x['input_tokens'],x['users']):x for x in rows}
plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#29333D','text.color':'#172432','axes.edgecolor':'#AEB7BE','xtick.color':'#344352','ytick.color':'#344352'})
def figure(name,metrics,titles):
 fig,axes=plt.subplots(2,3,figsize=(16,9),sharey='row');fig.patch.set_facecolor('white')
 for row,(metric,title) in enumerate(zip(metrics,titles)):
  maxval=max((x[metric] or 0 for x in rows if x['admitted']==x['users']),default=1)
  for col,n in enumerate(lengths):
   ax=axes[row,col];ax.set_axisbelow(True);ax.grid(axis='y',color='#E9EDF0',linewidth=.8);values=[]
   for users in levels:
    x=lookup.get((n,users));values.append(x[metric] if x and x['admitted']==users else 0)
   bars=ax.bar(range(4),values,color=colors,width=.64,zorder=3)
   for i,(bar,v) in enumerate(zip(bars,values)):
    x=lookup.get((n,levels[i]));label=f'{v:,.0f}' if x and x['admitted']==levels[i] else ('Not measured' if not x else f"{x['admitted']}/{levels[i]} admitted")
    ax.text(bar.get_x()+bar.get_width()/2,v+maxval*.025,label,ha='center',va='bottom',fontsize=11)
   ax.set_ylim(0,maxval*1.2);ax.set_xticks(range(4),levels);ax.set_xlabel('Concurrent users');ax.yaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}'));ax.set_title(f'{n//1000}K input tokens',pad=14)
   if col==0:ax.set_ylabel(title+'\n(tokens/sec)',labelpad=12)
 fig.suptitle('Liquid LFM2.5 · 96,000-token window',x=.08,ha='left',fontsize=23,fontweight='bold',y=.985)
 fig.text(.08,.928,'DSpark 8-bit drafter · 256-token replies · distinct synthetic prompts · cache disabled · one attempt per cell',fontsize=12,color='#536575')
 fig.text(.08,.018,'Per-user rates include pauses during overlapping requests. Aggregate rates describe the whole group. Synthetic inputs measure speed, not quality.',fontsize=10.5,color='#536575')
 fig.subplots_adjust(top=.86,bottom=.105,left=.08,right=.985,wspace=.14,hspace=.48)
 fig.savefig(a.output/(name+'.png'),dpi=170,facecolor='white');fig.savefig(a.output/(name+'.svg'),facecolor='white');plt.close(fig)
 svg=a.output/(name+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
figure('liquid-per-user',['median_effective_prompt_tps','median_decode_tps'],['Effective prompt processing','Output generation per user'])
figure('liquid-aggregate',['aggregate_prompt_tps','aggregate_decode_tps'],['Aggregate prompt processing','Aggregate output generation'])
with (a.output/'results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(a.output/'chart-data.json').write_text(json.dumps({'context':96000,'output_tokens':256,'repetitions':1,'metrics':rows},indent=2))
print(a.output)
