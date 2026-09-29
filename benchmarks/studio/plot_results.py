import json, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

base=Path(__file__).resolve().parent
d=json.loads((base/'studio-benchmark-resumed/results.json').read_text())
extra=json.loads((base/'studio-benchmark-100k/results.json').read_text())
rerun=json.loads((base/'studio-benchmark-128k-higher-memory/results.json').read_text())
assert len(rerun['requests']) == 3 and all(r['status']=='ok' for r in rerun['requests'])
rows=[r for r in d['requests']+extra['requests'] if r['context_tokens'] != 128000]+rerun['requests']
batches=[r for r in d['batches']+extra['batches'] if r['context_tokens'] != 128000]+rerun['batches']
contexts=[20000,40000,60000,80000,100000,128000]
colors=['#2563eb','#e77b24']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,'axes.titleweight':'bold','axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#bcc5cf','text.color':'#172b42','axes.labelcolor':'#34465b','xtick.color':'#536377','ytick.color':'#536377'})
fig,axs=plt.subplots(2,2,figsize=(16,11.5))
fig.patch.set_facecolor('white')
fig.subplots_adjust(left=.065,right=.96,top=.81,bottom=.19,hspace=.43,wspace=.22)
fig.text(.065,.956,'TensorFold • Swift 1.5 4-bit',fontsize=24,weight='bold')
fig.text(.065,.922,'M5 Max Mac Studio · 64 GB unified memory · 1 vs 2 concurrent requests',fontsize=15,color='#536377')
fig.legend(handles=[Line2D([0],[0],color=colors[0],marker='o',lw=2,label='1 request'),Line2D([0],[0],color=colors[1],marker='s',lw=2,label='2 simultaneous submissions (may queue)')],loc='upper left',bbox_to_anchor=(.06,.90),frameon=False,ncol=2,fontsize=12)
plots=[('decode_tps','Generation speed per request','Tokens / second'),('effective_input_tps','Effective prompt throughput¹','Input tokens / second'),('ttft_seconds','Time to first token','Seconds'),('aggregate_tps','Combined end-to-end output throughput²','Output tokens / second')]
for ax,(metric,title,unit) in zip(axs.flat,plots):
    ax.set_title(title,loc='left',pad=16)
    ax.set_ylabel(unit);ax.set_xlabel('Input context (tokens)')
    ax.set_xticks(np.arange(len(contexts)),[f'{n//1000}K' + ('*' if n==128000 else '') for n in contexts])
    ax.grid(axis='y',color='#e7ebf0',lw=.8);ax.set_axisbelow(True)
    maxima=[]
    for c,col in [(1,colors[0]),(2,colors[1])]:
        values=[];lo=[];hi=[]
        for n in contexts:
            source=batches if metric=='aggregate_tps' else rows
            vals=[r[metric] for r in source if r['context_tokens']==n and r['concurrency']==c and r.get(metric) is not None]
            values.append(statistics.median(vals));lo.append(min(vals));hi.append(max(vals))
        maxima+=hi
        x=np.arange(len(contexts))+(c-1.5)*.035
        ax.errorbar(x,values,yerr=[np.array(values)-lo,np.array(hi)-values],color=col,lw=2.4,marker='o' if c==1 else 's',ms=6,capsize=3,elinewidth=1,alpha=.95)
        for xx,y in zip(x,values):
            offset=11 if c==1 else -18
            if metric in ('aggregate_tps','ttft_seconds'):offset=-19 if c==1 else 11
            ax.annotate(f'{y:.1f}',(xx,y),xytext=(0,offset),textcoords='offset points',ha='center',color=col,fontsize=10,weight='bold')
    ax.set_xlim(-.3,len(contexts)-.65)
    ax.set_ylim(-max(maxima)*.09,max(maxima)*1.16)
    ax.axhline(0,color='#b9c4d1',lw=.7)
fig.text(.065,.132,'128K rerun: overlapping responses; timings suggest generation stalls during prefill.',fontsize=13,weight='bold')
fig.text(.065,.110,'Two-user decode: 3.5 / 39.5 tok/s; first tokens at 266 / 531s; finishes at 554 / 556s. Combined output: 3.68 tok/s.',fontsize=10,color='#536377')
fig.text(.065,.091,'*128K: higher-memory rerun (58 GiB process target; actual RAM not recorded). Other points retain the earlier configuration.',fontsize=10,color='#536377')
fig.text(.065,.071,'32 successful requests · 1,024 output tokens each · uncached synthetic prompts · greedy · reasoning off · drafting on · September 28, 2026',fontsize=10,color='#536377')
fig.text(.065,.049,'Points: medians; whiskers: observed min–max. 3 batches at 20K/40K and 60K single; 1 batch elsewhere. Mixed runs; 128K replaced with the new test.',fontsize=9.5,color='#536377')
fig.text(.065,.029,'¹ Input tokens ÷ (queue + prefill time), not isolated GPU prefill.  ² Total output ÷ batch wall time, including queueing and prefill.',fontsize=9.5,color='#536377')
out=base/'charts'
fig.savefig(out/'tensorfold-studio-benchmark.svg',dpi=180,facecolor='white')
print(out/'tensorfold-studio-benchmark.svg')
