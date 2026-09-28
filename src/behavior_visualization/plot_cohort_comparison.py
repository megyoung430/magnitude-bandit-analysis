"""Matched-axis descriptive comparisons of two cohorts (mice are independent)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.behavior_analysis.trial_block_summary import build_trial_block_tables
from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_rank_counts_by_good_reversal import get_rank_counts_by_good_reversal
from src.behavior_analysis.get_total_reversals import get_total_reversals
from src.behavior_analysis.matched_trial_budget import match_trial_budget
from src.behavior_visualization.plot_style import MOUSE_COLORS


def plot_cohort_comparison(cohorts, subject_numbers, output_dir, problems=(1,2,3), trial_budget=510, show=False):
    """Save full-data comparisons, plus a separate capped problem-2 comparison.

    Columns are cohorts, rows are problems; paired panels share axes. Mouse
    means receive equal weight. Distributions pool observations within cohort;
    faint ECDFs show individual mice. Time is days since each mouse's first
    observed session in that problem. Block lengths include observed edges.
    Rank proportions use the existing good-block quality filters and pool
    retained trials within mouse. No between-cohort significance test is implied.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    labels = list(cohorts)
    if len(labels)!=2:raise ValueError('Exactly two cohorts are required')
    tables = {}
    paths = []
    ranks = ('best','second','third')
    for label,data in cohorts.items():
        missing=set(problems)-set(data)
        if missing:raise ValueError(f'{label}: missing problems {sorted(missing)}')
        selected={p:data[p] for p in problems}
        sessions,blocks=build_trial_block_tables(selected,subject_numbers[label])
        tables[label]={'session':sessions,'block':blocks}
        for unit,table in tables[label].items():table.to_csv(output_dir/f'{label}_{unit}_trials.csv',index=False)

    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=output_dir/f'{name}.{ext}'
            fig.savefig(path,dpi=180,bbox_inches='tight');paths.append(path)
        if show:plt.show()
        plt.close(fig)

    def grid(rows,title):
        fig,axes=plt.subplots(len(rows),2,figsize=(13,4*len(rows)),sharex='row',sharey='row',squeeze=False)
        fig.suptitle(title,fontsize=16)
        for i,p in enumerate(rows):
            for j,label in enumerate(labels):axes[i,j].set_title(f'{label} — problem {p}')
        return fig,axes

    for unit in ('session','block'):
        fig,axes=grid(problems,f'Trials per {unit}: distributions (full data)')
        for i,p in enumerate(problems):
            for j,label in enumerate(labels):
                ax=axes[i,j];t=tables[label][unit];t=t[t.problem==p]
                for k,(mouse,rows) in enumerate(t.groupby('mouse')):
                    x=np.sort(rows.trials.to_numpy())
                    ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',alpha=.35,color=MOUSE_COLORS[k%len(MOUSE_COLORS)])
                x=np.sort(t.trials.to_numpy())
                ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color='black',lw=2,
                        label=f'{t.mouse.nunique()} mice; {len(t)} {unit}s')
                ax.set(xlabel=f'Trials per {unit}',ylabel='Cumulative fraction',ylim=(0,1.03));ax.legend(fontsize=9)
        save(fig,f'Trials per {unit} - distributions')
        fig,axes=grid(problems,f'Trials per {unit}: over time (full data)')
        for i,p in enumerate(problems):
            for j,label in enumerate(labels):
                ax=axes[i,j];t=tables[label][unit];t=t[t.problem==p]
                for k,(mouse,rows) in enumerate(t.groupby('mouse')):
                    origin=tables[label]['session'].query('problem == @p and mouse == @mouse').date.min()
                    daily=rows.groupby('date').trials.mean()
                    ax.plot((daily.index-origin).days,daily.values,'.-',alpha=.8,label=mouse,color=MOUSE_COLORS[k%len(MOUSE_COLORS)])
                ax.set(xlabel='Days since first observed session in this problem',ylabel=f'Daily mean trials per {unit}')
                ax.legend(fontsize=8,ncol=2)
        save(fig,f'Trials per {unit} - over time')
        fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True,sharey=True)
        for j,label in enumerate(labels):
            ax=axes[j];per_mouse=tables[label][unit].groupby(['problem','mouse']).trials.mean().reset_index()
            per_mouse.to_csv(output_dir/f'{label}_{unit}_means_by_mouse.csv',index=False)
            for k,(mouse,rows) in enumerate(per_mouse.groupby('mouse')):
                ax.plot(rows.problem,rows.trials,'o-',alpha=.45,label=mouse,color=MOUSE_COLORS[k%len(MOUSE_COLORS)])
            means=per_mouse.groupby('problem').trials.agg(['mean','sem','count'])
            ax.errorbar(means.index,means['mean'],yerr=means['sem'].fillna(0),color='black',fmt='o-',capsize=4,lw=2,label='Mean ± SEM across mice')
            ax.set_xticks(list(problems),[f'{p}\n(n={int(means.loc[p,"count"])})' for p in problems])
            ax.set(title=label,xlabel='Problem (number of mice)',ylabel=f'Mean trials per {unit}');ax.legend(fontsize=8,ncol=2)
        fig.suptitle(f'Trials per {unit}: across problems (equal weight per mouse)')
        save(fig,f'Trials per {unit} - across problems')

    def rank_and_reversal_panels(datasets,selected_problems,suffix):
        rank_fig,rank_axes=grid(selected_problems,f'Pooled true-value rank proportions — {suffix}')
        rev_fig,rev_axes=grid(selected_problems,f'Total reversals per mouse — {suffix}')
        rank_records,rev_records=[],[]
        for i,p in enumerate(selected_problems):
            for j,label in enumerate(labels):
                subjects=datasets[label][p]
                windows=get_good_reversal_info(subjects,include_first_block=True)
                counts=get_rank_counts_by_good_reversal(windows)
                rank_values=[];rev_values=[]
                for k,mouse in enumerate(sorted(subjects)):
                    rows=counts[mouse];n=sum(r['total'] for r in rows)
                    color=MOUSE_COLORS[k%len(MOUSE_COLORS)]
                    if n:
                        values=[sum(r[rank] for r in rows)/n for rank in ranks]
                        rank_values.append(values)
                        rank_records.append(dict(cohort=label,problem=p,mouse=mouse,eligible_trials=n,**dict(zip(ranks,values))))
                        rank_axes[i,j].plot(range(3),values,'o-',alpha=.65,color=color,label=mouse)
                    totals=get_total_reversals(subjects[mouse])
                    values=[totals.get('good_reversals',np.nan),totals.get('bad_reversals',np.nan),totals['total_reversals']]
                    rev_values.append(values)
                    rev_records.append(dict(cohort=label,problem=p,mouse=mouse,**totals))
                    rev_axes[i,j].plot(range(3),values,'o-',alpha=.65,color=color,label=mouse)
                for ax,values,names,ylabel in ((rank_axes[i,j],rank_values,('Best','Second','Third'),'Proportion of choices'),
                                               (rev_axes[i,j],rev_values,('Good','Bad','All'),'Number of reversals')):
                    arr=np.asarray(values,dtype=float)
                    if len(arr):
                        mean=np.nanmean(arr,axis=0)
                        sem=np.nanstd(arr,axis=0,ddof=1)/np.sqrt(np.sum(np.isfinite(arr),axis=0)) if len(arr)>1 else np.zeros(3)
                        ax.bar(range(3),mean,color='gray',alpha=.2,zorder=0)
                        ax.errorbar(range(3),mean,yerr=sem,fmt='ko',capsize=4,label='Mean ± SEM')
                        if ax is rank_axes[i,j]:
                            for x, value, error in zip(range(3), mean, sem):
                                if np.isfinite(value):
                                    ax.text(x, value + (error if np.isfinite(error) else 0) + 0.035,
                                            f"{value:.2f}", ha='center', va='bottom', fontsize=11)
                    ax.set_xticks(range(3),names);ax.set_ylabel(ylabel);ax.legend(fontsize=8,ncol=2)
                    ax.set_title(f'{label} — problem {p} (n={len(values)} mice)')
                rank_axes[i,j].set_ylim(0,1)
        pd.DataFrame(rank_records).to_csv(output_dir/f'Rank proportions - {suffix}.csv',index=False)
        pd.DataFrame(rev_records).to_csv(output_dir/f'Reversal totals - {suffix}.csv',index=False)
        save(rank_fig,f'Rank proportions - {suffix}')
        save(rev_fig,f'Reversal totals - {suffix}')
    rank_and_reversal_panels(cohorts,problems,'full data')
    if 2 in problems and trial_budget is not None:
        capped={};reports=[]
        for label in labels:
            data,report=match_trial_budget(cohorts[label][2],subject_numbers[label],trial_budget=trial_budget)
            capped[label]={2:data}
            reports.extend(dict(cohort=label,**row) for row in report['mice'])
        pd.DataFrame(reports).to_csv(output_dir/'Problem 2 trial cap.csv',index=False)
        rank_and_reversal_panels(capped,(2,),f'problem 2 first {trial_budget} trials (or all available)')
    return paths
