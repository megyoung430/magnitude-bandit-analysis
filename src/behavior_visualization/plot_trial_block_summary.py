"""Six cohort plots of observed session and block trial counts."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.behavior_visualization.plot_style import MOUSE_COLORS


def plot_trial_block_summaries(sessions, blocks, save_dir, show=True, split_subject_groups=True):
    """Save PNG/PDF figures plus source tables and equal-mouse problem summaries.

    Distribution panels show individual observations per mouse and pooled group
    ECDFs. Time plots show each mouse (daily means when multiple observations
    share a date). Problem plots average within mouse first, then show mean ± SEM
    across mice; n is the number of mice. Blocks include observed edge portions.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tables = {'session': sessions, 'block': blocks}
    for unit, table in tables.items():
        if table.empty:
            raise ValueError(f'No {unit} observations available for the selected problems')
    paths = []
    mice = sorted(set(sessions.mouse) | set(blocks.mouse),
                  key=lambda m: int(sessions.loc[sessions.mouse == m, 'subject_number'].iloc[0]))
    colors = {m: MOUSE_COLORS[i % len(MOUSE_COLORS)] for i,m in enumerate(mice)}

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            path = save_dir / f'{name}.{extension}'
            fig.savefig(path, dpi=180, bbox_inches='tight')
            paths.append(path)
        if show:
            plt.show()
        plt.close(fig)

    for unit, table in tables.items():
        if table.empty:
            raise ValueError(f'No {unit} observations available')
        table.to_csv(save_dir / f'{unit}_trial_counts.csv', index=False)
        title = f'Trials per {unit}'
        note = 'Blocks include observed portions at problem edges.' if unit == 'block' else 'Each point is one session.'
        groups = ('sub-01–04', 'sub-05–08') if split_subject_groups else ('All mice',)
        fig, axes = plt.subplots(1, 1 + len(groups), figsize=(17 if split_subject_groups else 13, 5),
                                 gridspec_kw={'width_ratios': [2] + [1] * len(groups)})
        arrays = [table.loc[table.mouse == m, 'trials'].to_numpy() for m in mice]
        axes[0].boxplot(arrays, showfliers=False)
        rng = np.random.default_rng(0)
        labels = []
        for i,(mouse,values) in enumerate(zip(mice,arrays),1):
            axes[0].scatter(i+rng.uniform(-.15,.15,len(values)),values,s=10,alpha=.4,color=colors[mouse])
            number = int(table.loc[table.mouse == mouse,'subject_number'].iloc[0])
            labels.append(f'sub-{number:02d}\n{mouse}\nn={len(values)}')
        axes[0].set_xticks(range(1,len(mice)+1),labels,rotation=45,ha='right',fontsize=9)
        axes[0].set_ylabel(title)
        axes[0].set_title('Distribution by mouse')
        for ax,group in zip(axes[1:],groups):
            group_table = table[table.group == group] if split_subject_groups else table
            for mouse, rows in group_table.groupby('mouse'):
                x = np.sort(rows.trials.to_numpy())
                ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=colors[mouse],alpha=.65,label=mouse)
            x = np.sort(group_table.trials.to_numpy())
            if len(x):
                ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color='black',lw=2,label='Pooled observations')
                ax.legend(fontsize=8)
            ax.set(title=f'{group} (n={len(x)})',xlabel=title,ylabel='Cumulative fraction',ylim=(0,1.03))
        fig.suptitle(f'{title}: distributions\n{note}',fontsize=13)
        save(fig,f'{title} - distributions')

        fig,ax = plt.subplots(figsize=(12,5))
        daily = table.groupby(['mouse','date'],as_index=False).trials.mean()
        for mouse,rows in daily.groupby('mouse'):
            ax.plot(rows.date,rows.trials,'.-',color=colors[mouse],alpha=.8,label=mouse)
        ax.set(xlabel='Calendar date' if unit == 'session' else 'Block end date (last observed trial)',
               ylabel=f'Mean trials per {unit} on that date',title=f'{title} over time')
        ax.legend(fontsize=9,ncol=4)
        fig.autofmt_xdate()
        save(fig,f'{title} - over time')

        per_mouse = table.groupby(['problem','mouse'],as_index=False).trials.mean()
        summary = per_mouse.groupby('problem').trials.agg(['mean','sem','count']).rename(columns={'count':'n_mice'})
        summary.to_csv(save_dir / f'{unit}_trials_by_problem_summary.csv')
        per_mouse.to_csv(save_dir / f'{unit}_trials_by_problem_per_mouse.csv',index=False)
        fig,ax = plt.subplots(figsize=(9,5))
        for mouse,rows in per_mouse.groupby('mouse'):
            ax.plot(rows.problem,rows.trials,'o-',color=colors[mouse],alpha=.5,label=mouse)
        ax.errorbar(summary.index,summary['mean'],yerr=summary['sem'].fillna(0),fmt='o-',color='black',lw=2.5,capsize=4,label='Across mice: mean ± SEM')
        ax.set_xticks(summary.index,[f'{p}\n(n={int(row.n_mice)})' for p,row in summary.iterrows()])
        ax.set(xlabel='Problem (number of mice)',ylabel=f'Mean trials per {unit}',title=f'{title} across problems — equal weight per mouse')
        ax.legend(fontsize=8,ncol=3)
        save(fig,f'{title} - across problems')
    return paths
