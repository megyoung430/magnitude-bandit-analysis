"""Notebook viewer for matched, held-out neural-model session trajectories."""
from pathlib import Path

import numpy as np

from src.agents.neural_bandit import NEURAL_MODELS, DISPLAY_NAMES


def held_out_sessions(results, sequences):
    """Pair each eligible session with its completed test-fold result only."""
    folds = results.get('folds', [results])
    eligible = []
    for sequence in sequences:
        matches = []
        for fold in folds:
            if not fold.get('complete'):
                continue
            if sequence['mouse'] not in fold['split_mice']['test']:
                continue
            problems = fold.get('split_problems')
            if problems is not None and int(sequence['problem']) not in problems['test']:
                continue
            if not all(any(row['model'] == model for row in fold['selected']) for model in NEURAL_MODELS):
                continue
            matches.append(fold)
        if len(matches) > 1:
            raise ValueError('Session belongs to multiple test folds; cannot choose weights unambiguously.')
        if matches:
            eligible.append((sequence, matches[0]))
    return eligible


def replay_numpy(parameters, sequence, model):
    """Replay saved weights on CPU without JAX compilation (plotting only)."""
    p = {k: np.asarray(v, dtype=float) for k, v in parameters.items()}
    k = len(sequence['towers'])
    vanilla = model == 'vanilla_rnn'
    h = p['joint_b1' if vanilla else 'reward_b1'].size
    q = np.full(k, 0. if vanilla else float(p['q_init']))
    c = np.full(k, 0. if vanilla else float(p['c_init']))
    sr, sa, joint = np.zeros(h), np.zeros(h), np.zeros(h)
    records = []
    for action, reward in zip(sequence['actions'], sequence['rewards']):
        logits = joint @ p['joint_w2'] + p['joint_b2'] if vanilla else q+c
        shifted = logits - np.max(logits)
        row = dict(logits=logits.copy(), log_probabilities=shifted-np.log(np.exp(shifted).sum()),
                   pre_q=q.copy(), pre_c=c.copy(), pre_reward_hidden=sr.copy(),
                   pre_action_hidden=sa.copy(), pre_joint_hidden=joint.copy())
        onehot = np.eye(k)[int(action)]
        if vanilla:
            joint = np.tanh(np.concatenate([onehot, [reward], joint]) @ p['joint_w1']+p['joint_b1'])
        else:
            if model == 'rl_ann':
                ri, ai = np.array([q[action], reward]), onehot
            elif model == 'context_ann':
                ri, ai = np.concatenate([[q[action], reward], q]), np.concatenate([onehot, c])
            elif model == 'memory_ann':
                ri, ai = np.concatenate([[0., reward], sr]), np.concatenate([onehot, sa])
            else:
                raise ValueError(f'Unknown model: {model}')
            sr = np.tanh(ri @ p['reward_w1']+p['reward_b1'])
            sa = np.tanh(ai @ p['action_w1']+p['action_b1'])
            update = (sr @ p['reward_w2']+p['reward_b2']).item()
            forgetting = np.exp(-np.logaddexp(0., -p['forgetting_raw']))
            q = (1-forgetting)*q+forgetting*p['q_init']
            if model == 'memory_ann':
                q[action] = update
            else:
                q[action] += update
            c = sa @ p['action_w2']+p['action_b2']
        row.update(post_q=q.copy(), post_c=c.copy())
        records.append(row)
    return {key: np.stack([row[key] for row in records]) for key in records[0]}


def session_traces(sequence, fold):
    """Load each architecture's selected weights and replay the observed history."""
    traces = {}
    for model in NEURAL_MODELS:
        selected = next(row for row in fold['selected'] if row['model'] == model)
        path = Path(fold['output_dir']) / selected['parameters_file']
        with np.load(path, allow_pickle=False) as saved:
            parameters = {key: saved[key] for key in saved.files}
        traces[model] = replay_numpy(parameters, sequence, model)
    return traces


def plot_trajectories(sequence, traces, metric='probability', trial_range=None):
    """Four model panels plus observed actions/rewards, with consistent arm colors."""
    import matplotlib.pyplot as plt

    keys = dict(probability='log_probabilities', logits='logits', q='pre_q', c='pre_c')
    labels = dict(probability='Choice probability', logits='Choice logit',
                  q='Reward channel Q', c='Action channel c')
    if metric not in keys:
        raise ValueError(f'Unknown trajectory metric: {metric}')
    total = len(sequence['actions'])
    first, last = trial_range or (1, total)
    if not 1 <= first <= last <= total:
        raise ValueError('Trial range must lie within the session (one-based, inclusive).')
    subset = slice(first-1, last)
    trials = np.arange(first, last+1)
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 2, 2, 1, 1]},
                             layout='constrained')
    for ax, model in zip(axes[:4], NEURAL_MODELS):
        ax.set_title(DISPLAY_NAMES[model], loc='left', fontsize=11)
        ax.set_ylabel(labels[metric])
        if model == 'vanilla_rnn' and metric in ('q', 'c'):
            ax.text(.5, .5, 'Vanilla RNN has no separate Q / c channels',
                    ha='center', va='center', transform=ax.transAxes)
            continue
        values = traces[model][keys[metric]][subset]
        if metric == 'probability':
            values = np.exp(values)
            ax.set_ylim(0, 1)
        for arm, tower in enumerate(sequence['towers']):
            ax.plot(trials, values[:, arm], color=colors[arm], label=str(tower), linewidth=1)
        ax.legend(title='Tower', ncol=len(sequence['towers']), fontsize=8, loc='upper right')
    actions = np.asarray(sequence['actions'])[subset]
    axes[4].scatter(trials, actions, c=[colors[int(a)] for a in actions], s=10)
    axes[4].set_yticks(range(len(sequence['towers'])), [str(t) for t in sequence['towers']])
    axes[4].set_ylabel('Observed choice')
    axes[5].plot(trials, np.asarray(sequence['rewards'])[subset], color='black', linewidth=.8)
    axes[5].set(xlabel='Trial (1-based)', ylabel='Reward\n(pulses)')
    axes[5].set_xlim(first-.5, last+.5)
    fig.suptitle(f"{sequence['mouse']} · {sequence['session']} · problem {sequence['problem']}\n"
                 'Pre-choice predictions from observed past actions and rewards')
    return fig


def trajectory_viewer(results, sequences):
    """Return linked subject/session dropdowns and a held-out trajectory plot."""
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError('Install ipywidgets in the notebook kernel: %pip install ipywidgets') from exc
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt

    eligible = held_out_sessions(results, sequences)
    if not eligible:
        raise ValueError('No held-out sessions with completed fits for all four architectures. '
                         'Complete at least one fold before opening the viewer.')
    subject = widgets.Dropdown(options=sorted({s['mouse'] for s, _ in eligible}), description='Subject:')
    session = widgets.Dropdown(description='Session:', layout=widgets.Layout(width='650px'))
    metric = widgets.Dropdown(options=[('Choice probabilities', 'probability'), ('Choice logits', 'logits'),
                                       ('Reward channel Q', 'q'), ('Action channel c', 'c')],
                              description='Trajectory:', layout=widgets.Layout(width='350px'))
    window = widgets.IntRangeSlider(value=(1, 2), min=1, max=2, description='Trials:',
                                     continuous_update=False, layout=widgets.Layout(width='650px'))
    output = widgets.Output()
    state = dict(updating=False, index=None, traces=None)

    def draw(change=None):
        if state['updating'] or session.value is None:
            return
        with output:
            clear_output(wait=True)
            try:
                index = session.value
                sequence, fold = eligible[index]
                if state['index'] != index:
                    state['traces'] = session_traces(sequence, fold)
                    state['index'] = index
                problems = fold.get('split_problems')
                if problems:
                    print(f"Held-out problem {problems['test']}; validation {problems['validation']}; "
                          f"pooled training {problems['train']}.")
                else:
                    print('Held-out mouse: this subject was excluded from training and validation.')
                fig = plot_trajectories(sequence, state['traces'], metric.value, window.value)
                display(fig)
                plt.close(fig)
            except Exception as exc:
                print(f'{type(exc).__name__}: {exc}')

    def change_session(change=None):
        if state['updating'] or session.value is None:
            return
        state['updating'] = True
        total = len(eligible[session.value][0]['actions'])
        window.value = (1, 1)
        window.max = total
        window.value = (1, total)
        state['updating'] = False
        draw()

    def change_subject(change=None):
        state['updating'] = True
        session.options = [(f"{s['session']} · problem {s['problem']}", i)
                           for i, (s, _) in enumerate(eligible) if s['mouse'] == subject.value]
        session.value = session.options[0][1]
        state['updating'] = False
        change_session()

    subject.observe(change_subject, names='value')
    session.observe(change_session, names='value')
    metric.observe(draw, names='value')
    window.observe(draw, names='value')
    change_subject()
    return widgets.VBox([subject, session, widgets.HBox([metric]), window, output])
