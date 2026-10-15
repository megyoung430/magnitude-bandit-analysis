"""Notebook viewer for matched, held-out neural-model session trajectories."""

from pathlib import Path

import numpy as np

from src.agents.neural_bandit import DISPLAY_NAMES, NEURAL_MODELS


def held_out_sessions(results, sequences):
    """Pair each eligible session with its completed test-fold result only."""
    folds = results.get("folds", [results])
    eligible = []
    for sequence in sequences:
        matches = []
        for fold in folds:
            if not fold.get("complete"):
                continue
            if sequence["mouse"] not in fold["split_mice"]["test"]:
                continue
            problems = fold.get("split_problems")
            if problems is not None and int(sequence["problem"]) not in problems["test"]:
                continue
            if not all(
                any(row["model"] == model for row in fold["selected"]) for model in NEURAL_MODELS
            ):
                continue
            matches.append(fold)
        if len(matches) > 1:
            raise ValueError(
                "Session belongs to multiple test folds; cannot choose weights unambiguously."
            )
        if matches:
            eligible.append((sequence, matches[0]))
    return eligible


def replay_numpy(parameters, sequence, model):
    """Replay saved weights on CPU without JAX compilation (plotting only)."""
    p = {k: np.asarray(v, dtype=float) for k, v in parameters.items()}
    k = len(sequence["towers"])
    vanilla = model == "vanilla_rnn"
    h = p["joint_b1" if vanilla else "reward_b1"].size
    q = np.full(k, 0.0 if vanilla else float(p["q_init"]))
    c = np.full(k, 0.0 if vanilla else float(p["c_init"]))
    sr, sa, joint = np.zeros(h), np.zeros(h), np.zeros(h)
    records = []
    # Known issue: zip truncates silently if actions and rewards differ in length.
    for action, reward in zip(sequence["actions"], sequence["rewards"], strict=False):
        logits = joint @ p["joint_w2"] + p["joint_b2"] if vanilla else q + c
        shifted = logits - np.max(logits)
        row = dict(
            logits=logits.copy(),
            log_probabilities=shifted - np.log(np.exp(shifted).sum()),
            pre_q=q.copy(),
            pre_c=c.copy(),
            pre_reward_hidden=sr.copy(),
            pre_action_hidden=sa.copy(),
            pre_joint_hidden=joint.copy(),
        )
        onehot = np.eye(k)[int(action)]
        if vanilla:
            joint = np.tanh(
                np.concatenate([onehot, [reward], joint]) @ p["joint_w1"] + p["joint_b1"]
            )
        else:
            if model == "rl_ann":
                ri, ai = np.array([q[action], reward]), onehot
            elif model == "context_ann":
                ri, ai = np.concatenate([[q[action], reward], q]), np.concatenate([onehot, c])
            elif model == "memory_ann":
                ri, ai = np.concatenate([[0.0, reward], sr]), np.concatenate([onehot, sa])
            else:
                raise ValueError(f"Unknown model: {model}")
            sr = np.tanh(ri @ p["reward_w1"] + p["reward_b1"])
            sa = np.tanh(ai @ p["action_w1"] + p["action_b1"])
            update = (sr @ p["reward_w2"] + p["reward_b2"]).item()
            forgetting = np.exp(-np.logaddexp(0.0, -p["forgetting_raw"]))
            q = (1 - forgetting) * q + forgetting * p["q_init"]
            if model == "memory_ann":
                q[action] = update
            else:
                q[action] += update
            c = sa @ p["action_w2"] + p["action_b2"]
        row.update(post_q=q.copy(), post_c=c.copy())
        records.append(row)
    return {key: np.stack([row[key] for row in records]) for key in records[0]}


def session_traces(sequence, fold):
    """Load each architecture's selected weights and replay the observed history."""
    traces = {}
    for model in NEURAL_MODELS:
        selected = next(row for row in fold["selected"] if row["model"] == model)
        path = Path(fold["output_dir"]) / selected["parameters_file"]
        with np.load(path, allow_pickle=False) as saved:
            parameters = {key: saved[key] for key in saved.files}
        traces[model] = replay_numpy(parameters, sequence, model)
    return traces


def plot_trajectories(sequence, traces, metric="probability", trial_range=None):
    """Four model panels plus observed actions/rewards, with consistent arm colors."""
    import matplotlib.pyplot as plt

    keys = dict(probability="log_probabilities", logits="logits", q="pre_q", c="pre_c")
    labels = dict(
        probability="Choice probability",
        logits="Choice logit",
        q="Reward channel Q",
        c="Action channel c",
    )
    if metric not in keys:
        raise ValueError(f"Unknown trajectory metric: {metric}")
    total = len(sequence["actions"])
    first, last = trial_range or (1, total)
    if not 1 <= first <= last <= total:
        raise ValueError("Trial range must lie within the session (one-based, inclusive).")
    subset = slice(first - 1, last)
    trials = np.arange(first, last + 1)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(13, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 2, 2, 1, 1]},
        layout="constrained",
    )
    for ax, model in zip(axes[:4], NEURAL_MODELS, strict=True):
        ax.set_title(DISPLAY_NAMES[model], loc="left", fontsize=11)
        ax.set_ylabel(labels[metric])
        if model == "vanilla_rnn" and metric in ("q", "c"):
            ax.text(
                0.5,
                0.5,
                "Vanilla RNN has no separate Q / c channels",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        values = traces[model][keys[metric]][subset]
        if metric == "probability":
            values = np.exp(values)
            ax.set_ylim(0, 1)
        for arm, tower in enumerate(sequence["towers"]):
            ax.plot(trials, values[:, arm], color=colors[arm], label=str(tower), linewidth=1)
        ax.legend(title="Tower", ncol=len(sequence["towers"]), fontsize=8, loc="upper right")
    actions = np.asarray(sequence["actions"])[subset]
    axes[4].scatter(trials, actions, c=[colors[int(a)] for a in actions], s=10)
    axes[4].set_yticks(range(len(sequence["towers"])), [str(t) for t in sequence["towers"]])
    axes[4].set_ylabel("Observed choice")
    axes[5].plot(trials, np.asarray(sequence["rewards"])[subset], color="black", linewidth=0.8)
    axes[5].set(xlabel="Trial (1-based)", ylabel="Reward\n(pulses)")
    axes[5].set_xlim(first - 0.5, last + 0.5)
    fig.suptitle(
        f"{sequence['mouse']} · {sequence['session']} · problem {sequence['problem']}\n"
        "Pre-choice predictions from observed past actions and rewards"
    )
    return fig


def _import_ipywidgets():
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError(
            "Install ipywidgets in the notebook kernel: %pip install ipywidgets"
        ) from exc
    return widgets


def _build_controls(widgets, eligible: list) -> dict:
    """The four dropdown/slider controls and the output area, with their initial options."""
    metric_options = [
        ("Choice probabilities", "probability"),
        ("Choice logits", "logits"),
        ("Reward channel Q", "q"),
        ("Action channel c", "c"),
    ]
    return dict(
        subject=widgets.Dropdown(
            options=sorted({s["mouse"] for s, _ in eligible}), description="Subject:"
        ),
        session=widgets.Dropdown(description="Session:", layout=widgets.Layout(width="650px")),
        metric=widgets.Dropdown(
            options=metric_options,
            description="Trajectory:",
            layout=widgets.Layout(width="350px"),
        ),
        window=widgets.IntRangeSlider(
            value=(1, 2),
            min=1,
            max=2,
            description="Trials:",
            continuous_update=False,
            layout=widgets.Layout(width="650px"),
        ),
        output=widgets.Output(),
    )


def _split_description(fold: dict) -> str:
    """One line saying which data the fold held out."""
    problems = fold.get("split_problems")
    if problems:
        return (
            f"Held-out problem {problems['test']}; validation {problems['validation']}; "
            f"pooled training {problems['train']}."
        )
    return "Held-out mouse: this subject was excluded from training and validation."


class _TrajectoryViewer:
    """Wires the controls to the plot: subject -> session -> window, and redraw on change."""

    def __init__(self, eligible: list, controls: dict, *, display, clear_output, plt):
        self.eligible = eligible
        self.subject, self.session = controls["subject"], controls["session"]
        self.metric, self.window = controls["metric"], controls["window"]
        self.output = controls["output"]
        self._display, self._clear_output, self._plt = display, clear_output, plt
        # _updating blocks redraws while a handler is still rewriting the other controls;
        # the replayed traces are cached per session index.
        self._updating = False
        self._traces_index = None
        self._traces = None
        self.subject.observe(self.change_subject, names="value")
        self.session.observe(self.change_session, names="value")
        self.metric.observe(self.draw, names="value")
        self.window.observe(self.draw, names="value")

    def _show_selected_session(self) -> None:
        index = self.session.value
        sequence, fold = self.eligible[index]
        if self._traces_index != index:
            self._traces = session_traces(sequence, fold)
            self._traces_index = index
        print(_split_description(fold))
        fig = plot_trajectories(sequence, self._traces, self.metric.value, self.window.value)
        self._display(fig)
        self._plt.close(fig)

    def draw(self, change=None) -> None:
        if self._updating or self.session.value is None:
            return
        with self.output:
            self._clear_output(wait=True)
            # Exceptions inside a widget callback never reach the notebook cell, so they
            # are printed into the output area instead.
            # Known issue: the broad except also catches the first draw (which runs in the
            # cell, where errors would show normally) and turns programming errors into
            # one line of text with no traceback.
            try:
                self._show_selected_session()
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}")

    def change_session(self, change=None) -> None:
        if self._updating or self.session.value is None:
            return
        # Known issue: _updating is reset without try/finally, so an exception in between
        # (for example window.max = 0 for an empty session) leaves it True and every later
        # redraw returns early. The same holds in change_subject.
        self._updating = True
        total = len(self.eligible[self.session.value][0]["actions"])
        self.window.value = (1, 1)
        self.window.max = total
        self.window.value = (1, total)
        self._updating = False
        self.draw()

    def change_subject(self, change=None) -> None:
        self._updating = True
        self.session.options = [
            (f"{s['session']} · problem {s['problem']}", i)
            for i, (s, _) in enumerate(self.eligible)
            if s["mouse"] == self.subject.value
        ]
        self.session.value = self.session.options[0][1]
        self._updating = False
        self.change_session()


def trajectory_viewer(results, sequences):
    """Return linked subject/session dropdowns and a held-out trajectory plot."""
    widgets = _import_ipywidgets()
    # IPython is imported first so a missing IPython is reported before a missing matplotlib.
    from IPython.display import clear_output, display  # noqa: I001
    import matplotlib.pyplot as plt

    eligible = held_out_sessions(results, sequences)
    if not eligible:
        raise ValueError(
            "No held-out sessions with completed fits for all four architectures. "
            "Complete at least one fold before opening the viewer."
        )
    controls = _build_controls(widgets, eligible)
    viewer = _TrajectoryViewer(
        eligible, controls, display=display, clear_output=clear_output, plt=plt
    )
    viewer.change_subject()
    return widgets.VBox(
        [
            controls["subject"],
            controls["session"],
            widgets.HBox([controls["metric"]]),
            controls["window"],
            controls["output"],
        ]
    )
