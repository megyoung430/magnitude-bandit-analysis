"""Behaviour of the held-out trajectory viewer, driven through its public functions.

Weights are small random arrays saved to a temporary directory; display and
matplotlib.close are replaced by recorders so each callback's drawing can be inspected.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.agents import trajectory_viewer as tv  # noqa: E402
from src.agents.neural_bandit import NEURAL_MODELS  # noqa: E402

pytest.importorskip("ipywidgets")
K, H = 3, 4
TOWERS = ["A1", "B1", "C1"]


def random_weights(model, rng):
    def r(*shape):
        return rng.normal(0, 0.5, shape)

    if model == "vanilla_rnn":
        return dict(joint_w1=r(K + 1 + H, H), joint_b1=r(H), joint_w2=r(H, K), joint_b2=r(K))
    reward_in, action_in = {
        "rl_ann": (2, K),
        "context_ann": (2 + K, 2 * K),
        "memory_ann": (2 + H, K + H),
    }[model]
    return dict(
        reward_w1=r(reward_in, H), reward_b1=r(H), reward_w2=r(H, 1), reward_b2=r(1),
        action_w1=r(action_in, H), action_b1=r(H), action_w2=r(H, K), action_b2=r(K),
        forgetting_raw=np.array(0.3), q_init=np.array(0.2), c_init=np.array(-0.1),
    )  # fmt: skip


def make_fold(directory, mice, problems=None, complete=True, seed=0):
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    for model in NEURAL_MODELS:
        np.savez(directory / f"{model}.npz", **random_weights(model, rng))
    fold = dict(
        complete=complete,
        split_mice=dict(test=mice),
        selected=[dict(model=m, parameters_file=f"{m}.npz") for m in NEURAL_MODELS],
        output_dir=str(directory),
    )
    if problems is not None:
        fold["split_problems"] = problems
    return fold


def make_sequence(mouse, problem, session, n, seed):
    rng = np.random.default_rng(seed)
    return dict(
        mouse=mouse,
        problem=problem,
        session=session,
        towers=list(TOWERS),
        actions=rng.integers(0, K, n),
        rewards=rng.integers(0, 5, n).astype(float),
    )


@pytest.fixture
def results(tmp_path):
    problems = dict(test=[5], validation=[6], train=[7, 8])
    return dict(
        folds=[
            make_fold(tmp_path / "f1", ["m1"], problems, seed=1),
            make_fold(tmp_path / "f2", ["m2"], seed=2),  # held-out mouse, no problem split
            make_fold(tmp_path / "f3", ["m3"], problems, complete=False, seed=3),
        ]
    )


@pytest.fixture
def sequences():
    return [
        make_sequence("m1", 5, "ses-b", 12, 10),
        make_sequence("m1", 5, "ses-a", 8, 11),
        make_sequence("m1", 6, "ses-c", 9, 12),  # problem 6 is not in the test split
        make_sequence("m2", 5, "ses-d", 10, 13),
        make_sequence("m3", 5, "ses-e", 7, 14),  # its fold is incomplete
    ]


class Screen:
    """Records what the callbacks display and close."""

    def __init__(self, monkeypatch):
        self.figures, self.cleared = [], 0
        monkeypatch.setattr("IPython.display.display", self.figures.append)
        monkeypatch.setattr("IPython.display.clear_output", self._clear)
        real_close = plt.close
        monkeypatch.setattr(
            plt, "close", lambda fig=None: (self.closed.append(fig), real_close(fig))
        )
        self.closed = []

    def _clear(self, *args, **kwargs):
        self.cleared += 1
        assert kwargs == {"wait": True}


@pytest.fixture
def screen(monkeypatch):
    return Screen(monkeypatch)


def controls(box):
    subject, session, metric_row, window, output = box.children
    return subject, session, metric_row.children[0], window, output


def test_held_out_sessions_pairs_each_session_with_its_completed_test_fold(results, sequences):
    pairs = tv.held_out_sessions(results, sequences)
    assert [(s["session"], f["split_mice"]["test"]) for s, f in pairs] == [
        ("ses-b", ["m1"]),
        ("ses-a", ["m1"]),
        ("ses-d", ["m2"]),
    ]
    single = tv.held_out_sessions(results["folds"][0], sequences)
    assert [s["session"] for s, _ in single] == ["ses-b", "ses-a"]


def test_session_in_two_test_folds_is_ambiguous(results, sequences):
    results["folds"].append(dict(results["folds"][0]))
    with pytest.raises(ValueError, match="multiple test folds"):
        tv.held_out_sessions(results, sequences)


def test_fold_missing_an_architecture_is_not_eligible(results, sequences):
    results["folds"][0]["selected"] = results["folds"][0]["selected"][:3]
    assert [s["mouse"] for s, _ in tv.held_out_sessions(results, sequences)] == ["m2"]


def test_replay_matches_hand_computed_first_step_and_is_causal():
    rng = np.random.default_rng(0)
    sequence = make_sequence("m", 5, "s", 6, 1)
    for model in NEURAL_MODELS:
        weights = random_weights(model, rng)
        trace = tv.replay_numpy(weights, sequence, model)
        assert trace["logits"].shape == (6, K)
        np.testing.assert_allclose(np.exp(trace["log_probabilities"]).sum(axis=1), 1)
        changed = dict(sequence, rewards=sequence["rewards"].copy())
        changed["rewards"][-1] += 7
        other = tv.replay_numpy(weights, changed, model)
        for key in ("logits", "pre_q", "pre_c", "log_probabilities"):
            np.testing.assert_array_equal(trace[key][:-1], other[key][:-1])
    rl = tv.replay_numpy(random_weights("rl_ann", rng), sequence, "rl_ann")
    np.testing.assert_array_equal(rl["pre_q"][0], np.full(K, 0.2))
    np.testing.assert_array_equal(rl["pre_c"][0], np.full(K, -0.1))
    np.testing.assert_allclose(rl["logits"], rl["pre_q"] + rl["pre_c"])
    vanilla = tv.replay_numpy(random_weights("vanilla_rnn", rng), sequence, "vanilla_rnn")
    np.testing.assert_array_equal(vanilla["pre_q"], np.zeros((6, K)))


def test_replay_rejects_unknown_model():
    weights = random_weights("rl_ann", np.random.default_rng(0))
    with pytest.raises(ValueError, match="Unknown model: bogus"):
        tv.replay_numpy(weights, make_sequence("m", 5, "s", 3, 1), "bogus")


def test_session_traces_load_each_architectures_saved_weights(results, sequences):
    pairs = tv.held_out_sessions(results, sequences)
    traces = tv.session_traces(*pairs[0])
    assert list(traces) == list(NEURAL_MODELS)
    with np.load(f"{pairs[0][1]['output_dir']}/rl_ann.npz") as saved:
        expected = tv.replay_numpy({k: saved[k] for k in saved.files}, pairs[0][0], "rl_ann")
    np.testing.assert_array_equal(traces["rl_ann"]["logits"], expected["logits"])


def test_plot_trajectories_panels(results, sequences):
    sequence, fold = tv.held_out_sessions(results, sequences)[0]
    traces = tv.session_traces(sequence, fold)
    fig = tv.plot_trajectories(sequence, traces, "probability", (3, 6))
    assert len(fig.axes) == 6
    for ax in fig.axes[:4]:
        assert len(ax.lines) == 3 and ax.get_ylim() == (0, 1)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [3, 4, 5, 6])
    summed = sum(line.get_ydata() for line in fig.axes[0].lines)
    np.testing.assert_allclose(summed, 1)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == TOWERS
    assert fig.axes[4].collections[0].get_offsets().shape == (4, 2)
    np.testing.assert_array_equal(fig.axes[5].lines[0].get_ydata(), sequence["rewards"][2:6])
    assert fig._suptitle.get_text().startswith("m1 · ses-b · problem 5")
    plt.close(fig)


def test_vanilla_panel_has_no_q_or_c_channels(results, sequences):
    sequence, fold = tv.held_out_sessions(results, sequences)[0]
    traces = tv.session_traces(sequence, fold)
    fig = tv.plot_trajectories(sequence, traces, "q")
    assert len(fig.axes[3].lines) == 0 and "no separate Q" in fig.axes[3].texts[0].get_text()
    assert len(fig.axes[0].lines) == 3
    plt.close(fig)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(metric="nope"), "Unknown trajectory metric: nope"),
        (dict(trial_range=(0, 3)), "Trial range must lie within the session"),
        (dict(trial_range=(5, 3)), "Trial range must lie within the session"),
        (dict(trial_range=(1, 99)), "Trial range must lie within the session"),
    ],
)
def test_plot_trajectories_validates_inputs(results, sequences, kwargs, message):
    sequence, fold = tv.held_out_sessions(results, sequences)[0]
    with pytest.raises(ValueError, match=message):
        tv.plot_trajectories(sequence, tv.session_traces(sequence, fold), **kwargs)


def test_initial_controls(results, sequences, screen, capsys):
    box = tv.trajectory_viewer(results, sequences)
    subject, session, metric, window, output = controls(box)
    assert subject.options == ("m1", "m2") and subject.value == "m1"
    assert [label for label, _ in session.options] == ["ses-b · problem 5", "ses-a · problem 5"]
    assert session.value == 0
    assert dict(metric.options) == {
        "Choice probabilities": "probability",
        "Choice logits": "logits",
        "Reward channel Q": "q",
        "Action channel c": "c",
    }
    assert metric.value == "probability"
    assert (window.min, window.max, window.value) == (1, 12, (1, 12))
    assert window.continuous_update is False
    assert [w.layout.width for w in (session, metric, window)] == ["650px", "350px", "650px"]
    assert [w.description for w in (subject, session, metric, window)] == [
        "Subject:", "Session:", "Trajectory:", "Trials:",
    ]  # fmt: skip
    assert len(screen.figures) == 1 and screen.cleared == 1 and len(screen.closed) == 1
    assert (
        capsys.readouterr().out == "Held-out problem [5]; validation [6]; pooled training [7, 8].\n"
    )


def test_each_control_change_redraws_once(results, sequences, screen):
    box = tv.trajectory_viewer(results, sequences)
    subject, session, metric, window, _ = controls(box)
    n = len(screen.figures)
    metric.value = "logits"
    assert len(screen.figures) == n + 1
    ax = screen.figures[-1].axes[0]
    assert ax.get_ylabel() == "Choice logit"
    window.value = (3, 6)
    assert len(screen.figures) == n + 2
    np.testing.assert_array_equal(screen.figures[-1].axes[0].lines[0].get_xdata(), [3, 4, 5, 6])
    session.value = 1
    assert len(screen.figures) == n + 3  # one draw, not one per rewritten control
    assert window.value == (1, 8) and window.max == 8
    assert screen.figures[-1]._suptitle.get_text().startswith("m1 · ses-a")
    subject.value = "m2"
    assert len(screen.figures) == n + 4
    assert [label for label, _ in session.options] == ["ses-d · problem 5"]
    assert session.value == 2 and window.value == (1, 10)
    assert len(screen.closed) == len(screen.figures) == screen.cleared


def test_held_out_mouse_message_and_returning_to_first_subject(results, sequences, screen, capsys):
    box = tv.trajectory_viewer(results, sequences)
    subject = controls(box)[0]
    capsys.readouterr()
    subject.value = "m2"
    assert capsys.readouterr().out == (
        "Held-out mouse: this subject was excluded from training and validation.\n"
    )
    subject.value = "m1"
    assert capsys.readouterr().out.startswith("Held-out problem [5]")
    assert controls(box)[1].value == 0


def test_replay_is_cached_per_session(results, sequences, screen, monkeypatch):
    calls = []
    real = tv.session_traces
    monkeypatch.setattr(
        tv, "session_traces", lambda *a: (calls.append(a[0]["session"]), real(*a))[1]
    )
    box = tv.trajectory_viewer(results, sequences)
    _, session, metric, window, _ = controls(box)
    metric.value = "q"
    window.value = (2, 5)
    assert calls == ["ses-b"]
    session.value = 1
    assert calls == ["ses-b", "ses-a"]
    session.value = 0
    assert calls == ["ses-b", "ses-a", "ses-b"]


def test_replay_failure_is_shown_in_the_output_area(results, sequences, screen, capsys):
    (Path(results["folds"][0]["output_dir"]) / "rl_ann.npz").unlink()
    box = tv.trajectory_viewer(results, sequences)
    assert screen.figures == [] and screen.cleared == 1
    assert capsys.readouterr().out.startswith("FileNotFoundError: ")
    controls(box)[2].value = "q"
    assert screen.figures == []


def test_viewer_needs_a_completed_fold(results, sequences):
    with pytest.raises(ValueError, match="No held-out sessions with completed fits"):
        tv.trajectory_viewer({"folds": [results["folds"][2]]}, sequences)
    with pytest.raises(ValueError, match="No held-out sessions with completed fits"):
        tv.trajectory_viewer(results, [])


def test_missing_ipywidgets_explains_the_install(results, sequences, monkeypatch):
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    with pytest.raises(ImportError, match="Install ipywidgets in the notebook kernel") as info:
        tv.trajectory_viewer(results, sequences)
    assert isinstance(info.value.__cause__, ImportError)


def test_inputs_are_not_modified(results, sequences, screen):
    before = repr(results), [(s["actions"].tolist(), s["rewards"].tolist()) for s in sequences]
    box = tv.trajectory_viewer(results, sequences)
    controls(box)[3].value = (2, 4)
    assert before == (
        repr(results),
        [(s["actions"].tolist(), s["rewards"].tolist()) for s in sequences],
    )
