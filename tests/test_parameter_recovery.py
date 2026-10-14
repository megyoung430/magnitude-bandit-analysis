"""Behaviour of recovery_design, run_recovery and summarize_recovery via the public API.

The optimiser is replaced by small fakes that echo what they were given, so the tests pin
the sweep orchestration: seeding, checkpointing, resume, failure handling and layout.
"""

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agents import parameter_recovery
from src.agents.parameter_recovery import (
    NAMES,
    recovery_design,
    run_recovery,
    summarize_recovery,
)

BOUNDS = [(0.001, 1.0), (0.01, 10.0), (0.0, 3.0)]


def make_config(**changes):
    config = dict(
        parameter_grid={"learning_rate": [0.1, 0.3], "beta": [2.0], "alpha": [0.0, 0.5]},
        swap_modes=["random", "most_chosen"],
        trial_lengths=[20],
        n_repeats=2,
        n_starts=3,
        seed=77,
        initial_arms=[0, 1, 4],
        task_parameters={"drift_rate": 0.1, "min_block_length": 4, "max_block_length": 6},
    )
    return {**config, **changes}


def make_fit(estimates=None, fails_on=(), crashes_on=()):
    """Fake fit: returns fixed estimates and records the calls it received.

    A plain function, because run_recovery hashes the source of the fit into the manifest.
    """
    calls = []
    estimates = estimates or dict(learning_rate=0.2, beta=2.0, alpha=0.25)

    def fit(actions, rewards, num_arms, n_starts, seed):
        calls.append(
            dict(
                actions=actions.copy(),
                rewards=rewards.copy(),
                num_arms=num_arms,
                n_starts=n_starts,
                seed=seed,
            )
        )
        if len(calls) in fails_on:
            raise RuntimeError(f"no convergence #{len(calls)}")
        if len(calls) in crashes_on:
            raise ValueError("optimiser crashed")
        return dict(estimates), SimpleNamespace(fun=1.5, message="converged")

    fit.calls = calls
    return fit


def mean_choice_loss(params, actions, rewards, num_arms):
    return float(np.mean(actions) + sum(params))


def run(root, fit=None, config=None, bounds=BOUNDS, loss=mean_choice_loss, **kwargs):
    fit = fit or make_fit()
    return run_recovery(config or make_config(), fit, loss, bounds, root, **kwargs)


def experiment_dirs(root):
    return [p for p in root.iterdir() if p.is_dir()]


def test_design_is_full_factorial_with_stable_independent_seeds():
    design = recovery_design(make_config())
    assert len(design) == 2 * 1 * 2 * 1 * 2 * 2
    assert design.run_id.is_unique and design.seed.is_unique
    assert list(design.columns) == [
        "run_id", "swap_mode", "num_trials", "repeat", "seed",
        "true_learning_rate", "true_beta", "true_alpha",
    ]  # fmt: skip
    pd.testing.assert_frame_equal(design, recovery_design(make_config()))
    assert recovery_design(make_config(seed=78)).seed.tolist() != design.seed.tolist()
    # a condition keeps its id and seed when other conditions are added
    bigger = recovery_design(make_config(trial_lengths=[20, 30]))
    kept = bigger[bigger.num_trials == 20].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        kept.sort_values("run_id").reset_index(drop=True),
        design.sort_values("run_id").reset_index(drop=True),
    )
    row = design.iloc[0]
    identity = [row.swap_mode, 20, row.true_learning_rate, 2.0, row.true_alpha, int(row["repeat"])]
    digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    assert row.run_id == digest[:16]
    assert row.seed == int(np.random.SeedSequence([77, int(digest[:8], 16)]).generate_state(1)[0])


def test_run_writes_layout_and_complete_results(tmp_path):
    fit = make_fit()
    results, output = run(tmp_path, fit)
    design = recovery_design(make_config())
    assert output.parent == tmp_path and len(results) == len(design) == len(fit.calls)
    names = {p.name for p in output.iterdir()}
    assert {"manifest.json", "design.csv", "results.json", "results.csv"} <= names
    assert {f"session_{run_id}.npz" for run_id in design.run_id} <= names
    assert not [n for n in names if n.endswith(".tmp")]
    manifest_text = (output / "manifest.json").read_text()
    assert output.name == hashlib.sha256(manifest_text.encode()).hexdigest()[:12]
    manifest = json.loads(manifest_text)
    assert manifest["config"] == make_config() and manifest["bounds"] == [list(b) for b in BOUNDS]
    assert {"task", "mouse", "runner", "module", "design", "fit", "loss"} <= set(manifest["code"])
    assert manifest["code"]["module"] == Path(parameter_recovery.__file__).read_text()
    pd.testing.assert_frame_equal(pd.read_csv(output / "design.csv"), design)
    saved = json.loads((output / "results.json").read_text())
    assert [r["run_id"] for r in saved] == design.run_id.tolist()
    assert pd.read_csv(output / "results.csv").run_id.tolist() == design.run_id.tolist()


def test_result_rows(tmp_path):
    fit = make_fit(estimates=dict(learning_rate=0.00105, beta=5.0, alpha=3.0))
    results, output = run(tmp_path, fit)
    row = results.iloc[0]
    assert bool(row.success) and row.loss == 1.5 and row.message == "converged"
    assert (row.recovered_learning_rate, row.recovered_beta, row.recovered_alpha) == (
        0.00105,
        5.0,
        3.0,
    )
    assert [row.at_bound_learning_rate, row.at_bound_beta, row.at_bound_alpha] == [
        True, False, True]  # fmt: skip
    data = np.load(output / f"session_{row.run_id}.npz")
    assert row.uniform_loss == pytest.approx(np.log(3))
    true = [row.true_learning_rate, row.true_beta, row.true_alpha]
    assert row.true_loss == pytest.approx(np.mean(data["actions"]) + sum(true))
    assert (results.fit_seconds >= 0).all()
    assert list(results.columns[:8]) == list(recovery_design(make_config()).columns)


def test_session_data_and_fit_inputs_are_seeded_per_run(tmp_path):
    fit = make_fit()
    results, output = run(tmp_path, fit)
    for call, (_, row) in zip(fit.calls, results.iterrows(), strict=True):
        data = np.load(output / f"session_{row.run_id}.npz")
        np.testing.assert_array_equal(call["actions"], data["actions"])
        np.testing.assert_array_equal(call["rewards"], data["rewards"])
        assert data["arm_history"].shape == (20, 3)
        assert len(data["actions"]) == 20 == row.num_trials
        assert (call["num_arms"], call["n_starts"], call["seed"]) == (3, 3, row.seed)
    assert len({tuple(c["actions"]) for c in fit.calls}) > 1


def test_simulated_data_does_not_depend_on_ambient_random_state(tmp_path):
    np.random.seed(1)
    random.seed(1)
    first = run(tmp_path / "a")[1]
    np.random.seed(999)
    random.seed(999)
    second = run(tmp_path / "b")[1]
    for path in sorted(first.glob("session_*.npz")):
        a, b = np.load(path), np.load(second / path.name)
        for key in ("actions", "rewards", "arm_history"):
            np.testing.assert_array_equal(a[key], b[key])


def test_global_random_state_is_restored(tmp_path):
    np.random.seed(5)
    random.seed(6)
    expected = (np.random.random(2), random.random())
    np.random.seed(5)
    random.seed(6)
    run(tmp_path)
    np.testing.assert_array_equal(np.random.random(2), expected[0])
    assert random.random() == expected[1]


def test_global_random_state_is_restored_after_a_crash(tmp_path):
    np.random.seed(5)
    random.seed(6)
    expected = (np.random.random(2), random.random())
    np.random.seed(5)
    random.seed(6)
    with pytest.raises(ValueError, match="optimiser crashed"):
        run(tmp_path, make_fit(crashes_on={3}))
    np.testing.assert_array_equal(np.random.random(2), expected[0])
    assert random.random() == expected[1]


def test_pilot_then_resume_runs_only_the_remaining_sessions(tmp_path, capsys):
    fit = make_fit()
    partial, output = run(tmp_path, fit, max_new_runs=5)
    assert len(partial) == 5 and len(fit.calls) == 5
    assert len(pd.read_csv(output / "results.csv")) == 5
    assert capsys.readouterr().out.splitlines()[0] == "0/16 sessions saved; running 5."
    same, _ = run(tmp_path, fit, max_new_runs=0)
    assert len(same) == 5 and len(fit.calls) == 5
    capsys.readouterr()
    full, again = run(tmp_path, fit)
    assert again == output and len(full) == 16 and len(fit.calls) == 16
    assert full.run_id.tolist() == recovery_design(make_config()).run_id.tolist()
    assert capsys.readouterr().out.splitlines()[0] == "5/16 sessions saved; running 11."
    run(tmp_path, fit)
    assert len(fit.calls) == 16
    assert len(experiment_dirs(tmp_path)) == 1


def test_directory_changes_with_config_bounds_and_code(tmp_path):
    base = run(tmp_path)[1]
    assert run(tmp_path)[1] == base
    assert run(tmp_path, config=make_config(seed=1))[1] != base
    assert run(tmp_path, config=make_config(n_starts=2))[1] != base
    assert run(tmp_path, bounds=[(0.0, 1.0), (0.01, 10.0), (0.0, 3.0)])[1] != base

    def other_loss(params, actions, rewards, num_arms):
        return 0.0

    assert run(tmp_path, loss=other_loss)[1] != base


def test_directory_changes_with_a_module_constant(tmp_path, edit_module_text):
    base = run(tmp_path)[1]
    edit_module_text(parameter_recovery, "ATOL = 1e-4", "ATOL = 1e-3")
    assert run(tmp_path)[1] != base


def test_at_bound_tolerance_edges(tmp_path):
    # 0.001 is the lower learning-rate bound: 5e-5 away is hit, 1.5e-4 away is not
    for name, estimate, expected in [("inside", 0.00105, True), ("outside", 0.00115, False)]:
        fit = make_fit(estimates=dict(learning_rate=estimate, beta=5.0, alpha=1.0))
        results, _ = run(tmp_path / name, fit)
        assert (results.at_bound_learning_rate == expected).all()


def test_runtime_error_is_a_visible_failed_row_and_not_retried(tmp_path):
    fit = make_fit(fails_on={2, 4})
    results, output = run(tmp_path, fit)
    failed = results[~results.success.astype(bool)]
    assert failed.index.tolist() == [1, 3]
    assert failed.message.tolist() == ["no convergence #2", "no convergence #4"]
    assert failed.loss.isna().all() and failed.recovered_beta.isna().all()
    assert failed.at_bound_alpha.isna().all()
    assert "true_loss" in failed and failed.true_loss.notna().all()
    again, _ = run(tmp_path, fit)
    assert len(fit.calls) == 16
    assert (output / "results.json").exists() and len(again) == 16


def test_other_exceptions_propagate_and_keep_finished_runs(tmp_path):
    fit = make_fit(crashes_on={4})
    with pytest.raises(ValueError, match="optimiser crashed"):
        run(tmp_path, fit)
    (output,) = experiment_dirs(tmp_path)
    assert len(json.loads((output / "results.json").read_text())) == 3
    assert not (output / "results.csv").exists()
    resumed, _ = run(tmp_path, make_fit())
    assert len(resumed) == 16


def test_progress_lines(tmp_path, capsys, monkeypatch):
    clock = iter(float(i) for i in range(10_000))
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))
    run(tmp_path, config=make_config(n_repeats=6, swap_modes=["random"]))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "0/24 sessions saved; running 24."
    assert lines[1] == str(next(iter(experiment_dirs(tmp_path))))
    progress = lines[2:]
    assert [line.split(" saved")[0] for line in progress] == ["1/24", "10/24", "20/24", "24/24"]
    assert progress[-1].endswith("remaining estimate 0.0 min")


@pytest.mark.parametrize(
    "config",
    [make_config(n_repeats=0), make_config(swap_modes=["random", "random"])],
)
def test_empty_or_duplicated_design_is_rejected_before_writing(tmp_path, config):
    with pytest.raises(ValueError, match="unique, nonempty"):
        run(tmp_path, config=config)
    assert list(tmp_path.iterdir()) == []


def test_config_is_not_modified(tmp_path):
    config = make_config()
    snapshot = json.dumps(config, sort_keys=True)
    bounds = list(BOUNDS)
    run(tmp_path, config=config, bounds=bounds)
    assert json.dumps(config, sort_keys=True) == snapshot and bounds == BOUNDS


def test_summarize_recovery_without_pooling_conditions():
    results = pd.DataFrame(
        {
            "mode": ["a"] * 4 + ["b"] * 2,
            "success": [True, True, True, False, True, True],
            **{f"true_{n}": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0] for n in NAMES},
            **{f"recovered_{n}": [1.5, 1.5, 3.5, np.nan, 5.0, 6.0] for n in NAMES},
            **{f"at_bound_{n}": [True, False, False, np.nan, True, True] for n in NAMES},
        }
    )
    summary = summarize_recovery(results, ["mode"])
    assert summary.parameter.tolist() == NAMES * 2
    a = summary[(summary["mode"] == "a") & (summary.parameter == "beta")].iloc[0]
    assert (a.n_sessions, a.n_success) == (4, 3)
    assert a.failure_rate == pytest.approx(0.25)
    errors = np.array([0.5, -0.5, 0.5])
    assert a.bias == pytest.approx(errors.mean())
    assert a.mae == pytest.approx(0.5) and a.rmse == pytest.approx(0.5)
    assert a.correlation == pytest.approx(np.corrcoef([1, 2, 3], [1.5, 1.5, 3.5])[0, 1])
    assert a.boundary_rate == pytest.approx(1 / 3)
    b = summary[(summary["mode"] == "b") & (summary.parameter == "beta")].iloc[0]
    assert np.isnan(b.correlation)  # a single true value has no correlation
