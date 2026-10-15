"""Behaviour of run_problem_cross_validation and prepare_cohort_sequences, via the public API.

Fits are replaced by a recording fake so the tests pin the orchestration (folds, seeds,
resume, on-disk layout, summary rows) without depending on the optimiser.
"""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents import cohort_fitting
from src.agents.cohort_fitting import (
    DEFAULT_BOUNDS,
    prepare_cohort_sequences,
    run_problem_cross_validation,
    sequence_log_probabilities,
)
from src.agents.run_store import ProvenanceMismatchWarning, identity_hash

TOWERS = ["A1", "B1", "C1"]
PARAMS = [0.2, 2.0, 0.5]


def make_sequence(mouse, problem, session=None, n=6, seed=0):
    rng = np.random.default_rng(1000 * problem + seed + len(mouse))
    return dict(
        mouse=mouse,
        problem=problem,
        session=session or f"ses-{problem}",
        towers=list(TOWERS),
        actions=rng.integers(0, 3, n),
        rewards=rng.integers(0, 5, n).astype(float),
    )


@pytest.fixture
def sequences():
    out = [make_sequence(mouse, problem) for mouse in ("m1", "m2") for problem in (5, 6, 7)]
    out.append(make_sequence("m1", 6, session="ses-6b", n=4))  # second session in one problem
    out.append(make_sequence("partial", 5))  # lacks problems 6 and 7
    out.append(make_sequence("m1", 99))  # outside the requested problems
    return out


class FitSpy:
    """Replaces fit_mouse_sessions; records arguments and returns fixed parameters."""

    def __init__(self, fit=None, fails_for=()):
        self.calls = []
        self.fit = dict(parameters=PARAMS, loss=0.8) if fit is None else fit
        self.fails_for = set(fails_for)

    def __call__(self, training, bounds, n_starts, seed):
        held_out = (
            {s["problem"] for s in self.all_problems} - {s["problem"] for s in training}
        ).pop()
        self.calls.append(
            dict(
                mouse=training[0]["mouse"],
                held_out=held_out,
                problems=sorted({s["problem"] for s in training}),
                n_sessions=len(training),
                bounds=bounds,
                n_starts=n_starts,
                seed=seed,
            )
        )
        if held_out in self.fails_for:
            return None, [dict(success=False)]
        return self.fit, [dict(success=True), dict(success=False)]


@pytest.fixture
def spy(monkeypatch, sequences):
    fake = FitSpy()
    fake.all_problems = [s for s in sequences if s["problem"] in (5, 6, 7)]
    monkeypatch.setattr(cohort_fitting, "fit_mouse_sessions", fake)
    return fake


def run(sequences, root, **kwargs):
    kwargs.setdefault("problem_numbers", [5, 6, 7])
    return run_problem_cross_validation(sequences, root, **kwargs)


def test_output_layout_and_manifest_hash(sequences, spy, tmp_path):
    _, _, _, output = run(sequences, tmp_path, n_starts=3, seed=5, source_metadata={"k": 1})
    manifest = json.loads((output / "manifest.json").read_text())
    assert output.parent == tmp_path
    identity = {k: v for k, v in manifest.items() if k != "implementation"}
    assert output.name == identity_hash(identity)
    assert manifest["implementation"] == Path(cohort_fitting.__file__).read_text()
    assert manifest["algorithm_version"] == cohort_fitting.ALGORITHM_VERSION
    assert manifest["at_bound_atol"] == cohort_fitting._AT_BOUND_ATOL
    assert set(manifest["versions"]) == {"numpy", "scipy"}
    assert manifest["problems"] == [5, 6, 7]
    assert (manifest["n_starts"], manifest["seed"], manifest["model"]) == (3, 5, "sticky_rl")
    assert manifest["source"] == {"k": 1}
    assert manifest["bounds"] == [list(b) for b in DEFAULT_BOUNDS]
    names = sorted(p.name for p in output.iterdir())
    assert {
        "fold_results.csv",
        "held_out_sessions.csv",
        "manifest.json",
        "mouse_eligibility.csv",
    } <= set(names)
    assert len([n for n in names if n.startswith("fold_") and n.endswith(".json")]) == 6
    assert len([n for n in names if n.startswith("predictions_")]) == 6
    assert not [n for n in names if n.endswith(".tmp")]


def test_output_directory_changes_with_data_and_settings(sequences, spy, tmp_path):
    base = run(sequences, tmp_path)[3]
    assert run(sequences, tmp_path)[3] == base
    assert run(sequences, tmp_path, seed=1)[3] != base
    assert run(sequences, tmp_path, n_starts=2)[3] != base
    assert run(sequences, tmp_path, source_metadata="x")[3] != base
    changed = [dict(s, rewards=s["rewards"] + 1) if s["mouse"] == "m1" else s for s in sequences]
    assert run(changed, tmp_path)[3] != base
    # sequences outside the requested problems do not enter the data fingerprint
    extra = [*sequences, make_sequence("m1", 98)]
    assert run(extra, tmp_path)[3] == base


def test_output_directory_changes_with_a_module_constant(sequences, spy, tmp_path, monkeypatch):
    base = run(sequences, tmp_path)[3]
    monkeypatch.setattr(cohort_fitting, "_AT_BOUND_ATOL", 1e-3)
    assert run(sequences, tmp_path)[3] != base


def test_output_directory_changes_with_the_algorithm_version(sequences, spy, tmp_path, monkeypatch):
    base = run(sequences, tmp_path)[3]
    monkeypatch.setattr(cohort_fitting, "ALGORITHM_VERSION", cohort_fitting.ALGORITHM_VERSION + 1)
    assert run(sequences, tmp_path)[3] != base


def test_output_directory_changes_with_library_versions(sequences, spy, tmp_path, monkeypatch):
    base = run(sequences, tmp_path)[3]
    monkeypatch.setattr(
        cohort_fitting, "library_versions", lambda *names: dict.fromkeys(names, "0")
    )
    assert run(sequences, tmp_path)[3] != base


def test_editing_source_text_keeps_the_directory_and_resumes(
    sequences, spy, tmp_path, edit_module_text
):
    _, _, _, base = run(sequences, tmp_path)
    original = json.loads((base / "manifest.json").read_text())["implementation"]
    n_fits = len(spy.calls)
    edit_module_text(cohort_fitting, "_AT_BOUND_ATOL = 1e-4", "_AT_BOUND_ATOL = 1e-4  # note")
    with pytest.warns(ProvenanceMismatchWarning, match="implementation"):
        _, _, _, resumed = run(sequences, tmp_path)
    assert resumed == base
    assert len(spy.calls) == n_fits  # every fold was loaded, none refitted
    # the manifest keeps describing the code that produced the saved folds
    assert json.loads((resumed / "manifest.json").read_text())["implementation"] == original


def _model_identity(sequences, model, module):
    return cohort_fitting._build_manifest(
        cohort_fitting._model_spec(model, module.DEFAULT_BOUNDS),
        sequences,
        problem_numbers=[5, 6, 7],
        bounds=module.DEFAULT_BOUNDS,
        n_starts=1,
        seed=0,
        model=model,
        source_metadata=None,
    )


@pytest.mark.parametrize("model", ["best_rl", "released_rl"])
def test_editing_model_source_changes_provenance_only(sequences, edit_module_text, model):
    module = importlib.import_module(f"src.agents.{model}")
    identity, provenance = _model_identity(sequences, model, module)
    edit_module_text(module, "import", "import  # edited")
    edited_identity, edited_provenance = _model_identity(sequences, model, module)
    assert edited_identity == identity
    assert edited_provenance["model_implementation"] != provenance["model_implementation"]


@pytest.mark.parametrize(
    ("model", "bumped"),
    [("best_rl", ["best_rl"]), ("released_rl", ["released_rl"]), ("released_rl", ["best_rl"])],
)
def test_model_version_is_part_of_the_identity(sequences, monkeypatch, model, bumped):
    module = importlib.import_module(f"src.agents.{model}")
    base = identity_hash(_model_identity(sequences, model, module)[0])
    version_module = importlib.import_module(f"src.agents.{bumped[0]}")
    monkeypatch.setattr(version_module, "MODEL_VERSION", version_module.MODEL_VERSION + 1)
    assert identity_hash(_model_identity(sequences, model, module)[0]) != base


@pytest.mark.parametrize(("learning_rate", "expected"), [(0.00105, True), (0.00115, False)])
def test_at_bound_tolerance_edges(sequences, spy, tmp_path, learning_rate, expected):
    # 0.001 is the lower learning-rate bound: 5e-5 away is hit, 1.5e-4 away is not
    spy.fit = dict(parameters=[learning_rate, 2.0, 0.5], loss=0.8)
    results = run(sequences, tmp_path)[0]
    assert (results.at_bound_learning_rate == expected).all()


def test_eligibility_table_and_fold_loop(sequences, spy, tmp_path, capsys):
    results, sessions, eligibility, output = run(sequences, tmp_path)
    assert eligibility.to_dict("records") == [
        dict(mouse="m1", eligible=True, missing_problems=""),
        dict(mouse="m2", eligible=True, missing_problems=""),
        dict(mouse="partial", eligible=False, missing_problems="6,7"),
    ]
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "mouse_eligibility.csv").fillna(""), eligibility
    )
    assert [(c["mouse"], c["held_out"]) for c in spy.calls] == [
        (m, p) for m in ("m1", "m2") for p in (5, 6, 7)
    ]
    assert results[["mouse", "held_out_problem"]].values.tolist() == [
        [m, p] for m in ("m1", "m2") for p in (5, 6, 7)
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"{m}: held out problem {p}; fit saved" for m in ("m1", "m2") for p in (5, 6, 7)
    ]
    assert len(sessions) == 7  # m1 has two sessions in problem 6
    saved = pd.read_csv(output / "fold_results.csv")
    assert saved.shape[0] == 6 and saved.mouse.tolist() == results.mouse.tolist()


def test_fit_receives_training_sessions_and_options(sequences, spy, tmp_path):
    bounds = [(0.0, 1.0), (0.1, None), (0.0, 2.0)]
    run(sequences, tmp_path, bounds=bounds, n_starts=4)
    m1_p6 = next(c for c in spy.calls if (c["mouse"], c["held_out"]) == ("m1", 6))
    assert m1_p6["problems"] == [5, 7] and m1_p6["n_sessions"] == 2
    m1_p5 = next(c for c in spy.calls if (c["mouse"], c["held_out"]) == ("m1", 5))
    assert m1_p5["n_sessions"] == 3  # both problem-6 sessions stay in training
    assert all(c["bounds"] is bounds and c["n_starts"] == 4 for c in spy.calls)


def test_fold_seeds_are_derived_per_mouse_and_problem(sequences, spy, tmp_path):
    run(sequences, tmp_path, seed=7)
    seeds = {(c["mouse"], c["held_out"]): c["seed"] for c in spy.calls}
    assert len(set(seeds.values())) == 6
    key = hashlib.sha256(b"m2:6").hexdigest()[:12]
    expected = int(np.random.SeedSequence([7, int(key[:8], 16)]).generate_state(1)[0])
    assert seeds[("m2", 6)] == expected
    spy.calls.clear()
    run(sequences, tmp_path / "again", seed=7)
    assert {(c["mouse"], c["held_out"]): c["seed"] for c in spy.calls} == seeds
    spy.calls.clear()
    run(sequences, tmp_path / "other", seed=8)
    assert all(c["seed"] != seeds[(c["mouse"], c["held_out"])] for c in spy.calls)


def test_fold_does_not_touch_global_random_state(sequences, spy, tmp_path):
    np.random.seed(11)
    expected = np.random.random(3)
    np.random.seed(11)
    run(sequences, tmp_path)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_fold_row_contents(sequences, spy, tmp_path):
    spy.fit = dict(parameters=[1.0, 5.0, 0.00005], loss=0.75)
    bounds = [(0.001, 1.0), (0.01, None), (0.0, 3.0)]
    results, _, _, _ = run(sequences, tmp_path, bounds=bounds)
    row = results[(results.mouse == "m1") & (results.held_out_problem == 6)].iloc[0]
    assert row.training_problems == "5,7"
    assert (row.n_train_trials, row.n_test_trials) == (12, 10)
    assert (row.n_train_sessions, row.n_test_sessions) == (2, 2)
    assert bool(row.success) and row.converged_starts == 1
    validation = [s for s in sequences if s["mouse"] == "m1" and s["problem"] == 6]
    nll = sum(
        -sequence_log_probabilities(spy.fit["parameters"], s)[
            np.arange(len(s["actions"])), s["actions"]
        ].sum()
        for s in validation
    )
    assert row.train_cross_entropy == 0.75
    assert row.test_cross_entropy == pytest.approx(nll / 10, rel=1e-12)
    assert row.uniform_cross_entropy == pytest.approx(np.log(3))
    assert row.improvement_over_uniform == pytest.approx(np.log(3) - nll / 10, rel=1e-12)
    assert (row.learning_rate, row.beta, row.alpha) == (1.0, 5.0, 0.00005)
    # an unbounded side (None) is never "hit"; 5.0 lies 5 away from the finite end only
    assert (row.at_bound_learning_rate, row.at_bound_beta, row.at_bound_alpha) == (
        True,
        False,
        True,
    )


def test_held_out_sessions_and_predictions(sequences, spy, tmp_path):
    _, sessions, _, output = run(sequences, tmp_path)
    rows = sessions[(sessions.mouse == "m1") & (sessions.held_out_problem == 6)]
    assert rows.session.tolist() == ["ses-6", "ses-6b"] and rows.n_trials.tolist() == [6, 4]
    key = hashlib.sha256(b"m1:6").hexdigest()[:12]
    predictions = pd.read_csv(output / f"predictions_{key}.csv")
    assert predictions.trial.tolist() == [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    assert set(predictions.session) == {"ses-6", "ses-6b"}
    assert np.allclose(predictions[["p_A1", "p_B1", "p_C1"]].sum(axis=1), 1)
    first = next(s for s in sequences if s["mouse"] == "m1" and s["session"] == "ses-6")
    assert predictions.chosen_tower[:6].tolist() == [TOWERS[a] for a in first["actions"]]
    assert predictions.reward[:6].tolist() == first["rewards"].tolist()
    chosen = predictions[["p_A1", "p_B1", "p_C1"]].to_numpy()[np.arange(6), first["actions"]]
    np.testing.assert_allclose(-np.log(chosen), predictions.negative_log_probability[:6])


def test_resume_refits_only_missing_folds(sequences, spy, tmp_path, capsys):
    first, _, _, output = run(sequences, tmp_path)
    capsys.readouterr()
    spy.calls.clear()
    again, _, _, same = run(sequences, tmp_path)
    assert same == output and spy.calls == [] and capsys.readouterr().out == ""
    pd.testing.assert_frame_equal(again, first)
    victim = sorted(output.glob("fold_*.json"))[2]
    victim.unlink()
    (output / "fold_results.csv").unlink()
    resumed, _, _, _ = run(sequences, tmp_path)
    assert len(spy.calls) == 1
    pd.testing.assert_frame_equal(resumed, first)
    assert victim.exists() and (output / "fold_results.csv").exists()
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_failed_fit_is_recorded_without_predictions(sequences, spy, tmp_path, capsys):
    spy.fails_for = {6}
    results, _, _, output = run(sequences, tmp_path)
    failed = results[results.held_out_problem == 6]
    assert not failed.success.any() and failed.test_cross_entropy.isna().all()
    assert failed.uniform_cross_entropy.tolist() == [pytest.approx(np.log(3))] * 2
    assert len(list(output.glob("predictions_*.csv"))) == 4
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "m1: held out problem 6; all starts failed"
    saved = json.loads(sorted(output.glob("fold_*.json"))[0].read_text())
    assert set(saved) == {"fold", "sessions", "starts"}


def test_nonfinite_loss_is_not_saved_silently(sequences, spy, tmp_path):
    spy.fit = dict(parameters=PARAMS, loss=float("nan"))
    with pytest.raises(ValueError, match="not JSON compliant"):
        run(sequences, tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(model="nope"), "Unknown model"),
        (dict(model="best_rl"), "six bounds"),
        (dict(model="released_rl"), "five bounds"),
        (dict(problem_numbers=[5]), "at least two problems"),
        (dict(problem_numbers=[6, 6]), "at least two problems"),
    ],
)
def test_invalid_arguments_fail_before_writing(sequences, spy, tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        run(sequences, tmp_path, **kwargs)
    assert list(tmp_path.iterdir()) == []


def test_problem_numbers_may_be_unsorted_and_one_shot(sequences, spy, tmp_path):
    expected = run(sequences, tmp_path / "a")[0]
    unsorted = run(sequences, tmp_path / "b", problem_numbers=(p for p in (7, 5, 6)))[0]
    pd.testing.assert_frame_equal(expected, unsorted)


def test_inputs_are_not_modified(sequences, spy, tmp_path):
    bounds = [(0.0, 1.0), (0.1, 5.0), (0.0, 2.0)]
    before = (
        json.dumps(
            [
                [s["mouse"], s["problem"], s["actions"].tolist(), s["rewards"].tolist()]
                for s in sequences
            ]
        ),
        json.dumps(bounds),
    )
    run(sequences, tmp_path, bounds=bounds)
    after = (
        json.dumps(
            [
                [s["mouse"], s["problem"], s["actions"].tolist(), s["rewards"].tolist()]
                for s in sequences
            ]
        ),
        json.dumps(bounds),
    )
    assert before == after


def test_real_fit_end_to_end_is_reproducible(tmp_path):
    seqs = [make_sequence(m, p, n=12) for m in ("a", "b") for p in (5, 6)]
    kwargs = dict(problem_numbers=[5, 6], n_starts=2, seed=3)
    first = run_problem_cross_validation(seqs, tmp_path / "x", **kwargs)[0]
    second = run_problem_cross_validation(seqs, tmp_path / "y", **kwargs)[0]
    pd.testing.assert_frame_equal(first, second)
    assert first.success.all() and (first.test_cross_entropy > 0).all()
    assert first.learning_rate.between(0.001, 1.0).all()


def audit_session(**changes):
    session = dict(
        choice=["A1", "B1", "A1"],
        reward_pulses=[0, 4, 2],
        choice_towers=[TOWERS] * 3,
        choices_by_tower={t: [] for t in TOWERS},
    )
    return {**session, **changes}


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (dict(choices_by_tower={"A1": [], "B1": []}), "Expected exactly three choice towers."),
        (dict(choice=[]), "No completed choices."),
        (dict(choice=["A1", "Z", "A1"]), "Missing or unrecognised choice."),
        (dict(reward_pulses=[1, 2]), "Missing or misaligned delivered reward pulses."),
        (dict(reward_pulses=[-1, 1, 1]), "Missing, nonfinite, or negative delivered reward."),
        (dict(reward_pulses=[float("nan"), 1, 1]), "Missing, nonfinite, or negative delivered"),
        (dict(choice_towers=[TOWERS] * 2), "Missing/changing available towers within session."),
        (dict(choice_towers=[TOWERS, "ABC", TOWERS]), "Missing/changing available towers"),
        (dict(choice_towers=[TOWERS, TOWERS, ["A1", "B1", "X"]]), "Missing/changing available"),
    ],
)
def test_audit_reports_why_a_session_is_rejected(changes, reason):
    sequences, audit = prepare_cohort_sequences({5: {"m": {"ses-1": audit_session(**changes)}}}, [5])
    assert sequences == []
    row = audit.iloc[0]
    assert not row.included and row.reason.startswith(reason[:40])
    assert list(audit.columns) == [
        "mouse",
        "problem",
        "session",
        "n_trials",
        "included",
        "reason",
        "reward_min",
        "reward_max",
        "towers",
    ]


def test_audit_orders_sessions_and_summarises_included_ones():
    problems = {
        5: {
            "m": {
                "ses-10_date-20260101": audit_session(reward_pulses=[0, 0, 9]),
                "ses-2_date-20260101": audit_session(reward_pulses=[1, 1, 3]),
            }
        }
    }
    sequences, audit = prepare_cohort_sequences(problems, [5, 6])
    assert [s["session"] for s in sequences] == ["ses-2_date-20260101", "ses-10_date-20260101"]
    assert sequences[0]["towers"] == TOWERS and sequences[0]["actions"].tolist() == [0, 1, 0]
    assert audit.included.tolist() == [True, True]
    assert audit.reward_max.tolist() == [3.0, 9.0] and audit.towers.tolist() == ["A1,B1,C1"] * 2
