"""Golden-value tests for the paper-style RL fitting loop.

The expected numbers were captured from the pre-refactor implementation on the
synthetic data below. They pin the things a structural refactor can silently
change without breaking a shape or finiteness check: the order in which random
keys are drawn and split, which iterate is kept as "best", which partition each
score is computed on, and the names of the files a sweep writes.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.agents import paper_rl_fitting as paper
from src.agents.run_store import ProvenanceMismatchWarning, identity_hash

pytestmark = pytest.mark.skipif(
    not (importlib.util.find_spec("jax") and importlib.util.find_spec("optax")),
    reason="Optional JAX/Optax dependencies not installed",
)

RTOL = 1e-9


def golden_sequences():
    """Eight mice x three problems with a deliberate train/validation mismatch.

    Training mice always choose arm 0 and are rewarded; the validation mouse always
    chooses arm 2 and is never rewarded. Fitting the training data therefore makes
    validation worse after a point, so the best iterate is not the final one.
    """
    rng = np.random.default_rng(0)
    data = [
        dict(
            mouse=f"m{i}",
            problem=p,
            session=f"ses-{p}",
            towers=["A", "B", "C"],
            actions=rng.integers(0, 3, n),
            rewards=rng.integers(0, 5, n).astype(float),
        )
        for i in range(8)
        for p in (5, 6, 7)
        for n in [int(rng.integers(4, 8))]
    ]
    ids = paper.split_by_mouse(data, 42)[1]
    for s in data:
        n = len(s["actions"])
        if s["mouse"] in ids["train"]:
            s["actions"], s["rewards"] = np.zeros(n, dtype=int), np.full(n, 4.0)
        elif s["mouse"] in ids["validation"]:
            s["actions"], s["rewards"] = np.full(n, 2), np.zeros(n)
    return data


SWEEP = dict(
    models=("sticky_rl",),
    grid=[dict(optimizer_learning_rate=2.0, weight_decay=0.1, batch_size=3)],
    n_seeds=2,
    max_steps=5,  # not a multiple of validation_every: the last chunk is shorter
    validation_every=2,
    fit_seed=5,
    profile="golden",
)


class TestInitialParameters:
    key_seed = 5

    def init(self, model, **kwargs):
        import jax

        return paper._initial_raw_parameters(model, jax.random.PRNGKey(self.key_seed), **kwargs)

    def test_sticky_draws_three_normals_scaled_by_one_half(self):
        np.testing.assert_allclose(
            self.init("sticky_rl"),
            [0.17168590385845806, -0.8364987663104276, 0.49088550300515493],
            rtol=RTOL,
        )

    def test_best_and_released_fix_cogmod_values_and_draw_q_init(self):
        q_init = 1.343371807716916
        np.testing.assert_allclose(
            self.init("best_rl"), [1.0, 1.0, 0.0, 0.0, 0.0, q_init], rtol=RTOL
        )
        np.testing.assert_allclose(
            self.init("released_rl"), [1.0, 1.0, 0.0, 0.0, q_init], rtol=RTOL
        )

    def test_neural_initialization_is_pinned_by_per_array_sums(self):
        tree = self.init("rl_ann", hidden_units=4)
        sums = {name: float(np.asarray(v).sum()) for name, v in tree.items()}
        expected = dict(
            action_b1=0.0,
            action_b2=0.0,
            action_w1=-1.7104200828922944,
            action_w2=-2.1989890628439506,
            c_init=2.2179708003196756,
            forgetting_raw=1.1945744070074933,
            q_init=2.8480555766035396,
            reward_b1=0.0,
            reward_b2=0.0,
            reward_w1=0.7989520003175801,
            reward_w2=-0.233663725098821,
        )
        assert sums == pytest.approx(expected, rel=RTOL, abs=1e-12)


class TestTrainerChunk:
    def run_chunk(self, weight_decay):
        import jax
        import jax.numpy as jnp

        _, _, make_trainer = paper.make_backend("sticky_rl")
        optimizer, chunk = make_trainer(
            dict(optimizer_learning_rate=0.05, weight_decay=weight_decay, batch_size=3)
        )
        raw = jnp.asarray([0.3, -0.2, 0.5])
        arrays = tuple(jnp.asarray(x) for x in paper.pack_sequences(golden_sequences()[:6]))
        (new_raw, _, new_key), loss = chunk(
            raw, optimizer.init(raw), jax.random.PRNGKey(11), *arrays, 3
        )
        return np.asarray(new_raw), np.asarray(new_key), float(loss)

    def test_three_steps_from_a_fixed_key_give_exact_parameters(self):
        # Pins the minibatch sampling stream (key split order), the gradient, and the
        # AdamW decay taken from the setting.
        new_raw, new_key, loss = self.run_chunk(weight_decay=0.3)
        np.testing.assert_allclose(
            new_raw, [0.31763380396276253, -0.22175765628985095, 0.44697574486496455], rtol=RTOL
        )
        assert new_key.tolist() == [2768831903, 4209263146]
        assert loss == pytest.approx(8.70680462492162, rel=RTOL)

    def test_weight_decay_changes_the_update(self):
        decayed, _, _ = self.run_chunk(weight_decay=0.3)
        plain, _, _ = self.run_chunk(weight_decay=0.0)
        assert not np.allclose(decayed, plain, rtol=1e-6)


class TestScoreSequences:
    raw = np.array([0.3, -0.2, 0.5])

    def test_scores_drop_the_first_trial_with_exact_values(self):
        total, rows = paper.score_sequences(self.raw, golden_sequences()[:3], "sticky_rl")
        assert total == pytest.approx(5.667431188987722, rel=RTOL)
        assert [r["n_trials"] for r in rows] == [6, 5, 6]
        assert [r["n_observed_trials"] for r in rows] == [7, 6, 7]
        np.testing.assert_allclose(
            [r["nll"] for r in rows], [18.784196948704164, 34.466364171402695, 43.095769092684414]
        )
        np.testing.assert_allclose(
            [r["release_nll"] for r in rows],
            [7.6019801164222836, 18.452837525147242, 29.502075997138398],
        )
        np.testing.assert_allclose(
            [r["argmax_accuracy"] for r in rows], [5 / 6, 2 / 5, 2 / 6], rtol=RTOL
        )

    def test_uniform_policy_scores_log_three_per_scored_trial(self):
        # beta = relu(raw[1]) = 0 makes every arm equally likely, so each of the
        # n - 1 scored trials costs log(3); scoring the first trial would add one more.
        raw = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        sequence = dict(golden_sequences()[0], actions=np.array([0, 0, 1, 2, 0]))
        sequence["rewards"] = np.ones(5)
        total, (row,) = paper.score_sequences(raw, [sequence], "best_rl")
        assert row["n_trials"] == 4 and row["n_observed_trials"] == 5
        assert row["nll"] == pytest.approx(4 * np.log(3.0))
        assert total == pytest.approx(np.log(3.0))
        released = -np.log((1 - 1e-5) / 3 + 5e-4)
        assert row["release_nll"] == pytest.approx(4 * released)
        assert row["argmax_accuracy"] == pytest.approx(0.5)  # ties pick arm 0


class TestFitGolden:
    @pytest.fixture(scope="class")
    @classmethod
    def sweep(cls, tmp_path_factory):
        root = tmp_path_factory.mktemp("golden")
        result = paper.run_paper_sweep(golden_sequences(), root, **SWEEP)
        output = Path(result["output_dir"])
        records = [
            json.loads((output / f"sticky_rl_setting-00_seed-0{i}.json").read_text())
            for i in range(2)
        ]
        return result, output, records

    def test_output_file_names(self, sweep):
        _, output, _ = sweep
        assert sorted(p.name for p in output.iterdir()) == [
            "manifest.json",
            "results.json",
            "sticky_rl_selected_parameters.npz",
            "sticky_rl_setting-00_seed-00.json",
            "sticky_rl_setting-00_seed-00.npz",
            "sticky_rl_setting-00_seed-01.json",
            "sticky_rl_setting-00_seed-01.npz",
        ]

    def test_manifest_is_written_with_sorted_keys_and_names_the_directory(self, sweep):

        _, output, _ = sweep
        text = (output / "manifest.json").read_text()
        manifest = json.loads(text)
        assert list(manifest) == sorted(manifest)
        assert text == json.dumps(manifest, sort_keys=True, indent=2)
        identity = {k: v for k, v in manifest.items() if k != "implementation"}
        assert output.name == identity_hash(identity)
        assert manifest["algorithm_version"] == paper.ALGORITHM_VERSION

    def test_schedule_ends_with_the_shorter_chunk(self, sweep):
        _, _, records = sweep
        for record in records:
            assert [h["step"] for h in record["history"]] == [0, 2, 4, 5]
            assert record["step"] == 5

    def test_seeds_start_from_different_initializations(self, sweep):
        _, _, records = sweep
        # The sampling and initialization streams depend on fit_seed + repeat.
        first, second = (r["best_raw"] for r in records)
        assert not np.allclose(first, second, rtol=1e-3)
        np.testing.assert_allclose(
            first, [1.4447343841365543, 1.9700523145702895, 1.3557122797057446], rtol=RTOL
        )
        np.testing.assert_allclose(
            second, [1.1190453526562438, 1.5732846092768697, 0.9737926816251562], rtol=RTOL
        )

    def test_best_iterate_not_the_final_one_is_saved_and_returned(self, sweep):
        result, output, records = sweep
        assert [r["best_step"] for r in records] == [4, 4]
        finals = []
        for i, record in enumerate(records):
            with np.load(output / f"sticky_rl_setting-00_seed-0{i}.npz") as saved:
                finals.append(saved["raw_0"])
                # The checkpoint's best_* arrays are the step-4 iterate that the
                # record reports, and differ from the step-5 parameters it also holds.
                np.testing.assert_array_equal(saved["best_0"], record["best_raw"])
                assert json.loads(str(saved["metadata"]))["best_step"] == 4
            assert not np.allclose(finals[-1], record["best_raw"], rtol=1e-3)
        np.testing.assert_allclose(
            finals[0], [1.3907770479277912, 1.8610380876959955, 1.2771756845368059], rtol=RTOL
        )
        np.testing.assert_allclose(records[0]["parameters"]["beta"], 8.777391154337847, rtol=RTOL)
        # Selection returns the best iterate of the best seed.
        with np.load(output / "sticky_rl_selected_parameters.npz") as saved:
            np.testing.assert_array_equal(saved["parameters"], records[0]["best_raw"])
        (selected,) = result["selected"]
        assert selected["selected_step"] == 4 and selected["repeat"] == 0

    def test_test_scores_come_from_the_test_partition_not_validation(self, sweep):
        result, output, _ = sweep
        data = golden_sequences()
        parts = paper.split_by_mouse(data, 42)[0]
        with np.load(output / "sticky_rl_selected_parameters.npz") as saved:
            best_raw = saved["parameters"]
        test_ce, test_rows = paper.score_sequences(best_raw, parts["test"], "sticky_rl")
        validation_ce, _ = paper.score_sequences(best_raw, parts["validation"], "sticky_rl")
        (selected,) = result["selected"]
        assert selected["test_cross_entropy"] == pytest.approx(test_ce, rel=RTOL)
        assert selected["validation_cross_entropy"] == pytest.approx(validation_ce, rel=RTOL)
        # The two partitions score very differently here, so a mix-up cannot pass.
        assert test_ce > 1e6 * validation_ce
        assert selected["test_cross_entropy"] == pytest.approx(13.928145226143588, rel=RTOL)
        assert selected["test_release_score"] == pytest.approx(3.504774971172379, rel=RTOL)
        assert selected["validation_cross_entropy"] == pytest.approx(1.6172122555585536e-09)
        assert {r["mouse"] for r in result["test_sessions"]} == {s["mouse"] for s in parts["test"]}
        assert len(result["test_sessions"]) == len(parts["test"]) == 3


def fail_on_chunk(monkeypatch, fail_at):
    """Make the ``fail_at``-th training chunk report a nonfinite loss (an interruption)."""
    real_make_backend = paper.make_backend
    calls = []

    def make_backend(*args, **kwargs):
        stats, init, make_trainer = real_make_backend(*args, **kwargs)

        def patched_make_trainer(setting):
            optimizer, chunk = make_trainer(setting)

            def failing_chunk(*chunk_args):
                calls.append(1)
                state, loss = chunk(*chunk_args)
                return state, (np.float64("nan") if len(calls) == fail_at else loss)

            return optimizer, failing_chunk

        return stats, init, patched_make_trainer

    monkeypatch.setattr(paper, "make_backend", make_backend)


class TestResumeAfterValidationGotWorse:
    """A checkpoint whose best iterate differs from its current iterate must resume exactly."""

    sweep = dict(SWEEP, n_seeds=1, max_steps=7)  # chunks end at steps 2, 4, 6, 7

    def test_resumed_run_keeps_the_best_iterate_not_the_current_one(self, tmp_path, monkeypatch):
        fail_on_chunk(monkeypatch, fail_at=4)
        with pytest.raises(FloatingPointError, match="nonfinite objective"):
            paper.run_paper_sweep(golden_sequences(), tmp_path, **self.sweep)
        (output,) = tmp_path.iterdir()
        with np.load(output / "sticky_rl_setting-00_seed-00.npz") as saved:
            metadata = json.loads(str(saved["metadata"]))
            current, best = saved["raw_0"], saved["best_0"]
        # The checkpoint was saved at step 6, after validation got worse than at step 4.
        assert metadata["step"] == 6 and metadata["best_step"] == 4
        assert not np.allclose(current, best, rtol=1e-3)
        np.testing.assert_allclose(
            best, [1.4447343841365543, 1.9700523145702895, 1.3557122797057446], rtol=RTOL
        )
        np.testing.assert_allclose(
            current, [1.3029058811565277, 1.721064109471238, 1.1768798670042278], rtol=RTOL
        )

        monkeypatch.undo()
        result = paper.run_paper_sweep(golden_sequences(), tmp_path, **self.sweep)
        record = json.loads((output / "sticky_rl_setting-00_seed-00.json").read_text())
        assert record["best_step"] == 4 and record["step"] == 7
        np.testing.assert_allclose(record["best_raw"], best, rtol=RTOL)
        (selected,) = result["selected"]
        assert selected["selected_step"] == 4
        assert selected["test_cross_entropy"] == pytest.approx(13.928145226143588, rel=RTOL)


class TestScoringFailure:
    def test_earlier_winners_keep_their_files_and_no_results_are_written(
        self, tmp_path, monkeypatch
    ):
        real_score = paper.score_sequences

        def failing_score(raw, sequences, model, statistics=None, *args, **kwargs):
            # Only the final held-out scoring omits ``statistics``.
            if model == "best_rl" and statistics is None:
                raise RuntimeError("scoring failed")
            return real_score(raw, sequences, model, statistics, *args, **kwargs)

        monkeypatch.setattr(paper, "score_sequences", failing_score)
        models = ("sticky_rl", "best_rl", "released_rl")
        with pytest.raises(RuntimeError, match="scoring failed"):
            paper.run_paper_sweep(
                golden_sequences(), tmp_path, **dict(SWEEP, models=models, n_seeds=1)
            )
        (output,) = tmp_path.iterdir()
        written = sorted(p.name for p in output.glob("*_selected_parameters.npz"))
        assert written == ["sticky_rl_selected_parameters.npz"]
        assert not (output / "results.json").exists()


class TestRunDirectoryIdentity:
    """The directory follows the algorithm version and configuration, not the source text."""

    def sweep(self, root, **overrides):
        return paper.run_paper_sweep(golden_sequences(), root, **{**SWEEP, **overrides})

    def test_editing_source_text_resumes_in_the_same_directory(
        self, tmp_path, edit_module_text, monkeypatch
    ):
        first = self.sweep(tmp_path)
        fits = []
        fit_run = paper._fit_run
        monkeypatch.setattr(paper, "_fit_run", lambda *a, **k: fits.append(a) or fit_run(*a, **k))
        original = json.loads((Path(first["output_dir"]) / "manifest.json").read_text())
        url = 'SOURCE_URL = "https://github.com/google-deepmind/hybrid_rnns_reward_learning"'
        edit_module_text(paper, url, url + "  # note")
        with pytest.warns(ProvenanceMismatchWarning, match="implementation"):
            second = self.sweep(tmp_path)
        assert second == first and second["output_dir"] == first["output_dir"]
        assert fits == []  # every run was loaded from disk
        manifest = json.loads((Path(first["output_dir"]) / "manifest.json").read_text())
        assert manifest["implementation"] == original["implementation"]

    def test_settings_data_and_version_changes_start_a_new_directory(self, tmp_path, monkeypatch):
        first = self.sweep(tmp_path, max_new_runs=0)
        assert self.sweep(tmp_path, max_new_runs=0)["output_dir"] == first["output_dir"]
        other_grid = [dict(SWEEP["grid"][0], batch_size=2)]
        changed = [
            self.sweep(tmp_path, max_new_runs=0, fit_seed=6),
            self.sweep(tmp_path, max_new_runs=0, grid=other_grid),
            self.sweep(tmp_path, max_new_runs=0, max_steps=SWEEP["max_steps"] + 1),
            self.sweep(tmp_path, max_new_runs=0, n_seeds=3),
            self.sweep(tmp_path, max_new_runs=0, validation_every=3),
        ]
        towers = [dict(s, towers=["A", "B", "Z"]) for s in golden_sequences()]
        changed.append(paper.run_paper_sweep(towers, tmp_path, max_new_runs=0, **SWEEP))
        dirs = {first["output_dir"], *(r["output_dir"] for r in changed)}
        assert len(dirs) == len(changed) + 1
        monkeypatch.setattr(paper, "ALGORITHM_VERSION", paper.ALGORITHM_VERSION + 1)
        assert self.sweep(tmp_path)["output_dir"] != first["output_dir"]
