"""Unit tests for the small helpers behind ``make_backend`` and ``run_paper_sweep``."""

import importlib.util

import numpy as np
import pytest

from src.agents import paper_rl_fitting as paper

needs_jax = pytest.mark.skipif(
    not (importlib.util.find_spec("jax") and importlib.util.find_spec("optax")),
    reason="Optional JAX/Optax dependencies not installed",
)


def make_sequences(n_mice=6, problems=(5, 6, 7), n_trials=5):
    return [
        dict(
            mouse=f"m{i}",
            problem=p,
            session=f"ses-{p}",
            towers=["A", "B", "C"],
            actions=np.array([0, 1, 2, 1, 0, 2][:n_trials], dtype=int),
            rewards=np.array([1.0, 2.0, 4.0, 0.0, 1.0, 3.0][:n_trials]),
        )
        for i in range(n_mice)
        for p in problems
    ]


class TestSplitRLParameters:
    def test_sticky_fixes_absent_terms_at_zero(self):
        rates = paper.split_rl_parameters([0.1, 2.0, 0.5], "sticky_rl")
        assert rates == (0.1, 2.0, 0.5, 0.0, 0.0, 0.0)

    def test_released_has_no_bias(self):
        rates = paper.split_rl_parameters([0.1, 2.0, 0.5, 0.2, 1.5], "released_rl")
        assert rates == (0.1, 2.0, 0.5, 0.0, 0.2, 1.5)

    def test_best_uses_all_six_in_order(self):
        rates = paper.split_rl_parameters([1, 2, 3, 4, 5, 6], "best_rl")
        assert list(rates) == [1, 2, 3, 4, 5, 6]
        assert rates.stickiness == 3 and rates.q_init == 6

    @pytest.mark.parametrize("model", paper.MODEL_NAMES)
    def test_wrong_length_fails_loudly(self, model):
        with pytest.raises(ValueError):
            paper.split_rl_parameters([0.1, 0.2], model)


class TestWeightDecay:
    def test_weight_decay_wins_over_legacy_key(self):
        assert paper._weight_decay(dict(weight_decay=0.1, l2_weight=0.9)) == 0.1

    def test_legacy_key_is_used_when_weight_decay_absent(self):
        assert paper._weight_decay(dict(l2_weight=0.3)) == 0.3

    def test_defaults_to_zero(self):
        assert paper._weight_decay({}) == 0.0


@needs_jax
class TestBackend:
    def setup_method(self):
        self.packed = paper.pack_sequences(make_sequences(n_mice=1, n_trials=5))
        self.raw = np.linspace(-0.5, 0.5, 3)

    def test_unknown_model_fails_loudly(self):
        with pytest.raises(ValueError, match="Unknown model"):
            paper.make_backend("not_a_model")

    def test_skip_first_drops_exactly_the_first_trial(self):
        stats, _, _ = paper.make_backend("sticky_rl")
        full, _ = stats(self.raw, *self.packed)
        shifted, _ = stats(self.raw, *self.packed, False, True)
        assert np.all(np.asarray(shifted) < np.asarray(full))
        # Dropping the first trial removes exactly its negative log-probability,
        # which for three arms at q=0 and no previous choice is log(3).
        np.testing.assert_allclose(np.asarray(full - shifted), np.log(3.0), atol=1e-6)

    def test_released_loss_is_bounded_by_the_probability_floor(self):
        stats, _, _ = paper.make_backend("sticky_rl")
        raw = np.array(
            [8.0, 8.0, -8.0]
        )  # near-deterministic policy, so some choices are surprising
        plain, _ = stats(raw, *self.packed, False, True)
        released, _ = stats(raw, *self.packed, True, True)
        n_scored = self.packed[2][:, 1:].sum()
        assert float(plain[0]) > float(released[0])
        assert float(released[0]) <= n_scored * -np.log(5e-4)

    def test_fully_masked_session_contributes_nothing(self):
        stats, _, _ = paper.make_backend("best_rl")
        actions, rewards, mask = self.packed
        nll, correct = stats(np.zeros(6), actions, rewards, np.zeros_like(mask))
        assert float(nll[0]) == 0.0 and float(correct[0]) == 0.0

    def test_trainer_chunk_advances_key_and_accepts_legacy_decay_key(self):
        import jax
        import jax.numpy as jnp

        _, _, make_trainer = paper.make_backend("sticky_rl")
        optimizer, chunk = make_trainer(
            dict(optimizer_learning_rate=0.01, l2_weight=0.5, batch_size=1)
        )
        raw = jnp.asarray(self.raw)
        arrays = tuple(jnp.asarray(x) for x in self.packed)
        (new_raw, _, new_key), loss = chunk(
            raw, optimizer.init(raw), jax.random.PRNGKey(0), *arrays, 2
        )
        assert new_raw.shape == raw.shape
        assert np.isfinite(float(loss))
        assert not np.array_equal(np.asarray(new_key), np.asarray(jax.random.PRNGKey(0)))


class TestValidation:
    def test_models_accepts_distinct_supported_names(self):
        paper.validate_models(("sticky_rl", "best_rl"), paper.MODEL_NAMES)

    @pytest.mark.parametrize("models", [(), ("sticky_rl", "sticky_rl"), ("sticky_rl", "typo")])
    def test_models_rejects_empty_duplicate_and_unknown(self, models):
        with pytest.raises(ValueError, match="unique supported model names"):
            paper.validate_models(models, paper.MODEL_NAMES)

    def test_sizes_accept_zero_budget_and_no_budget(self):
        grid = [dict(optimizer_learning_rate=0.1, batch_size=2)]
        paper.validate_sweep_sizes(grid, 1, 1, 1, 0)
        paper.validate_sweep_sizes(grid, 1, 1, 1, None)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(grid=[]),
            dict(n_seeds=0),
            dict(max_steps=0),
            dict(validation_every=0),
            dict(max_new_runs=-1),
        ],
    )
    def test_sizes_reject_empty_or_nonpositive(self, kwargs):
        args = dict(
            grid=[dict(optimizer_learning_rate=0.1, batch_size=2)],
            n_seeds=1,
            max_steps=1,
            validation_every=1,
            max_new_runs=None,
        )
        with pytest.raises(ValueError, match="Invalid sweep sizes"):
            paper.validate_sweep_sizes(**{**args, **kwargs})

    def test_settings_accept_legacy_decay_key_and_hidden_units(self):
        paper.validate_settings(
            [dict(optimizer_learning_rate=0.1, l2_weight=0.0, batch_size=1, hidden_units=1)]
        )

    @pytest.mark.parametrize(
        "bad",
        [
            dict(optimizer_learning_rate=0),
            dict(weight_decay=-1e-9),
            dict(l2_weight=-1.0),
            dict(batch_size=0),
            dict(hidden_units=0),
        ],
    )
    def test_settings_reject_out_of_range_values(self, bad):
        setting = {**dict(optimizer_learning_rate=0.1, batch_size=2), **bad}
        with pytest.raises(ValueError, match="Invalid optimizer setting"):
            paper.validate_settings([setting])

    def test_settings_missing_required_key_fails_loudly(self):
        with pytest.raises(KeyError):
            paper.validate_settings([dict(batch_size=2)])

    def test_sequence_lengths_reject_single_trial_sessions(self):
        short = dict(make_sequences(1)[0], actions=np.array([0]), rewards=np.array([1.0]))
        with pytest.raises(ValueError, match="at least two"):
            paper.validate_sequence_lengths([*make_sequences(1), short])
        paper.validate_sequence_lengths([])


class TestSplitSequences:
    def test_mouse_split_by_default(self):
        parts, identities, details = paper.split_sequences(make_sequences(), 42)
        assert details == dict(split_strategy="mouse", split_problems=None)
        assert sum(map(len, parts.values())) == 18
        assert identities == paper.split_by_mouse(make_sequences(), 42)[1]

    def test_problem_split_lists_mice_per_partition(self):
        parts, identities, details = paper.split_sequences(
            make_sequences(), 42, test_problem=6, validation_problem=7
        )
        assert details["split_strategy"] == "pooled_problem"
        assert details["split_problems"] == dict(train=[5], validation=[7], test=[6])
        assert identities == {part: [f"m{i}" for i in range(6)] for part in parts}

    def test_validation_problem_without_test_problem_fails_loudly(self):
        with pytest.raises(ValueError, match="requires test_problem"):
            paper.split_sequences(make_sequences(), 42, validation_problem=5)


class TestFingerprintSequences:
    def test_is_deterministic_and_ignores_array_dtype(self):
        data = make_sequences(2)
        as_int32 = [dict(s, actions=s["actions"].astype(np.int32)) for s in data]
        assert paper.fingerprint_sequences(data) == paper.fingerprint_sequences(as_int32)

    def test_changes_with_rewards_order_and_identity(self):
        data = make_sequences(2)
        base = paper.fingerprint_sequences(data)
        first = paper.fingerprint_sequences(data[:1])
        assert paper.fingerprint_sequences(data[::-1]) != base
        assert paper.fingerprint_sequences([dict(data[0], rewards=data[0]["rewards"] + 1)]) != first
        assert paper.fingerprint_sequences([dict(data[0], mouse="other")]) != first

    def test_empty_input_is_the_empty_sha256(self):
        assert paper.fingerprint_sequences([]).startswith("e3b0c44298fc1c14")


class TestManifest:
    def manifest(self, **overrides):
        settings = dict(
            profile="p",
            models=["sticky_rl"],
            grid=[{}],
            n_seeds=1,
            max_steps=2,
            validation_every=1,
            split_seed=1,
            fit_seed=2,
        )
        args = dict(
            settings=settings,
            split_details=dict(split_strategy="mouse", split_problems=None),
            identities={"train": ["a"]},
            data_sha256="abc",
            versions=dict(jax="0"),
            snapshots=dict(implementation="src"),
        )
        return paper.build_manifest(**{**args, **overrides})

    def test_collects_settings_split_data_and_code_identity(self):
        manifest = self.manifest()
        assert manifest["profile"] == "p" and manifest["split_strategy"] == "mouse"
        assert manifest["split_mice"] == {"train": ["a"]}
        assert manifest["data_sha256"] == "abc" and manifest["implementation"] == "src"
        assert manifest["upstream_commit"] == paper.SOURCE_COMMIT

    def test_overlapping_keys_fail_loudly_instead_of_overwriting(self):
        with pytest.raises(TypeError):
            self.manifest(snapshots=dict(profile="shadow"))

    def test_output_directory_is_named_by_manifest_hash(self, tmp_path):
        import hashlib
        import json

        manifest = self.manifest()
        output = paper.write_manifest(tmp_path / "new" / "root", manifest)
        text = (output / "manifest.json").read_text()
        assert json.loads(text) == manifest
        assert output.name == hashlib.sha256(text.encode()).hexdigest()[:12]
        assert paper.write_manifest(tmp_path / "new" / "root", manifest) == output

    def test_different_manifests_get_different_directories(self, tmp_path):
        first = paper.write_manifest(tmp_path, self.manifest())
        second = paper.write_manifest(tmp_path, self.manifest(data_sha256="other"))
        assert first != second


class TestRunMetadata:
    def test_release_score_pools_trials_not_sessions(self):
        rows = [dict(release_nll=6.0, n_trials=2), dict(release_nll=4.0, n_trials=8)]
        assert paper.release_score_per_trial(rows) == 1.0

    def test_release_score_of_no_sessions_fails_loudly(self):
        with pytest.raises(ZeroDivisionError):
            paper.release_score_per_trial([])

    def test_initial_metadata_starts_at_step_zero(self):
        metadata = paper.initial_metadata(1.5, 2.5)
        assert (metadata["step"], metadata["best_step"], metadata["best_validation"]) == (0, 0, 2.5)
        assert metadata["history"] == [
            dict(step=0, validation_cross_entropy=1.5, validation_release_score=2.5)
        ]

    def test_improvement_updates_best_and_does_not_mutate_input(self):
        before = paper.initial_metadata(1.5, 2.5)
        snapshot = repr(before)
        after, improved = paper.advance_metadata(before, 3, 1.0, 2.0, 0.7)
        assert improved
        assert (after["step"], after["best_step"], after["best_validation"]) == (3, 3, 2.0)
        assert after["history"][-1] == dict(
            step=3,
            validation_cross_entropy=1.0,
            validation_release_score=2.0,
            last_batch_objective=0.7,
        )
        assert repr(before) == snapshot
        assert list(after) == list(before)

    @pytest.mark.parametrize("validation", [2.5, 3.0])
    def test_tie_or_worse_keeps_the_earlier_best(self, validation):
        before = paper.initial_metadata(1.5, 2.5)
        after, improved = paper.advance_metadata(before, 2, 1.0, validation, 0.1)
        assert not improved
        assert (after["step"], after["best_step"], after["best_validation"]) == (2, 0, 2.5)
        assert len(after["history"]) == 2

    def test_steps_accumulate_across_calls(self):
        metadata = paper.initial_metadata(1.0, 1.0)
        for _ in range(3):
            metadata, _ = paper.advance_metadata(metadata, 5, 1.0, 1.0, 0.0)
        assert metadata["step"] == 15 and len(metadata["history"]) == 4


class TestSelection:
    records = [
        dict(model="a", best_validation=2.0, tag=0),
        dict(model="a", best_validation=1.0, tag=1),
        dict(model="a", best_validation=1.0, tag=2),
        dict(model="b", best_validation=0.1, tag=3),
    ]

    def test_lowest_validation_wins_within_the_model(self):
        assert paper.select_winner(self.records, "a")["tag"] == 1

    def test_model_without_records_fails_loudly(self):
        with pytest.raises(ValueError):
            paper.select_winner(self.records, "c")

    def test_sweep_result_is_complete_only_when_all_runs_exist(self, tmp_path):
        details = dict(split_strategy="mouse", split_problems=None)
        complete = paper._sweep_result([1, 2], 2, tmp_path, {"train": []}, details, [], [])
        partial = paper._sweep_result([1, 2], 3, tmp_path, {"train": []}, details, [], [])
        assert complete["complete"] and not partial["complete"]
        assert partial["completed_runs"] == 2 and partial["expected_runs"] == 3
        assert partial["output_dir"] == str(tmp_path)


@needs_jax
class TestRawParameters:
    def test_serialize_restore_round_trip_for_array_and_dict(self):
        import jax.numpy as jnp

        array = jnp.array([1.0, -2.5, 0.0])
        np.testing.assert_array_equal(paper.restore_raw(paper.serialize_raw(array)), array)
        tree = dict(w=jnp.ones((2, 2)), b=jnp.zeros(2))
        restored = paper.restore_raw(paper.serialize_raw(tree))
        assert set(restored) == {"w", "b"}
        np.testing.assert_array_equal(restored["w"], tree["w"])

    def test_serialized_form_is_plain_json(self):
        import json

        import jax.numpy as jnp

        json.dumps(paper.serialize_raw(jnp.arange(3.0)), allow_nan=False)
        json.dumps(paper.serialize_raw(dict(w=jnp.ones((2, 2)))), allow_nan=False)

    def test_initial_parameters_have_model_specific_sizes(self):
        import jax

        key = jax.random.PRNGKey(0)
        sizes = {m: paper.initial_raw_parameters(m, key).shape for m in paper.MODEL_NAMES}
        assert sizes == dict(sticky_rl=(3,), best_rl=(6,), released_rl=(5,))

    def test_rl_initial_parameters_share_the_q_init_draw_across_models(self):
        import jax

        key = jax.random.PRNGKey(7)
        best = paper.initial_raw_parameters("best_rl", key)
        released = paper.initial_raw_parameters("released_rl", key)
        assert best[-1] == released[-1]
        np.testing.assert_array_equal(best[:5], [1.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(released[:4], [1.0, 1.0, 0.0, 0.0])

    def test_neural_models_get_a_parameter_tree(self):
        import jax

        tree = paper.initial_raw_parameters("rl_ann", jax.random.PRNGKey(0), hidden_units=4)
        assert isinstance(tree, dict) and tree

    def test_summary_names_the_transformed_rl_parameters(self):
        summary = paper.summarize_parameters(np.zeros(3), "sticky_rl")
        assert list(summary) == paper.PARAMETER_NAMES["sticky_rl"]
        assert summary["learning_rate"] == pytest.approx(0.001 + 0.999 * 0.5)

    def test_save_selected_parameters_array_and_dict(self, tmp_path):
        paper.save_selected_parameters(tmp_path / "a.npz", np.arange(3.0))
        paper.save_selected_parameters(tmp_path / "d.npz", dict(w=np.ones(2), b=np.zeros(1)))
        with np.load(tmp_path / "a.npz") as saved:
            assert list(saved.files) == ["parameters"]
            np.testing.assert_array_equal(saved["parameters"], np.arange(3.0))
        with np.load(tmp_path / "d.npz") as saved:
            assert sorted(saved.files) == ["b", "w"]


@needs_jax
class TestRunLoop:
    def setup_method(self):
        import jax.numpy as jnp

        parts, _, _ = paper.split_sequences(make_sequences(), 0)
        train = tuple(jnp.asarray(x) for x in paper.pack_sequences(parts["train"]))
        self.parts = parts
        self.config = paper._FitConfig(parts, train, fit_seed=1, max_steps=4, validation_every=2)
        self.setting = dict(optimizer_learning_rate=0.01, weight_decay=0.0, batch_size=2)

    def backend(self):
        stats, _, make_trainer = paper.make_backend("sticky_rl")
        return (stats, *make_trainer(self.setting))

    def test_nonfinite_objective_stops_without_writing_a_checkpoint(self, tmp_path):
        stats, optimizer, _ = self.backend()

        def diverging_chunk(raw, state, key, *rest):
            return (raw, state, key), np.float64("nan")

        checkpoint = tmp_path / "run.npz"
        with pytest.raises(FloatingPointError, match="run-x: nonfinite objective"):
            paper._fit_run(
                "run-x",
                "sticky_rl",
                self.setting,
                0,
                (stats, optimizer, diverging_chunk),
                self.config,
                checkpoint,
            )
        assert not checkpoint.exists()

    def test_finished_checkpoint_is_resumed_without_more_training(self, tmp_path):
        backend = self.backend()
        checkpoint = tmp_path / "run.npz"
        args = ("r", "sticky_rl", self.setting, 0)
        best_raw, metadata = paper._fit_run(*args, backend, self.config, checkpoint)
        assert metadata["step"] == 4 and len(metadata["history"]) == 3

        def forbidden_chunk(*args):
            raise AssertionError("a finished run must not train again")

        stats, optimizer, _ = backend
        resumed_raw, resumed = paper._fit_run(
            *args, (stats, optimizer, forbidden_chunk), self.config, checkpoint
        )
        assert resumed == metadata
        np.testing.assert_array_equal(resumed_raw, best_raw)

    def test_budget_of_zero_trains_nothing_and_returns_finished_runs(self, tmp_path):
        import json

        grid = [self.setting]
        assert paper._sweep_fits(("sticky_rl",), grid, 2, self.config, tmp_path, 0) == []
        done = dict(model="sticky_rl", marker="from disk")
        (tmp_path / "sticky_rl_setting-00_seed-01.json").write_text(json.dumps(done))
        records = paper._sweep_fits(("sticky_rl",), grid, 2, self.config, tmp_path, 0)
        assert records == [done]
        assert not list(tmp_path.glob("*.npz"))

    def test_budget_limits_new_fits(self, tmp_path):
        records = paper._sweep_fits(("sticky_rl",), [self.setting], 3, self.config, tmp_path, 1)
        assert [r["repeat"] for r in records] == [0]
        assert (tmp_path / "sticky_rl_setting-00_seed-00.json").exists()
        assert not (tmp_path / "sticky_rl_setting-00_seed-01.json").exists()
        assert records[0]["effective_batch_size"] == 2

    def test_batch_size_is_capped_at_the_number_of_training_sessions(self, tmp_path):
        big = dict(self.setting, batch_size=10_000)
        records = paper._sweep_fits(("sticky_rl",), [big], 1, self.config, tmp_path, None)
        assert records[0]["effective_batch_size"] == len(self.parts["train"])
        assert records[0]["setting"]["batch_size"] == 10_000
