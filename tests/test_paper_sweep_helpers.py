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
