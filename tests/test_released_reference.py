"""Numerical comparisons against unmodified, commit-pinned released Haiku cores."""
import importlib.util
import unittest
import numpy as np
from src.agents import neural_bandit as neural
from src.agents import paper_rl_fitting as paper

AVAILABLE = all(importlib.util.find_spec(x) is not None for x in ['jax', 'optax', 'haiku', 'ml_collections'])


def reference(model, k, h):
    import haiku as hk
    from tests.reference.hybrid_rnns_reward_learning.rnn_config import get_config
    from tests.reference.hybrid_rnns_reward_learning.cogmod import CogMod
    from tests.reference.hybrid_rnns_reward_learning.bi_rnn import BiRNN
    from tests.reference.hybrid_rnns_reward_learning.rnn import RNN
    c = get_config()
    c.network_params.n_actions, c.network_params.hidden_size = k, h
    rp = c.rnn_rl_params
    rp.w_v = rp.w_h = 1.
    rp.fit_forget = True
    rp.fit_init_v = rp.fit_init_h = True
    rp.s = model in ('memory_ann', 'vanilla_rnn')
    rp.o = model == 'context_ann'
    rp.zero_values = model == 'memory_ann'
    if model == 'released_rl':
        rp.fit_alpha = rp.fit_beta = rp.fit_persev_t = rp.fit_bias = True
        rp.fit_init_h = rp.fit_persev_p = False
    def forward(inputs):
        cls = CogMod if model == 'released_rl' else RNN if model == 'vanilla_rnn' else BiRNN
        core = cls(rp, c.network_params)
        return hk.dynamic_unroll(core, inputs, core.initial_state(inputs.shape[0]),
                                 time_major=False, return_all_states=True)
    return hk.transform(forward)


def reference_parameters(p, model):
    import jax.numpy as jnp
    if model == 'released_rl':
        return {'cog_mod': {name: jnp.reshape(p[i], (1,)) for i, name in enumerate([
            'unsigmoid_alpha', 'unrelued_beta', 'untanh_persev_t', 'unsigmoid_forget', 'init_value_v'])}}
    if model == 'vanilla_rnn':
        return {'rnn/linear': {'w': p['joint_w1'], 'b': p['joint_b1']},
                'rnn/linear_1': {'w': p['joint_w2'], 'b': p['joint_b2']}}
    out = {'bi_rnn': {'init_value_v': p['q_init'].reshape(1),
                       'init_value_h': p['c_init'].reshape(1),
                       'unsigmoid_forget': p['forgetting_raw'].reshape(1)}}
    for prefix, module in [('reward', 'value'), ('action', 'habit')]:
        for layer in [1, 2]:
            name = f'bi_rnn/~_{module}_rnn/linear'+('_1' if layer == 2 else '')
            out[name] = {'w': p[f'{prefix}_w{layer}'], 'b': p[f'{prefix}_b{layer}']}
    return out


@unittest.skipUnless(AVAILABLE, 'Optional reference-test dependencies missing')
class ReleasedReferenceTests(unittest.TestCase):
    def test_predictions_losses_gradients_and_adamw_match_upstream(self):
        import jax
        import jax.numpy as jnp
        import optax
        for k in [3, 4]:
            actions = jnp.array([[0, 1, 2, 0, 1], [1, 1, 0, 2, 0]])
            rewards = jnp.array([[1., 4., 0., 2., 3.], [3., 0., 2., 1., 0.]])
            inputs = jnp.concatenate([jax.nn.one_hot(actions, k), rewards[:, :, None]], axis=-1)
            mask = jnp.ones_like(actions, dtype=bool)
            for model in (*neural.NEURAL_MODELS, 'released_rl'):
                with self.subTest(k=k, model=model):
                    p = (jnp.array([.8, 1.3, -.2, -.9, .7]) if model == 'released_rl' else
                         neural.initialize_parameters(jax.random.PRNGKey(9), model, 4, k))
                    ref = reference(model, k, 4)
                    template = ref.init(jax.random.PRNGKey(0), inputs)
                    translated = reference_parameters(p, model)
                    self.assertEqual(jax.tree_util.tree_structure(template), jax.tree_util.tree_structure(translated))
                    reference_probs, reference_states = ref.apply(translated, None, inputs)
                    if model != 'released_rl':
                        state = neural.initial_state(p, model, 2, k)
                        local_probs, states = [], []
                        for t in range(actions.shape[1]):
                            state = neural.observe(p, state, model, actions[:, t], rewards[:, t])
                            local_probs.append(jax.nn.softmax(neural.choice_logits(p, state, model), axis=-1))
                            states.append(state)
                        np.testing.assert_allclose(jnp.stack(local_probs, axis=1), reference_probs, atol=1e-12)
                        names = ('joint_hidden',) if model == 'vanilla_rnn' else ('action_hidden', 'reward_hidden', 'c', 'q')
                        refs = (reference_states[1],) if model == 'vanilla_rnn' else reference_states
                        for name, expected in zip(names, refs):
                            np.testing.assert_allclose(jnp.stack([s[name] for s in states], axis=1), expected, atol=1e-12)
                    else:
                        from src.agents.released_rl import sequence_log_probabilities
                        physical = paper.transform_parameters(p, model)
                        for b in range(2):
                            sequence = dict(actions=np.asarray(actions[b]), rewards=np.asarray(rewards[b]), towers=list(range(k)))
                            lp = sequence_log_probabilities(physical, sequence)
                            np.testing.assert_allclose(np.exp(lp[1:]), reference_probs[b, :-1], atol=1e-12)
                        # The example requests fit_bias, yet CogMod creates only five parameters.
                        self.assertEqual(sum(v.size for v in jax.tree_util.tree_leaves(template)), 5)
                    def ref_loss(params):
                        probs, _ = ref.apply(reference_parameters(params, model), None, inputs)
                        probs = (1-1e-5)*probs+5e-4
                        return -jnp.sum(jnp.log(probs[:, :-1])*inputs[:, 1:, :k])/inputs.shape[0]
                    _, objective, trainer = paper.make_backend(model, num_arms=k, hidden_units=4)
                    local_loss = lambda params: objective(params, actions, rewards, mask, 0.)
                    np.testing.assert_allclose(local_loss(p), ref_loss(p), atol=1e-12)
                    actual_grad, expected_grad = jax.grad(local_loss)(p), jax.grad(ref_loss)(p)
                    for actual, expected in zip(jax.tree_util.tree_leaves(actual_grad), jax.tree_util.tree_leaves(expected_grad)):
                        np.testing.assert_allclose(actual, expected, atol=1e-10)
                    setting = dict(optimizer_learning_rate=.001, weight_decay=.0001, batch_size=2)
                    optimizer, chunk = trainer(setting)
                    state = optimizer.init(p)
                    (actual_params, _, _), _ = chunk(p, state, jax.random.PRNGKey(1), actions, rewards, mask, 1)
                    updates, _ = optimizer.update(expected_grad, state, p)
                    expected_params = optax.apply_updates(p, updates)
                    for actual, expected in zip(jax.tree_util.tree_leaves(actual_params), jax.tree_util.tree_leaves(expected_params)):
                        np.testing.assert_allclose(actual, expected, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
