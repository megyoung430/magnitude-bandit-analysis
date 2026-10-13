import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from src.agents import neural_bandit as neural
from src.agents import paper_rl_fitting as paper
from tests.test_paper_rl_fitting import sequences


@unittest.skipUnless(importlib.util.find_spec('jax') and importlib.util.find_spec('optax'),
                     'Optional JAX/Optax dependencies not installed')
class NeuralBanditTests(unittest.TestCase):
    def parameters(self, model, h=4, k=3):
        import jax
        return neural.initialize_parameters(jax.random.PRNGKey(2), model, h, k)

    def test_parameter_counts_and_finite_gradients(self):
        import jax
        import jax.numpy as jnp
        expected = dict(rl_ann=216, context_ann=1352, memory_ann=2472, vanilla_rnn=4740)
        widths = dict(rl_ann=16, context_ann=64, memory_ann=32, vanilla_rnn=64)
        arrays = paper.pack_sequences(sequences()[:2])
        for model in neural.NEURAL_MODELS:
            summary = neural.parameter_summary(self.parameters(model, widths[model], 4), model)
            self.assertEqual(summary['n_parameters'], expected[model])
            p = self.parameters(model)
            _, objective, _ = neural.make_backend(model, 4)
            value, grad = jax.value_and_grad(objective)(p, *arrays, 1e-5)
            self.assertTrue(np.isfinite(float(value)))
            self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in grad.values()))
            # Differentiate one representative network weight independently.
            key = 'joint_w1' if model == 'vanilla_rnn' else 'reward_w1'
            plus, minus = dict(p), dict(p)
            plus[key] = p[key].at[0, 0].add(1e-5)
            minus[key] = p[key].at[0, 0].add(-1e-5)
            numeric = (float(objective(plus, *arrays, 1e-5))-float(objective(minus, *arrays, 1e-5)))/2e-5
            self.assertAlmostEqual(float(grad[key][0, 0]), numeric, places=6)

    def test_causal_predictions_padding_and_resets(self):
        seq = sequences()[0]
        changed = dict(seq, rewards=seq['rewards'].copy(), actions=seq['actions'].copy())
        changed['rewards'][-2:] = 100
        changed['actions'][-2:] = 2
        for model in neural.NEURAL_MODELS:
            p = self.parameters(model)
            trace = neural.trace_sequence(p, seq, model)
            other = neural.trace_sequence(p, changed, model)
            np.testing.assert_allclose(trace['log_probabilities'][:5], other['log_probabilities'][:5])
            stats, _, _ = neural.make_backend(model, 4)
            expected = -trace['log_probabilities'][np.arange(6), seq['actions']].sum()
            packed = paper.pack_sequences([seq, seq], max_trials=12)
            packed[1][~packed[2]] = 999
            np.testing.assert_allclose(stats(p, *packed)[0], [expected, expected])
            np.testing.assert_allclose(np.exp(trace['log_probabilities'][0]), 1/3)

    def test_hybrid_channel_separation_and_unchosen_decay(self):
        import jax.numpy as jnp
        seq = sequences()[0]
        changed_rewards = dict(seq, rewards=100+seq['rewards'])
        for model in neural.NEURAL_MODELS[:3]:
            p = self.parameters(model)
            trace = neural.trace_sequence(p, seq, model)
            other = neural.trace_sequence(p, changed_rewards, model)
            np.testing.assert_allclose(trace['pre_c'], other['pre_c'])
            np.testing.assert_allclose(trace['pre_action_hidden'], other['pre_action_hidden'])
            f = neural.parameter_summary(p, model)['forgetting']
            for t, action in enumerate(seq['actions']):
                unchosen = np.arange(3) != action
                np.testing.assert_allclose(trace['post_q'][t, unchosen],
                    (1-f)*trace['pre_q'][t, unchosen]+f*float(p['q_init']))
            state = neural.initial_state(p, model)
            modified = dict(state, reward_hidden=state['reward_hidden']+1,
                            action_hidden=state['action_hidden']+1,
                            c=state['c']+1, q=state['q']+1)
            actions, rewards = jnp.array([0]), jnp.array([2.])
            first = neural.observe(p, state, model, actions, rewards)
            second = neural.observe(p, modified, model, actions, rewards)
            if model == 'rl_ann':
                np.testing.assert_allclose(first['c'], second['c'])
            if model == 'memory_ann':
                same_rewards_different_actions = dict(seq, actions=(seq['actions']+1)%3)
                alternate = neural.trace_sequence(p, same_rewards_different_actions, model)
                np.testing.assert_allclose(trace['pre_reward_hidden'], alternate['pre_reward_hidden'])

    def test_vanilla_and_memory_use_hidden_recurrence(self):
        import jax.numpy as jnp
        for model, field in [('vanilla_rnn', 'joint_hidden'), ('memory_ann', 'reward_hidden')]:
            p = self.parameters(model)
            state = neural.initial_state(p, model)
            changed = dict(state, **{field: state[field]+1})
            first = neural.observe(p, state, model, jnp.array([0]), jnp.array([2.]))
            second = neural.observe(p, changed, model, jnp.array([0]), jnp.array([2.]))
            self.assertFalse(np.allclose(first[field], second[field]))

    def test_simulation_adapter_and_controlled_probes(self):
        import jax.numpy as jnp
        seq = sequences()[0]
        for model in neural.NEURAL_MODELS:
            p = self.parameters(model)
            trace = neural.trace_sequence(p, seq, model)
            agent = neural.NeuralBanditMouse(p, model)
            for t, (a, r) in enumerate(zip(seq['actions'], seq['rewards'])):
                np.testing.assert_allclose(neural.choice_logits(p, agent.state, model)[0], trace['logits'][t])
                agent.update_q_values(int(a), float(r))
            agent.reset()
            np.testing.assert_allclose(neural.choice_logits(p, agent.state, model)[0], trace['logits'][0])
            if model != 'vanilla_rnn':
                output = neural.probe_reward_module(p, model, [0., 2., 4.], chosen_values=[0., 1.])
                self.assertEqual(output.shape, (2, 3))
                if model == 'memory_ann':
                    np.testing.assert_allclose(output[0], output[1])
                # Probe at zero initial state must match first update for reward 2.
                initial = neural.initial_state(p, model)
                initial = dict(initial, q=jnp.zeros_like(initial['q']))
                state = neural.observe(p, initial, model, jnp.array([0]), jnp.array([2.]))
                self.assertAlmostEqual(output[0, 1], float(state['q'][0, 0]))

    def test_neural_sweep_withheld_test_and_tree_checkpoint_resume(self):
        settings = dict(grid=[dict(hidden_units=4, optimizer_learning_rate=.001, l2_weight=1e-5, batch_size=2)],
                        models=neural.NEURAL_MODELS, n_seeds=1, max_steps=4,
                        validation_every=2, profile='neural-test')
        with tempfile.TemporaryDirectory() as root:
            partial = paper.run_paper_sweep(sequences(), root, max_new_runs=1, **settings)
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['test_sessions'], [])
            complete = paper.run_paper_sweep(sequences(), root, **settings)
            self.assertTrue(complete['complete'])
            self.assertEqual(len(complete['selected']), 4)
            for selected in complete['selected']:
                with np.load(Path(complete['output_dir'])/selected['parameters_file'], allow_pickle=False) as saved:
                    restored = {k: saved[k] for k in saved.files}
                score = paper.score_sequences(restored, paper.split_by_mouse(sequences())[0]['test'], selected['model'])[0]
                self.assertAlmostEqual(score, selected['test_cross_entropy'])
            self.assertEqual(complete, paper.run_paper_sweep(sequences(), root, **settings))
        # Interrupt after a saved Adam/weight-tree checkpoint and compare exact replay.
        settings['models'] = ('memory_ann',)
        original = paper._save_checkpoint
        def interrupt(*args):
            original(*args)
            raise KeyboardInterrupt('Synthetic interruption')
        with tempfile.TemporaryDirectory() as root:
            with patch.object(paper, '_save_checkpoint', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    paper.run_paper_sweep(sequences(), Path(root)/'resumed', **settings)
            resumed = paper.run_paper_sweep(sequences(), Path(root)/'resumed', **settings)
            fresh = paper.run_paper_sweep(sequences(), Path(root)/'fresh', **settings)
            self.assertEqual(resumed['selected'], fresh['selected'])
            self.assertEqual(resumed['test_sessions'], fresh['test_sessions'])


if __name__ == '__main__':
    unittest.main()
