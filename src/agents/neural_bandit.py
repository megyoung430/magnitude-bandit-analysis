"""Four neural architectures aligned to the pinned released Haiku implementation.

No beta multiplier: learned outputs are logits. The hybrid reward network only
adds to decayed chosen Q (RL/Context) or replaces chosen Q (Memory).
Action inputs are one-hot. Memory-ANN has a single global hidden state per
channel; vanilla recurrence explicitly includes its previous hidden state.
Initial Q and action-kernel values are learned scalars. Memory retains the
upstream zero-valued input slot. See the notebook for paper/code differences. All functions use predict-before-observe timing.
"""
import numpy as np
from src.agents.paper_rl_fitting import dependencies

NEURAL_MODELS = ('rl_ann', 'context_ann', 'memory_ann', 'vanilla_rnn')
DISPLAY_NAMES = dict(rl_ann='RL-ANN', context_ann='Context-ANN',
                     memory_ann='Memory-ANN', vanilla_rnn='Vanilla RNN')
PAPER_CHOSEN = {
    'rl_ann': dict(hidden_units=16, batch_size=32, max_steps=1_000_000),
    'context_ann': dict(hidden_units=64, batch_size=64, max_steps=1_000_000),
    'memory_ann': dict(hidden_units=32, batch_size=128, max_steps=1_000_000),
    'vanilla_rnn': dict(hidden_units=64, batch_size=64, max_steps=1_000_000),
}


def initialize_parameters(key, model, hidden_units=16, num_arms=3):
    """Haiku Linear distribution: truncated N(0, 1/sqrt(fan_in)); zero biases.

    Fitted Q/c initial scalars ~ N(1,1), raw forgetting ~ N(0,1).
    Matches upstream distributions, not Haiku module PRNG key allocation.
    """
    jax, jnp, _ = dependencies()
    if model not in NEURAL_MODELS or hidden_units < 1 or num_arms < 2:
        raise ValueError('Unknown architecture or invalid layer dimensions.')
    keys = iter(jax.random.split(key, 7))
    def weights(n_in, n_out):
        return jax.random.truncated_normal(next(keys), -2., 2., (n_in, n_out),
                                           dtype=jnp.float64)/np.sqrt(n_in)
    h, k = hidden_units, num_arms
    if model == 'vanilla_rnn':
        return dict(joint_w1=weights(k+1+h, h), joint_b1=jnp.zeros(h),
                    joint_w2=weights(h, k), joint_b2=jnp.zeros(k))
    reward_inputs = {'rl_ann': 2, 'context_ann': 2+k, 'memory_ann': 2+h}[model]
    action_inputs = {'rl_ann': k, 'context_ann': 2*k, 'memory_ann': k+h}[model]
    return dict(reward_w1=weights(reward_inputs, h), reward_b1=jnp.zeros(h),
                reward_w2=weights(h, 1), reward_b2=jnp.zeros(1),
                action_w1=weights(action_inputs, h), action_b1=jnp.zeros(h),
                action_w2=weights(h, k), action_b2=jnp.zeros(k),
                forgetting_raw=jax.random.normal(next(keys), (), dtype=jnp.float64),
                q_init=1+jax.random.normal(next(keys), (), dtype=jnp.float64),
                c_init=1+jax.random.normal(next(keys), (), dtype=jnp.float64))


def hidden_width(parameters, model):
    return int(parameters['joint_b1' if model == 'vanilla_rnn' else 'reward_b1'].shape[0])


def parameter_summary(parameters, model):
    jax, _, _ = dependencies()
    summary = dict(hidden_units=hidden_width(parameters, model),
                   n_parameters=sum(np.asarray(v).size for v in parameters.values()))
    if model != 'vanilla_rnn':
        summary.update(forgetting=float(jax.nn.sigmoid(parameters['forgetting_raw'])),
                       q_init=float(parameters['q_init']), c_init=float(parameters['c_init']))
    return summary


def initial_state(parameters, model, batch_size=1, num_arms=3):
    _, jnp, _ = dependencies()
    if model != 'vanilla_rnn' and 'c_init' not in parameters:
        raise ValueError('Pre-release-alignment neural checkpoint: c_init is missing. Refit with the updated implementation; do not reuse old weights.')
    h = hidden_width(parameters, model)
    q_init = 0. if model == 'vanilla_rnn' else parameters['q_init']
    return dict(q=jnp.full((batch_size, num_arms), q_init),
                c=jnp.full((batch_size, num_arms), 0. if model == 'vanilla_rnn' else parameters['c_init']),
                reward_hidden=jnp.zeros((batch_size, h)),
                action_hidden=jnp.zeros((batch_size, h)),
                joint_hidden=jnp.zeros((batch_size, h)))


def choice_logits(parameters, state, model):
    if model == 'vanilla_rnn':
        return state['joint_hidden'] @ parameters['joint_w2']+parameters['joint_b2']
    return state['q']+state['c']


def observe(parameters, state, model, actions, rewards):
    """Advance batch states after rewards; the returned state predicts next choice."""
    jax, jnp, _ = dependencies()
    q, c = state['q'], state['c']
    k, batch = q.shape[1], q.shape[0]
    index = jnp.arange(batch)
    onehot = jax.nn.one_hot(actions, k, dtype=jnp.float64)
    reward_column = rewards[:, None]
    if model == 'vanilla_rnn':
        inputs = jnp.concatenate([onehot, reward_column, state['joint_hidden']], axis=1)
        hidden = jnp.tanh(inputs @ parameters['joint_w1']+parameters['joint_b1'])
        return dict(state, joint_hidden=hidden)
    if model == 'rl_ann':
        reward_inputs = jnp.concatenate([q[index, actions, None], reward_column], axis=1)
        action_inputs = onehot
    elif model == 'context_ann':
        reward_inputs = jnp.concatenate([q[index, actions, None], reward_column, q], axis=1)
        action_inputs = jnp.concatenate([onehot, c], axis=1)
    elif model == 'memory_ann':
        reward_inputs = jnp.concatenate([jnp.zeros_like(reward_column), reward_column, state['reward_hidden']], axis=1)
        action_inputs = jnp.concatenate([onehot, state['action_hidden']], axis=1)
    else:
        raise ValueError('Unknown model.')
    sr = jnp.tanh(reward_inputs @ parameters['reward_w1']+parameters['reward_b1'])
    sa = jnp.tanh(action_inputs @ parameters['action_w1']+parameters['action_b1'])
    updated_value = (sr @ parameters['reward_w2']+parameters['reward_b2'])[:, 0]
    forgetting = jax.nn.sigmoid(parameters['forgetting_raw'])
    q = (1-forgetting)*q+forgetting*parameters['q_init']
    if model == 'memory_ann':
        q = q.at[index, actions].set(updated_value)
    else:
        q = q.at[index, actions].add(updated_value)
    c = sa @ parameters['action_w2']+parameters['action_b2']
    return dict(state, q=q, c=c, reward_hidden=sr, action_hidden=sa)


def make_backend(model, hidden_units=16, num_arms=3):
    jax, jnp, optax = dependencies()
    if model not in NEURAL_MODELS:
        raise ValueError('Unknown model.')

    @jax.jit
    def statistics(parameters, actions, rewards, mask, released_loss=False, skip_first=False):
        batch = actions.shape[0]
        index = jnp.arange(batch)
        start = (initial_state(parameters, model, batch, num_arms), jnp.zeros(batch), jnp.zeros(batch))
        def trial(carry, inputs):
            state, nll, correct = carry
            action, reward, valid, t = inputs
            logits = choice_logits(parameters, state, model)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            chosen = log_probs[index, action]
            adjusted = jnp.logaddexp(jnp.log1p(-1e-5)+chosen, jnp.log(5e-4))
            scored = valid & ((t > 0) | ~jnp.asarray(skip_first))
            nll += jnp.where(scored, -jnp.where(released_loss, adjusted, chosen), 0.)
            correct += jnp.where(scored, jnp.argmax(logits, axis=-1) == action, 0.)
            # Sanitize padding before the network, then freeze its state.
            new_state = observe(parameters, state, model, action, jnp.where(valid, reward, 0.))
            state = jax.tree_util.tree_map(lambda new, old: jnp.where(valid[:, None], new, old), new_state, state)
            return (state, nll, correct), None
        final, _ = jax.lax.scan(trial, start, (actions.T, rewards.T, mask.T, jnp.arange(actions.shape[1])))
        return final[1], final[2]

    def objective(parameters, actions, rewards, mask, l2_weight):
        # l2_weight is retained as an API argument; AdamW applies decay separately.
        nll, _ = statistics(parameters, actions, rewards, mask, True, True)
        return nll.sum()/actions.shape[0]

    value_grad = jax.value_and_grad(objective)
    def trainer(setting):
        optimizer = optax.adamw(setting['optimizer_learning_rate'],
                                weight_decay=setting.get('weight_decay', setting.get('l2_weight', 0.)))
        def step(carry, unused):
            parameters, state, key, actions, rewards, mask = carry
            key, sampling_key = jax.random.split(key)
            ids = jax.random.choice(sampling_key, actions.shape[0], (setting['batch_size'],), replace=False)
            loss, grad = value_grad(parameters, actions[ids], rewards[ids], mask[ids], 0.)
            updates, state = optimizer.update(grad, state, parameters)
            parameters = optax.apply_updates(parameters, updates)
            return (parameters, state, key, actions, rewards, mask), loss
        def chunk(parameters, state, key, actions, rewards, mask, steps):
            carry, losses = jax.lax.scan(step, (parameters, state, key, actions, rewards, mask),
                                         xs=None, length=steps)
            return carry[:3], losses[-1]
        return optimizer, jax.jit(chunk, static_argnums=(6,))
    return statistics, objective, trainer


def trace_sequence(parameters, sequence, model):
    """Pre-choice probabilities/logits/states and post-observation channel states."""
    from src.agents.paper_rl_fitting import pack_sequences
    jax, jnp, _ = dependencies()
    num_arms = len(sequence['towers'])
    actions, rewards, _ = pack_sequences([sequence], num_arms=num_arms)
    @jax.jit
    def trace(parameters, actions, rewards):
        def trial(state, inputs):
            action, reward = inputs
            logits = choice_logits(parameters, state, model)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            new = observe(parameters, state, model, action[None], reward[None])
            result = dict(logits=logits[0], log_probabilities=log_probs[0],
                          pre_q=state['q'][0], pre_c=state['c'][0],
                          pre_reward_hidden=state['reward_hidden'][0],
                          pre_action_hidden=state['action_hidden'][0],
                          pre_joint_hidden=state['joint_hidden'][0],
                          post_q=new['q'][0], post_c=new['c'][0])
            return new, result
        _, result = jax.lax.scan(trial, initial_state(parameters, model, num_arms=num_arms), (actions, rewards))
        return result
    return {k: np.asarray(v) for k, v in trace(parameters, jnp.asarray(actions[0]), jnp.asarray(rewards[0])).items()}


class NeuralBanditMouse:
    """Adapter to DriftingBanditTask.simulate; reset between independent sessions.

    q_values exposes hybrid Q contributions or vanilla logits for plotting only.
    The task's previous_action is ignored: observe already carries the history.
    """
    def __init__(self, parameters, model, num_arms=3, seed=0):
        self.parameters, self.model, self.num_arms = parameters, model, num_arms
        self.rng = np.random.default_rng(seed)
        jax, _, _ = dependencies()
        self._observe = jax.jit(lambda state, a, r: observe(parameters, state, model, a, r))
        self.reset()

    def reset(self):
        self.state = initial_state(self.parameters, self.model, 1, self.num_arms)
        self._refresh_values()

    def _refresh_values(self):
        self.q_values = np.asarray(choice_logits(self.parameters, self.state, self.model)[0]
                                   if self.model == 'vanilla_rnn' else self.state['q'][0])

    def choose_action(self, previous_action=None):
        logits = np.asarray(choice_logits(self.parameters, self.state, self.model)[0])
        probs = np.exp(logits-logits.max())
        return self.rng.choice(self.num_arms, p=probs/probs.sum())

    def update_q_values(self, action, reward):
        _, jnp, _ = dependencies()
        self.state = self._observe(self.state, jnp.array([action]), jnp.array([reward], dtype=jnp.float64))
        self._refresh_values()


def probe_reward_module(parameters, model, rewards, chosen_values=(0.,), context=None,
                        reward_hidden=None, chosen_arm=0):
    """Probe the released hybrid chosen-value computation (including residual path).

    Context-ANN sets context[chosen_arm] to each chosen value for consistency.
    Memory-ANN ignores chosen_values and uses a fixed GLOBAL reward hidden state.
    These probes describe the supplied network, not validated behavioral effects.
    """
    jax, jnp, _ = dependencies()
    if model not in NEURAL_MODELS[:3]:
        raise ValueError('Vanilla RNN has no separate reward module.')
    rewards, values = np.asarray(rewards, dtype=float), np.asarray(chosen_values, dtype=float)
    if rewards.ndim != 1 or values.ndim != 1 or not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(values)):
        raise ValueError('Need finite one-dimensional probe values.')
    k, h = len(parameters['action_b2']), hidden_width(parameters, model)
    if chosen_arm < 0 or chosen_arm >= k:
        raise ValueError('Invalid chosen arm.')
    results = []
    for value in values:
        if model == 'memory_ann':
            hidden = np.zeros(h) if reward_hidden is None else np.asarray(reward_hidden)
            if hidden.shape != (h,):
                raise ValueError('reward_hidden must have hidden_units entries.')
            inputs = np.column_stack([np.zeros(len(rewards)), rewards, np.broadcast_to(hidden, (len(rewards), h))])
        else:
            inputs = np.column_stack([np.full(len(rewards), value), rewards])
            if model == 'context_ann':
                ctx = np.zeros(k) if context is None else np.asarray(context)
                if ctx.shape != (k,):
                    raise ValueError('context must have num_arms entries.')
                ctx_batch = np.broadcast_to(ctx, (len(rewards), k)).copy()
                ctx_batch[:, chosen_arm] = value
                inputs = np.column_stack([inputs, ctx_batch])
        hidden = jnp.tanh(jnp.asarray(inputs) @ parameters['reward_w1']+parameters['reward_b1'])
        output = (hidden @ parameters['reward_w2']+parameters['reward_b2'])[:, 0]
        f = jax.nn.sigmoid(parameters['forgetting_raw'])
        results.append(np.asarray(output if model == 'memory_ann' else
                                  (1-f)*value+f*parameters['q_init']+output))
    return np.asarray(results)
