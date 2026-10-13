# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Define classic alpha-beta RL model."""

import haiku as hk
import jax
import jax.numpy as jnp


class CogMod(hk.RNNCore):
  """Classic cog models, formulated as an RNN."""

  def __init__(self, rl_params, network_params, init_value=0.5):

    super().__init__()

    self._n_actions = network_params.n_actions
    self._hidden_size = network_params.hidden_size
    self._final_activation_fn = network_params.final_activation_fn

    if rl_params['fit_init_v']:
      init = hk.initializers.RandomNormal(stddev=1, mean=1)
      self._init_value = hk.get_parameter('init_value_v', (1,), init=init)
    else:
      self._init_value = init_value

    if rl_params['fit_alpha']:
      self.alpha = jax.nn.sigmoid(  # 0 < alpha < 1
          hk.get_parameter('unsigmoid_alpha', (1,), init=jnp.ones)
      )
    else:
      self.alpha = rl_params['alpha']

    if rl_params['fit_beta']:
      self.beta = jax.nn.relu(  # 0 < beta < inf
          hk.get_parameter('unrelued_beta', (1,), init=jnp.ones)
      )
    else:
      self.beta = rl_params['beta']

    if rl_params['fit_forget']:
      self.forget = jax.nn.sigmoid(  # 0 < forget < 1
          hk.get_parameter('unsigmoid_forget', (1,), init=jnp.zeros)
      )
    else:
      self.forget = rl_params['forget']

    if rl_params['fit_persev_p']:  # -1 < persev < 1
      self.persev_p = jax.nn.tanh(
          hk.get_parameter('untanh_persev_p', (1,), init=jnp.zeros)
      )
    else:
      self.persev_p = rl_params['persev_p']

    if rl_params['fit_persev_t']:  # -1 < persev < 1
      self.persev_t = jax.nn.tanh(
          hk.get_parameter('untanh_persev_t', (1,), init=jnp.zeros)
      )
    else:
      self.persev_t = rl_params['persev_t']

  def _rl_value_fn(self, prev_value, action, reward):
    rpe = reward - jnp.sum(action * prev_value, axis=-1)  # shape: (batch_size)
    new_value = (prev_value
                 + action * self.alpha * rpe[:, jnp.newaxis])  # (b_s, n_a)
    return new_value

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    prev_value = prev_state  # shape: (batch_size, n_actions)
    action = inputs[:, :self._n_actions]  # shape: (batch_size, n_actions)
    reward = inputs[:, -1]  # shape: (batch_size,)

    # Value update
    new_value = self._rl_value_fn(prev_value, action, reward)  # (bs, n_act)

    # Value forgetting
    new_value = (1 - self.forget) * new_value + self.forget * self._init_value

    # Perseverance and choice kernel
    new_value += self.persev_p * action  # (bs, n_act)
    pers_value = new_value + self.persev_t * action  # (bs, n_act)

    # Action selection probabilities
    action_probs = self._final_activation_fn(self.beta * pers_value)  # (bs, na)

    return action_probs, new_value

  def initial_state(self, batch_size):
    return self._init_value * jnp.ones([batch_size, self._n_actions])
