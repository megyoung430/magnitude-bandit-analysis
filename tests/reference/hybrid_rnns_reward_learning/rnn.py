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

"""RNN.
"""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

RNNState = tuple[jnp.ndarray, jnp.ndarray]


class RNN(hk.RNNCore):
  """RNN that predicts action logits based on all inputs (action, reward)."""

  def __init__(self, rl_params, network_params):

    super().__init__()

    self._s = rl_params.s
    self._o = rl_params.o

    self._n_actions = network_params.n_actions
    self._hidden_size = network_params.hidden_size
    self._final_activation_fn = network_params.final_activation_fn

  def __call__(
      self, inputs: jnp.ndarray, prev_state: RNNState
  ) -> tuple[jnp.ndarray, RNNState]:
    gist, state = prev_state

    if self._o:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, gist], axis=-1)
    if self._s:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
    next_gist = hk.Linear(self._n_actions)(next_state)

    action_probs = self._final_activation_fn(next_gist)  # ba

    return action_probs, (next_gist, next_state)

  def initial_state(self,
                    batch_size: Optional[int]) -> RNNState:

    return (
        jnp.zeros((batch_size, self._n_actions)),  # gist
        jnp.zeros((batch_size, self._hidden_size)),  # state
        )
