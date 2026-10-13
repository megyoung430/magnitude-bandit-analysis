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

"""Fit cogmods or LSTMs to human or simulated data."""

import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import optax
import pandas as pd

from hybrid_rnns_reward_learning import hyb_rnn_utilities
from hybrid_rnns_reward_learning import rnn_config
from hybrid_rnns_reward_learning.bi_rnn import BiRNN  # pylint: disable=g-importing-member
from hybrid_rnns_reward_learning.cogmod import CogMod  # pylint: disable=g-importing-member
from hybrid_rnns_reward_learning.rnn import RNN  # pylint: disable=g-importing-member


_CONFIG = config_flags.DEFINE_config_dict('config', rnn_config.get_config())
rl_param_names = [
    'alpha',
    'beta',
    'bias',
    'forget',
    'persev_p',
    'persev_t',
    'init_value_v',
    'init_value_h',
]


def train(config):
  """Fit one model (cogmod/RNN/biRNN...) to human bandit task behavior."""

  # Get model functions
  def _cogmod_fn(input_seq, return_all_states=False):
    """Cognitive models function."""

    model = CogMod(config.rnn_rl_params, config.network_params)
    initial_state = model.initial_state(config.batch_size)

    return hk.dynamic_unroll(
        model,
        input_seq,
        initial_state,
        time_major=False,
        return_all_states=return_all_states)

  def _birnn_fn(input_seq, return_all_states=False):
    """LSTM model function."""

    bi_rnn = BiRNN(config.rnn_rl_params, config.network_params)
    initial_state = bi_rnn.initial_state(config.batch_size)

    return hk.dynamic_unroll(
        bi_rnn,
        input_seq,
        initial_state,
        time_major=False,
        return_all_states=return_all_states)

  def _rnn_fn(input_seq, return_all_states=False):
    """RNN model function."""

    rnn = RNN(config.rnn_rl_params, config.network_params)
    initial_state = rnn.initial_state(config.batch_size)

    return hk.dynamic_unroll(
        rnn,
        input_seq,
        initial_state,
        time_major=False,
        return_all_states=return_all_states)

  # Get loss and update function
  @jax.jit
  def loss_fn(params, key, batch_dat):
    """Cross-entropy loss between model-predicted and input behavior."""

    batch_size = batch_dat.shape[0]
    action_probs_seq, _ = forward.apply(params, key, batch_dat)
    action_probs_seq = (1 - 1e-5) * action_probs_seq + 5e-4

    # calculate loss ("sum" so that missed trials don't influence the results)
    loss = -jnp.sum(jnp.log(action_probs_seq[:, :-1]) * batch_dat[:, 1:, :4]
                    ) / batch_size

    return loss

  @jax.jit
  def update_func(params, key, opt_state, batch_dat):
    """Updates function for the RNN."""

    loss, grads = jax.value_and_grad(loss_fn)(
        params, key, batch_dat
        )
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    scalars = {'train_loss': [loss]}
    return new_params, new_opt_state, scalars

  # Get training, validation, and testing data
  print('Loading data from {}'.format(config.dataset_path))

  hum_dat = pd.read_csv(config.dataset_path)
  hum_dat_tensor = hyb_rnn_utilities.format_data_for_model_training(hum_dat)

  train_dat = hum_dat_tensor['train_dat']
  valid_dat = hum_dat_tensor['valid_dat']
  test_dat = hum_dat_tensor['test_dat']
  print('Size of training data: {} blocks.'.format(len(train_dat)))

  # Get the model we're going to fit
  if config.model_name == 'cogmod':
    print('Using cogmod to fit data.')
    forward = hk.transform(_cogmod_fn)

  elif config.model_name == 'birnn':
    print('Using BiRNN to fit data.')
    forward = hk.transform(_birnn_fn)

  elif config.model_name == 'rnn':
    print('Using RNN to fit data.')
    forward = hk.transform(_rnn_fn)

  else:
    raise ValueError('Unknown model name: %s' % config.model_name)

  # Get initial model parameters
  rng_seq = hk.PRNGSequence(config.random_seed)
  train_batch = hyb_rnn_utilities.get_batch(
      train_dat, config.batch_size, next(rng_seq)
  )
  params = forward.init(next(rng_seq), train_batch)

  # Get optimizer
  optimizer = optax.adamw(
      learning_rate=config.learning_rate,
      weight_decay=config.weight_decay
      )
  init_opt = jax.jit(optimizer.init)
  opt_state = init_opt(params)

  print('Start fitting the model')
  for current_step in range(config.n_training_steps):  # pylint: disable=undefined-variable

    # Get next training batch
    train_batch = hyb_rnn_utilities.get_batch(
        train_dat, config.batch_size, next(rng_seq)
    )

    params, opt_state, scalars = update_func(
        params, next(rng_seq), opt_state, train_batch
    )

    # Print / save training data
    if current_step % 500 == 0:

      # Calculate loss on testing data
      test_batch = hyb_rnn_utilities.get_batch(
          test_dat, config.batch_size, next(rng_seq)
      )
      test_loss = loss_fn(params, next(rng_seq), test_batch)

      # Calculate loss on validation data
      valid_batch = hyb_rnn_utilities.get_batch(
          valid_dat, config.batch_size, next(rng_seq)
      )
      valid_loss = loss_fn(params, next(rng_seq), valid_batch)

      # Update and save scalars
      scalars.update({
          'step': [current_step],
          'test_loss': [jax.device_get(test_loss)],
          'valid_loss': [jax.device_get(valid_loss)],
      })

      for key, value in scalars.items():
        if key in ['train_loss', 'w'] + rl_param_names:
          scalars[key] = jax.device_get(value)

      print('Step: {},\nScalars: {}'.format(current_step, scalars))

  return scalars, params


def main(_):
  train(_CONFIG.value)

if __name__ == '__main__':
  main(None)
