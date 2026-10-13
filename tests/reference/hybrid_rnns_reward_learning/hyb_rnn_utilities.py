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

"""Utilities for fitting hybrid RNNs."""

import jax
import jax.numpy as jnp
import numpy as np


def format_data_for_model_training(hum_dat):
  """Format data for model training."""

  n_trials = len(np.unique(hum_dat['trial_id']))

  # Add one-hot encoding of actions
  hum_dat[['choice_{}'.format(i) for i in range(4)]] = jax.nn.one_hot(
      np.array(hum_dat['action']), 4
  )
  necessary_cols = ['choice_{}'.format(i) for i in range(4)] + ['reward',]
  hum_dat = hum_dat[necessary_cols]

  # Stack blocks on top of each other
  hum_dat_tensor = jnp.array(hum_dat).reshape(
      [int(len(hum_dat) / n_trials), n_trials, len(necessary_cols)]
  )
  n_subj_blocks = hum_dat_tensor.shape[0]

  # Split data into training, validation, and testing sets
  n_test_blocks = n_subj_blocks // 10

  train_dat = hum_dat_tensor[:n_test_blocks]
  valid_dat = hum_dat_tensor[n_test_blocks : 2 * n_test_blocks]
  test_dat = hum_dat_tensor[2 * n_test_blocks :]

  return {'train_dat': train_dat, 'valid_dat': valid_dat, 'test_dat': test_dat}


def get_batch(tensor_dat, batch_size, key):
  """Get a batch of data."""

  rnd_idxs = jax.random.choice(
      key=key, a=jnp.arange(len(tensor_dat)), shape=(batch_size,), replace=False
  )
  batch = tensor_dat[rnd_idxs]

  return batch
