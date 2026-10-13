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

"""Config for hybrid modeling."""

import jax

from ml_collections import config_dict


def get_config():
  """Config file for cognitive models and hybrid RNNs."""

  config = config_dict.ConfigDict()

  # Before anything else: Are we debugging?
  config.debug = True

  # Random seed and dataset path
  config.random_seed = 42
  config.dataset_path = (
      '/path/to/dataset/openSourceHumDataset.csv'
  )

  # Training dataset
  config.n_trials = 150  # Humans have 150 trials
  config.n_datasets = 3520 + 388

  # Training procedure
  config.model_name = 'birnn'  # Can be "rnn", "birnn", "cogmod"
  if config.debug:
    config.n_training_steps = 100
    config.batch_size = 2
  else:
    config.n_training_steps = int(1e6)
    config.batch_size = 32
  config.learning_rate = 1e-4
  config.weight_decay = 1e-5

  # MLP stuff
  config.network_params = config_dict.ConfigDict()
  config.network_params.n_actions = 4  # Humans have 4 actions
  config.network_params.final_activation_fn = jax.nn.softmax  # action probab.
  config.network_params.hidden_size = 16

  # Which parameters are fitted?
  config.rnn_rl_params = config_dict.ConfigDict()
  config.rnn_rl_params.fit_alpha = True
  config.rnn_rl_params.fit_beta = False
  config.rnn_rl_params.fit_bias = False
  config.rnn_rl_params.fit_forget = False
  config.rnn_rl_params.fit_persev_p = False
  config.rnn_rl_params.fit_persev_t = False
  config.rnn_rl_params.fit_init_v = True
  config.rnn_rl_params.fit_init_h = True
  config.rnn_rl_params.fit_init_v_state = False
  config.rnn_rl_params.fit_init_h_state = False
  config.rnn_rl_params.fit_w = False

  # Which values do the parameters take that are not fitted?
  config.rnn_rl_params.alpha = 0.  # alpha needs to be 0 for 0 value update
  config.rnn_rl_params.beta = 1.
  config.rnn_rl_params.bias = 0.
  config.rnn_rl_params.forget = 0.
  config.rnn_rl_params.persev_p = 0.
  config.rnn_rl_params.persev_t = 0.

  config.rnn_rl_params.w_v = 0.5
  config.rnn_rl_params.w_h = 0.5
  config.rnn_rl_params.o = False
  config.rnn_rl_params.s = False
  config.rnn_rl_params.vo = False
  config.rnn_rl_params.vs = False
  config.rnn_rl_params.ho = False
  config.rnn_rl_params.hs = False
  config.rnn_rl_params.zero_values = False  # for value update, prev_act_val = 0

  return config
