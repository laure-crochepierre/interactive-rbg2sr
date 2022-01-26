# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression

import warnings

warnings.filterwarnings("ignore")

from torch.autograd import set_detect_anomaly as torch_autograd_set_detect_anomaly
torch_autograd_set_detect_anomaly(True)

from torch import Tensor as torch_Tensor
from torch import BoolTensor as torch_BoolTensor
from torch import cat as torch_cat
from torch import multiply as torch_multiply

from torch.nn import Module, LeakyReLU, Linear, Sequential, Transformer, Conv1d, LSTM, Softmax, Parameter, \
    ConvTranspose1d, Tanh, TransformerEncoderLayer

from typing import NamedTuple, Dict, Union
from gym.spaces import Box, MultiBinary

from utils.masking_categorical import CategoricalMasked

TensorDict = Dict[Union[str, int], torch_Tensor]


class CuriousDictRolloutBufferSamples(NamedTuple):
    intrinsic_rewards: torch_Tensor
    extrinsic_rewards: torch_Tensor
    log_probs: torch_Tensor
    entropies: torch_Tensor


class Policy(Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_dim,
                 embedding_dim,
                 max_horizon,
                 embedding=False,
                 autoencoder=False,
                 batch_size=1,
                 reward_prediction=False,
                 use_transformer=False,
                 non_linearity=LeakyReLU(),
                 **kwargs):
        super(Policy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_horizon = max_horizon
        self.n_actions = self.action_space.shape[0]
        self.batch_size = batch_size
        self.non_linearity = non_linearity
        self.use_transformer = use_transformer
        # force embedding
        self.embedding = embedding or (sum([len(s.shape) >= 2 for s in self.observation_space.spaces.values()]) > 0)
        self.autoencoder = self.embedding & autoencoder
        self.reward_prediction = reward_prediction

        if self.embedding:
            # Define Encoder Architecture
            self.encoders = {}
            for key, space in self.observation_space.spaces.items():
                if isinstance(space, Box):
                    setattr(self, f"encoder_{key}", Linear(space.shape[0], self.embedding_dim))

                elif isinstance(space, MultiBinary):
                    if len(space.shape) == 1:
                        setattr(self, f"encoder_{key}", Sequential(
                            Linear(space.shape[0], self.embedding_dim),
                            self.non_linearity))
                    else:
                        if self.use_transformer:
                            transformer_encoder = Sequential(
                                TransformerEncoderLayer(d_model=space.shape[1], nhead=1, dim_feedforward=16,
                                                           batch_first=True),
                                Conv1d(space.shape[0], 1, 4),
                                self.non_linearity,
                                Linear(space.shape[1] - 4 + 1, self.embedding_dim),
                                self.non_linearity
                            )

                            setattr(self, f"encoder_{key}", transformer_encoder)
                        else:
                            encoder = Sequential(
                                Conv1d(space.shape[0], 1, 4),
                                self.non_linearity,
                                Linear(space.shape[1] - 4 + 1, self.embedding_dim),
                                self.non_linearity
                            )

                            setattr(self, f"encoder_{key}", encoder)
                self.encoders[key] = getattr(self, f"encoder_{key}")

            self.features_encoder_layer = Linear(self.embedding_dim * len(self.observation_space.spaces),
                                                    self.hidden_dim)
            if self.autoencoder:
                self.decoders = {}
                for key, space in self.observation_space.spaces.items():
                    if isinstance(space, Box):
                        setattr(self, f"decoder_{key}", Linear(self.embedding_dim, space.shape[0]))
                    elif isinstance(space, MultiBinary):
                        if len(space.shape) == 1:
                            setattr(self, f"decoder_{key}", Sequential(
                                Linear(self.embedding_dim, space.shape[0]),
                                self.non_linearity))
                        else:
                            setattr(self, f"decoder_{key}", Sequential(
                                Linear(self.embedding_dim, space.shape[1]),
                                self.non_linearity,
                                ConvTranspose1d(in_channels=self.batch_size,
                                                   out_channels=32,
                                                   kernel_size=1),
                                self.non_linearity,
                                ConvTranspose1d(in_channels=32,
                                                   out_channels=space.shape[0],
                                                   kernel_size=1),
                                self.non_linearity

                            ))
                    self.decoders[key] = getattr(self, f"decoder_{key}")
                self.ae_coeff_loss = Parameter(torch_Tensor(0.5), requires_grad=True)

        else:
            inputs_dim = sum([s.shape[0] for s in self.observation_space.spaces.values()])
            self.features_encoder_layer = Linear(inputs_dim, self.hidden_dim)

        # Define Action predictor Architecture

        self.lstm_layer = LSTM(input_size=self.hidden_dim,
                                  hidden_size=self.hidden_dim, batch_first=True)
        self.intermediate_layer = Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.action_decoder_layer = Linear(in_features=self.hidden_dim, out_features=self.n_actions)
        self.softmax = Softmax(dim=-1)

        self.multiply = torch_multiply
        self.score_predictor = Sequential(
            Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim / 2)),
            self.non_linearity,
            Linear(in_features=int(self.hidden_dim / 2), out_features=int(self.hidden_dim / 4)),
            self.non_linearity,
            Linear(in_features=int(self.hidden_dim / 4), out_features=self.n_actions),
        )

    def encode(self, inputs):
        if self.embedding:
            encoded_inputs = {}
            for k in self.observation_space.spaces.keys():
                if not isinstance(self.encoders[k], Transformer):
                    encoded_inputs[k] = self.encoders[k](torch_Tensor(inputs[k]))
                else:
                    encoded_inputs[k] = self.encoders[k](torch_Tensor(inputs[k]), torch_Tensor(inputs[k]))
            cat_inputs = torch_cat(list(encoded_inputs.values()), -1)
            if self.autoencoder:
                decoded_inputs = {k: self.decoders[k](encoded_input) for k, encoded_input in encoded_inputs.items()}
                return self.features_encoder_layer(cat_inputs), decoded_inputs
            else:
                return self.features_encoder_layer(cat_inputs), inputs
        else:
            cat_inputs = torch_cat([torch_Tensor(inputs[k]) for k in self.observation_space.spaces.keys()], -1)
            return self.features_encoder_layer(cat_inputs), inputs

    def forward(self, inputs, h_in, c_in):

        x_inputs, inputs_hat = self.encode(inputs)
        x = self.non_linearity(x_inputs)
        x_lstm, (h_out, c_out) = self.lstm_layer(x, (h_in, c_in))
        x_lstm = Tanh()(x_lstm)

        x = self.intermediate_layer(x_lstm)
        x = self.non_linearity(x)

        action_logits = self.action_decoder_layer(x)
        score_prediction = self.score_predictor(x_lstm)

        return action_logits, h_out, c_out, [inputs_hat, score_prediction]

    def select_action(self, state, h_in, c_in, action=None):
        action_logits, h_out, c_out, other_predictions = self.forward(state, h_in, c_in)

        # create a categorical distribution over the list of probabilities of actions
        m = CategoricalMasked(logits=action_logits, masks=torch_BoolTensor(state['current_mask']))

        # and sample an action using the distribution
        if action is None:
            action = m.sample()

        # compute log_probs
        log_probs = m.log_prob(action)
        entropy = m.entropy()

        return action, log_probs, entropy, h_out, c_out, other_predictions
