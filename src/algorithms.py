# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression

import os
import time
import numpy as np

import warnings

warnings.filterwarnings("ignore")

from torch.autograd import set_detect_anomaly as torch_autograd_set_detect_anomaly
torch_autograd_set_detect_anomaly(True)

from torch import inference_mode as torch_inference_mode
from torch import ones as torch_ones
from torch import zeros as torch_zeros
from torch import Tensor as torch_Tensor
from torch import BoolTensor as torch_BoolTensor
from torch import vstack as torch_vstack
from torch import float32 as torch_float32
from torch import randint as torch_randint
from torch import gather as torch_gather
from torch import mul as torch_mul
from torch.nn.utils import clip_grad_norm_ as torch_nn_utils_clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from torch.nn import Linear as nn_Linear
from torch.nn import Conv1d as nn_Conv1d
from torch.nn import init as nn_init
from torch.nn import LSTM as nn_LSTM
from torch.nn import MSELoss as nn_MSELoss

from collections import namedtuple
from abc import ABC, abstractmethod

from utils.masking_categorical import CategoricalMasked

Batch = namedtuple('Batch', ('state', 'h', 'c', 'action', 'past_done', 'final_reward'))


def init_weights(m):
    if isinstance(m, nn_Linear) or isinstance(m, nn_Conv1d):
        nn_init.xavier_uniform_(m.weight)
    elif isinstance(m, nn_LSTM):
        for p in m.parameters():
            if p.data.ndimension() >= 2:
                nn_init.xavier_uniform_(p.data)
            else:
                nn_init.uniform_(p.data)


class BaseAlgorithm(ABC):
    def __init__(self,
                 env_class,
                 env_kwargs,
                 policy_class,
                 policy_kwargs,
                 risk_seeking=True,
                 risk_eps=0.05,
                 entropy_coeff=0,
                 optimizer_class=Adam,
                 learning_rate=0.001,
                 reward_prediction=False,
                 batch_size=64,
                 init_type='zeros',
                 dataset='nguyen1',
                 verbose=1,
                 debug=0,
                 writer_logdir="runs/",
                 **kwargs):
        super(BaseAlgorithm, self).__init__()

        self.risk_eps = risk_eps
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size

        # Create env and policy
        self.dataset = dataset
        self.env = env_class(**env_kwargs)

        # Define logs
        self.logger = dict(best_reward=-1,
                           best_expression=None,
                           nb_invalid=0,
                           i_best_episode=0)
        self.i_episode = 0
        self.verbose = verbose
        self.debug = debug
        self.risk_seeking = risk_seeking
        self.reward_prediction = reward_prediction

        # Define custom policy
        policy_kwargs.update(dict(observation_space=self.env.observation_space,
                                  action_space=self.env.action_space,
                                  max_horizon=self.env.max_horizon))

        self.policy = policy_class(**policy_kwargs)
        self.policy.apply(init_weights)

        self.optimizer = optimizer_class(self.policy.parameters(), lr=learning_rate)
        self.eps = np.finfo(np.float32).eps.item()

        init_functions = {'zeros': lambda size: torch_zeros(size, dtype=torch_float32),
                          "randint": lambda size: torch_randint(-1, 1, size, dtype=torch_float32)}
        self.init_type = init_functions[init_type]

        # Logs
        self.writer_logdir = writer_logdir
        self.logger = {'best_expression': '',
                       'best_reward': -np.inf,
                       'i_epoch': -1}

    def train(self, n_epochs):
        batch, final_rewards = None, None
        for i_epoch in range(n_epochs):
            batch, final_rewards = self.sample_episodes(i_epoch=i_epoch)
            #  Early stopping
            if (1 - self.logger['best_reward']) < 1e-15:
                print('Early stopping', flush=True)
                print(f'Found {self.logger["best_expression"]} at epoch {self.logger["i_best_epoch"]} '
                      f'with reward {self.logger["best_reward"]}', flush=True)
                break

            self.optimize_model(batch, final_rewards, i_epoch=i_epoch)

    @abstractmethod
    @torch_inference_mode()
    def sample_episodes(self, i_epoch=0):
        return NotImplementedError

    @abstractmethod
    def optimize_model(self, batch, final_rewards, i_epoch):
        return NotImplementedError


class ReinforceAlgorithm(BaseAlgorithm):
    def __init__(self, **kwargs):
        super(ReinforceAlgorithm, self).__init__(**kwargs)
        if os.environ.get('DROPBOX_ACCESS_TOKEN') is None:
            self.writer = self.create_summary_writer()

    def create_summary_writer(self):
        return SummaryWriter(log_dir=self.writer_logdir,
                             comment=f"Reinforce_experiment_{self.dataset}_{time.time()}")

    @torch_inference_mode()
    def sample_episodes(self, i_epoch=0):
        batch = None
        final_rewards = None
        while batch is None:

            h_in = self.init_type((1, self.batch_size, self.env.hidden_size))
            c_in = self.init_type((1, self.batch_size, self.env.hidden_size))

            state = self.env.reset()
            past_done = torch_zeros((self.batch_size, 1))
            horizon = torch_ones((self.batch_size, 1))

            transitions = [[] for _ in range(self.batch_size)]
            for t in range(self.env.max_horizon):
                # Select an action

                with torch_inference_mode():
                    if self.env.observe_hidden_state:
                        state['h'] = h_in.reshape((self.batch_size, 1, self.env.hidden_size)).detach().numpy()
                        state['c'] = c_in.reshape((self.batch_size, 1, self.env.hidden_size)).detach().numpy()
                    action, log_prob, entropy, h_out, c_out, _ = self.policy.select_action(state, h_in, c_in)

                # Perform the selected action and compute the corresponding next_state and reward
                next_state, done = self.env.step(action)

                for i in range(self.batch_size):
                    if past_done[i] != 1:
                        transition = [{k: v[i] for k, v in state.items()},
                                      h_in[0, i],
                                      c_in[0, i],
                                      action[i], past_done[i], done[i]]
                        transitions[i].append(transition)
                        if done[i] == 1:
                            horizon[i] = t

                # Update step information
                past_done = torch_Tensor(done)
                state = next_state
                h_in = torch_Tensor(h_out)
                c_in = torch_Tensor(c_out)

                if done.sum() == self.batch_size:
                    break

            final_rewards = self.env.compute_final_reward()
            for i in range(self.batch_size):
                for j in range(len(transitions[i])):
                    transitions[i][j] += [final_rewards[i]]

            if final_rewards[final_rewards > 0].sum() == 0:
                continue

            batch = transitions
            batch_max = final_rewards.max()
            if batch_max > self.logger['best_reward']:
                i_best_reward = np.argmax(final_rewards)
                self.logger.update({'best_expression': self.env.translations[i_best_reward],
                                    "best_reward": batch_max,
                                    "i_best_epoch": i_epoch})
                if self.verbose:
                    print(f'Found {self.logger["best_expression"]} at epoch {self.logger["i_best_epoch"]} '
                          f'with reward {self.logger["best_reward"]}'
                          f'horizon {sum(self.env.done[i_best_reward])}', flush=True)

            if self.verbose:
                # Print batch stats
                self.writer.add_scalar('Batch/Mean', final_rewards.mean(), i_epoch)
                self.writer.add_scalar('Batch/Std', final_rewards.std(), i_epoch)
                self.writer.add_scalar('Batch/Max', final_rewards.max(), i_epoch)
                self.writer.add_scalar('Batch/Risk Eps Quantile',
                                       np.quantile(final_rewards, 1 - self.risk_eps), i_epoch)

                # Print debug elements
                if self.debug:
                    all_expr = " \n".join(self.env.translations)
                    self.writer.add_text("Batch/Sampled Expressions", all_expr, global_step=i_epoch)

                    if i_epoch % 10 == 0:
                        for name, weight in self.policy.named_parameters():
                            self.writer.add_histogram(f"Weigths/{name}", weight, global_step=i_epoch)
                        self.writer.flush()

        return batch, final_rewards

    def optimize_model(self, batch, final_rewards, i_epoch):
        top_epsilon_quantile = np.quantile(final_rewards, 1 - self.risk_eps)
        top_filter = final_rewards >= top_epsilon_quantile
        num_samples = sum(top_filter)

        def filter_top_epsilon(b, filter):
            filtered_state = {k: [] for k in self.env.observation_space.spaces.keys()}
            filtered_h_in, filtered_c_in, filtered_action, filtered_done, filtered_rewards = [], [], [], [], []
            for indices in np.where(filter)[0]:
                for step in b[indices]:
                    s, h, c, a, _, d, r = step
                    for k in self.env.observation_space.spaces.keys():
                        filtered_state[k].append(s[k])
                    filtered_h_in.append(h)
                    filtered_c_in.append(c)
                    filtered_action.append(a)
                    filtered_done.append(d)
                    filtered_rewards.append(r)

            for k in self.env.observation_space.spaces.keys():
                filtered_state[k] = torch_Tensor(filtered_state[k])

            if filtered_h_in != []:
                filtered_h_in = torch_vstack(filtered_h_in).unsqueeze(0)
                filtered_c_in = torch_vstack(filtered_c_in).unsqueeze(0)
                filtered_action = torch_vstack(filtered_action)
                filtered_done = torch_Tensor(filtered_done)
                filtered_rewards = torch_Tensor(filtered_rewards)
            return filtered_state, filtered_h_in, filtered_c_in, filtered_action, filtered_done, filtered_rewards

        # Filter top trajectories
        state, h_in, c_in, action, done, rewards = filter_top_epsilon(batch, top_filter)
        if h_in == []:
            return

        # reset gradients
        self.optimizer.zero_grad()

        # Perform forward pass
        action_logits, aaa, bbb, other_predictions = self.policy.forward(state, h_in, c_in)
        inputs_hat, score_estimations = other_predictions
        m = CategoricalMasked(logits=action_logits, masks=torch_BoolTensor(state['current_mask'].detach().numpy()))

        # compute log_probs
        log_probs = m.log_prob(action)[:, 0]
        entropy = m.entropy()

        # Compute loss
        policy_loss = - torch_mul(log_probs, rewards - top_epsilon_quantile) + self.get_bonus(rewards,
                                                                                              log_probs,
                                                                                              num_samples)
        score_error = 0
        if self.reward_prediction:
            score_estimation = torch_gather(score_estimations.squeeze(1), dim=1, index=action)
            score_error = nn_MSELoss()(torch_mul(score_estimation, done), torch_mul(rewards, done))

        # sum up all the values of policy_losses and value_losses
        loss = policy_loss.mean() - self.entropy_coeff * entropy.mean() + 0.0001 * score_error
        if self.policy.autoencoder:
            ae_loss = 0
            criterion = nn_MSELoss()
            for k, state_k in state.items():
                ae_loss += criterion(inputs_hat[k], state_k)

            ae_loss /= len(list(state.keys()))
            loss = (1 - self.policy.ae_coeff_loss) * loss + self.policy.ae_coeff_loss * ae_loss

        # perform backprop
        loss.backward()

        torch_nn_utils_clip_grad_norm_(self.policy.parameters(), 1)
        self.optimizer.step()

        if self.verbose:
            self.writer.add_scalar('Losses/Loss', loss.detach().numpy(), i_epoch)
            self.writer.add_scalar('Losses/Entropy Loss', entropy.mean().detach().numpy(), i_epoch)
            self.writer.add_scalar('Losses/Policy Loss', policy_loss.mean().detach().numpy(), i_epoch)
            if self.policy.autoencoder:
                self.writer.add_scalar('Losses/Autoencoder Loss', ae_loss.detach().numpy(), i_epoch)
                self.writer.add_scalar('Losses/Weight a', self.policy.ae_coeff_loss.detach().numpy(), i_epoch)

    def get_bonus(self, total_rewards, total_log_probs, num_samples=0):
        return 0
