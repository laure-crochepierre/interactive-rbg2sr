# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

from collections import namedtuple
from abc import ABC, abstractmethod

from policies import DqnPolicy
from utils.masking_categorical import CategoricalMasked
from utils.replay_buffers import ReplayMemory, Transition
Batch = namedtuple('Batch', ('state', 'h', 'c', 'action', 'past_done', 'final_reward'))


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM):
        for p in m.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.uniform_(p.data)


class BaseAlgorithm(ABC):
    def __init__(self,
                 env_class,
                 env_kwargs,
                 policy_class,
                 policy_kwargs,
                 risk_seeking=True,
                 risk_eps=0.05,
                 entropy_coeff=0,
                 optimizer_class=torch.optim.Adam,
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

        init_functions = {'zeros': lambda size: torch.zeros(size, dtype=torch.float32),
                          "randint": lambda size: torch.randint(-1, 1, size, dtype=torch.float32)}
        self.init_type = init_functions[init_type]

        # Logs
        self.writer_logdir=writer_logdir
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
    @torch.inference_mode()
    def sample_episodes(self, i_epoch=0):
        return NotImplementedError

    @abstractmethod
    def optimize_model(self, batch, final_rewards, i_epoch):
        return NotImplementedError


class ReinforceAlgorithm(BaseAlgorithm):
    def __init__(self, **kwargs):
        super(ReinforceAlgorithm, self).__init__(**kwargs)
        self.writer = self.create_summary_writer()

    def create_summary_writer(self):
        return SummaryWriter(log_dir=self.writer_logdir,
                             comment=f"Reinforce_experiment_{self.dataset}_{time.time()}")

    @torch.inference_mode()
    def sample_episodes(self, i_epoch=0):
        batch = None
        final_rewards = None
        while batch is None:

            h_in = self.init_type((1, self.batch_size, self.env.hidden_size))
            c_in = self.init_type((1, self.batch_size, self.env.hidden_size))

            state = self.env.reset()
            past_done = torch.zeros((self.batch_size, 1))
            horizon = torch.ones((self.batch_size, 1))

            transitions = [[] for _ in range(self.batch_size)]
            for t in range(self.env.max_horizon):
                # Select an action

                with torch.inference_mode():
                    if self.env.observe_hidden_state :
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
                past_done = torch.Tensor(done)
                state = next_state
                h_in = torch.Tensor(h_out)
                c_in = torch.Tensor(c_out)

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
                                       np.quantile(final_rewards, 1-self.risk_eps), i_epoch)

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
                filtered_state[k] = torch.Tensor(filtered_state[k])

            if filtered_h_in != []:
                filtered_h_in = torch.vstack(filtered_h_in).unsqueeze(0)
                filtered_c_in = torch.vstack(filtered_c_in).unsqueeze(0)
                filtered_action = torch.vstack(filtered_action)
                filtered_done = torch.Tensor(filtered_done)
                filtered_rewards = torch.Tensor(filtered_rewards)
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
        m = CategoricalMasked(logits=action_logits, masks=torch.BoolTensor(state['current_mask'].detach().numpy()))

        # compute log_probs
        log_probs = m.log_prob(action)[:, 0]
        entropy = m.entropy()

        # Compute loss
        policy_loss = - torch.mul(log_probs, rewards - top_epsilon_quantile) + self.get_bonus(rewards,
                                                                                              log_probs,
                                                                                              num_samples)
        score_error = 0
        if self.reward_prediction:
            score_estimation = torch.gather(score_estimations.squeeze(1), dim=1, index=action)
            score_error = nn.MSELoss()(torch.mul(score_estimation, done), torch.mul(rewards, done))

        # sum up all the values of policy_losses and value_losses
        loss = policy_loss.mean() - self.entropy_coeff * entropy.mean() + 0.0001 * score_error
        if self.policy.autoencoder:
            ae_loss = 0
            criterion = nn.MSELoss()
            for k, state_k in state.items():
                ae_loss += criterion(inputs_hat[k], state_k)

            ae_loss /= len(list(state.keys()))
            loss = (1 - self.policy.ae_coeff_loss) * loss + self.policy.ae_coeff_loss * ae_loss

        # perform backprop
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
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


class UREXAlgorithm(ReinforceAlgorithm):
    urex_tau = 0.1

    def get_bonus(self, total_rewards, total_log_probs, num_samples):
        """Exploration bonus."""
        discrepancy = total_rewards / self.urex_tau - total_log_probs
        normalized_d = num_samples * torch.softmax(discrepancy, 0)
        return self.urex_tau * normalized_d


class ActorCriticAlgorithm(BaseAlgorithm):
    def __init__(self,
                 dataset,
                 **kwargs):
        super(ActorCriticAlgorithm, self).__init__(**kwargs)

        # Define logs
        self.writer = SummaryWriter(comment=f"ActorCritic_experiment_{dataset}_{time.time()}")

    @torch.inference_mode()
    def sample_episodes(self, i_epoch=0):
        batch = None
        final_rewards = None
        while batch is None:

            h_in = self.init_type((1, self.batch_size, self.env.hidden_size))
            c_in = self.init_type((1, self.batch_size, self.env.hidden_size))

            state = self.env.reset()
            past_done = torch.zeros((self.batch_size, 1))

            transitions = [[] for _ in range(self.batch_size)]
            for t in range(self.env.max_horizon):
                # Select an action

                with torch.inference_mode():
                    if self.env.observe_hidden_state :
                        state['h'] = h_in.reshape((self.batch_size, 1, self.env.hidden_size)).detach().numpy()
                        state['c'] = c_in.reshape((self.batch_size, 1, self.env.hidden_size)).detach().numpy()
                    action, log_prob, entropy, h_out, c_out, _, critic_value = self.policy.select_action(state, h_in, c_in)

                # Perform the selected action and compute the corresponding next_state and reward
                next_state, done = self.env.step(action)

                for i in range(self.batch_size):
                    if past_done[i] != 1:
                        transition = [{k: v[i] for k, v in state.items()},
                                       h_in[0, i],
                                       c_in[0, i],
                                       action[i], past_done[i]]
                        transitions[i].append(transition)

                # Update step information
                past_done = torch.Tensor(done)
                state = next_state
                h_in = torch.Tensor(h_out)
                c_in = torch.Tensor(c_out)

                if self.debug == 3:
                    df = pd.DataFrame()
                    df.loc[:, '<action>'] = [self.env.grammar.productions_list[a]['raw'] for a in action]
                    df.loc[:, 'probs'] = np.exp(log_prob)
                    df.sort_values('<action>', inplace=True)

                    f, ax = plt.subplots(figsize=(7, 6))
                    sns.boxplot(x="probs", y="<action>", data=df,
                                whis=[0, 100], width=.6, palette="vlag")

                    sns.stripplot(x="probs", y="<action>", data=df,
                                  size=4, color=".3", linewidth=0)

                    # Tweak the visual presentation
                    ax.xaxis.grid(True)
                    ax.set(ylabel="")
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=45)
                    #plt.yticks([prod['raw'] for prod in self.env.grammar.productions_list])
                    sns.despine(trim=True, left=True)

                    self.writer.add_figure(f"Histogram step {t}", figure=f, global_step=i_epoch)

                if done.sum() == self.batch_size:
                    break

            final_rewards = self.env.compute_final_reward()
            for i in range(self.batch_size):
                for j in range(len(transitions[i])):
                    transitions[i][j] += [critic_value[i], final_rewards[i]]

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
                                       np.quantile(final_rewards, 1-self.risk_eps), i_epoch)

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

        def filter_top_epsilon(b, filter):
            filtered_state = {k: [] for k in self.env.observation_space.spaces.keys()}
            filtered_h_in, filtered_c_in, filtered_action, filtered_critic, filtered_rewards = [], [], [], [], []
            for indices in np.where(filter)[0]:
                for step in b[indices]:
                    s, h, c, a, _, k, r = step
                    for key in self.env.observation_space.spaces.keys():
                        filtered_state[key].append(s[key])
                    filtered_h_in.append(h)
                    filtered_c_in.append(c)
                    filtered_action.append(a)
                    filtered_critic.append(k)
                    filtered_rewards.append(r)

            for k in self.env.observation_space.spaces.keys():
                filtered_state[k] = torch.Tensor(filtered_state[k])

            filtered_h_in = torch.vstack(filtered_h_in).unsqueeze(0)
            filtered_c_in = torch.vstack(filtered_c_in).unsqueeze(0)
            filtered_action = torch.vstack(filtered_action)
            filtered_critic = torch.Tensor(filtered_critic)
            filtered_rewards = torch.Tensor(filtered_rewards)
            return filtered_state, filtered_h_in, filtered_c_in, filtered_action, filtered_critic, filtered_rewards
        # Filter top trajectories
        state, h_in, c_in, action, critic, rewards = filter_top_epsilon(batch, top_filter)

        # reset gradients
        self.optimizer.zero_grad()

        # Perform forward pass
        action_logits, _, _, inputs_hat, critic_logits = self.policy.forward(state, h_in, c_in)
        m = CategoricalMasked(logits=action_logits, masks=torch.BoolTensor(state['current_mask'].detach().numpy()))

        # compute log_probs
        log_probs = m.log_prob(action)[:, 0]
        entropy = m.entropy()

        # Compute loss
        policy_loss = - torch.mul(log_probs, rewards - top_epsilon_quantile)

        # Critic loss
        critic_loss = nn.MSELoss()(rewards, critic)

        # sum up all the values of policy_losses and value_losses
        loss = policy_loss.mean() - self.entropy_coeff * entropy.mean() + critic_loss
        if self.policy.autoencoder:
            ae_loss = 0
            criterion = nn.MSELoss()
            for k, state_k in state.items():
                ae_loss += criterion(inputs_hat[k], state_k)
            loss = (1-self.policy.ae_coeff_loss) * loss + self.policy.ae_coeff_loss * ae_loss

        # perform backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.optimizer.step()

        if self.debug == 2:
            for key, value in self.policy._modules.items():
                try:
                    print(key, value.weight.grad, flush=True)
                except Exception as e:
                    try:
                        print(key, value.all_weights, flush=True)
                    except Exception as e:
                        print(e, flush=True)
        self.writer.add_scalar('Losses/Loss', loss.detach().numpy(), i_epoch)
        self.writer.add_scalar('Losses/Entropy Loss', entropy.mean().detach().numpy(), i_epoch)
        self.writer.add_scalar('Losses/Policy Loss', policy_loss.mean().detach().numpy(), i_epoch)
        self.writer.add_scalar('Losses/Critic Loss', critic_loss.mean().detach().numpy(), i_epoch)

        if self.policy.autoencoder:
            self.writer.add_scalar('Losses/Autoencoder Loss', ae_loss.detach().numpy(), i_epoch)
            self.writer.add_scalar('Losses/Weight a', self.policy.ae_coeff_loss.detach().numpy(), i_epoch)


class DQN(BaseAlgorithm):
    def __init__(self,
                 policy_class,
                 policy_kwargs,
                 memory_size=100000,
                 target_update=100,
                 gamma=0.999,
                 tau=0.001,
                 **kwargs):
        if policy_class != DqnPolicy:
            raise TypeError("DQN algorithm must have DqnPolicy as policy_class")
        super(DQN, self).__init__(policy_class=policy_class, policy_kwargs=policy_kwargs, **kwargs)
        policy_kwargs.update(dict(observation_space=self.env.observation_space,
                                  action_space=self.env.action_space,
                                  max_horizon=self.env.max_horizon))
        self.target = policy_class(**policy_kwargs)

        self.replay_memory = ReplayMemory(memory_size)
        self.target.load_state_dict(self.policy.state_dict())
        self.target_update = target_update
        self.gamma = gamma
        self.tau = tau

        # Define logs
        self.writer = SummaryWriter(comment=f"DQN_experiment_{self.dataset}_{time.time()}")

    @torch.inference_mode()
    def sample_episodes(self, i_epoch=0):
        batch = None
        final_rewards = None

        while batch is None:

            h_in = self.init_type((1, self.batch_size, self.env.hidden_size))
            c_in = self.init_type((1, self.batch_size, self.env.hidden_size))

            state = self.env.reset()
            if self.env.observe_hidden_state:
                state['h'] = h_in.reshape((self.batch_size, 1, self.env.hidden_size)).detach().numpy()
                state['c'] = c_in.reshape((self.batch_size, 1, self.env.hidden_size)).detach().numpy()
            past_done = torch.zeros((self.batch_size, 1))

            transitions = [[] for _ in range(self.batch_size)]
            for t in range(self.env.max_horizon):
                # Select an action

                with torch.inference_mode():
                    action, q_value, h_out, c_out, _ = self.policy.select_action(state, h_in, c_in)

                # Perform the selected action and compute the corresponding next_state and reward
                next_state, done = self.env.step(action.detach().numpy().astype(np.int64))

                if self.env.observe_hidden_state:
                    next_state['h'] = h_out.reshape((self.batch_size, 1, self.env.hidden_size)).detach().numpy()
                    next_state['c'] = c_out.reshape((self.batch_size, 1, self.env.hidden_size)).detach().numpy()

                for i in range(self.batch_size):
                    if past_done[i] != 1:
                        transition = [{k: v[i] for k, v in state.items()},
                                               h_in[0, i], c_in[0, i],
                                               torch.Tensor([action[i]]),
                                              {False: {k: v[i] for k, v in next_state.items()}, True: None}[done[i, 0]],
                                               torch.Tensor(done[i])]
                        transitions[i].append(transition)

                # Update step information
                past_done = torch.Tensor(done)
                state = next_state
                h_in = torch.Tensor(h_out)
                c_in = torch.Tensor(c_out)

            final_rewards = self.env.compute_final_reward()
            for i in range(self.batch_size):
                for j in range(len(transitions[i])):
                    transitions[i][j] += [torch.Tensor([final_rewards[i]])]
                    self.replay_memory.push(*transitions[i][j])

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
                                       np.quantile(final_rewards, 1-self.risk_eps), i_epoch)

                # Print debug elements
                if self.debug:
                    all_expr = " \n".join(self.env.translations)
                    self.writer.add_text("Batch/Sampled Expressions", all_expr, global_step=i_epoch)

                    if i_epoch % 10 == 0:
                        for name, weight in self.policy.named_parameters():
                            self.writer.add_histogram(f"Weigths/{name}", weight, global_step=i_epoch)
                        self.writer.flush()

            # Sample the batch

            if len(self.replay_memory) > self.batch_size:
                tmp_batch = Transition(*zip(*self.replay_memory.sample(self.batch_size)))
                if torch.cat(tmp_batch.final_reward).max() > 0:
                    batch = tmp_batch

        return batch, final_rewards

    def optimize_model(self, batch, final_rewards, i_epoch):
        if self.risk_seeking:
            top_epsilon_quantile = np.quantile(final_rewards, 1 - self.risk_eps)
            top_filter = final_rewards >= top_epsilon_quantile

        non_final_mask = ~ torch.cat(batch.done).bool()

        non_final_next_states = {}
        for k in self.env.observation_space.spaces.keys():
            non_final_next_states[k] = torch.Tensor([s[k] for s in batch.next_state if s is not None])

        state_batch = {}
        for k in self.env.observation_space.spaces.keys():
            state_batch[k] = torch.Tensor([s[k] for s in batch.state])

        h_batch = torch.vstack(batch.h).unsqueeze(0)
        c_batch = torch.vstack(batch.c).unsqueeze(0)
        action_batch = torch.cat(batch.action)
        final_reward_batch = torch.cat(batch.final_reward)
        reward_batch = torch.mul(non_final_mask, final_reward_batch)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        v_values, _, _, inputs_hat = self.policy(state_batch, h_batch, c_batch)
        state_action_values = torch.index_select(v_values.squeeze(1), 1, action_batch.long())

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        nsv, _, _, _ = self.target(non_final_next_states,
                                h_batch[:, non_final_mask, :],
                                c_batch[:, non_final_mask, :])
        next_state_values[non_final_mask] = nsv.max(-1)[0].squeeze(1).detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        if self.risk_seeking:
            loss = criterion(state_action_values[top_filter], expected_state_action_values.unsqueeze(1)[top_filter])
        else :
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        if self.policy.autoencoder:
            ae_loss = 0
            criterion = nn.MSELoss()
            for k, state_k in state_batch.items():
                ae_loss += criterion(inputs_hat[k], state_k)
            loss = (1-self.policy.ae_coeff_loss) * loss + self.policy.ae_coeff_loss * ae_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.optimizer.step()

        if i_epoch % self.target_update == 0:
            self.soft_update()
            #self.target.load_state_dict(self.policy.state_dict())
        self.writer.add_scalar('Losses/Loss', loss.detach().numpy(), i_epoch)
        self.writer.add_scalar('Losses/eps_greedy', self.policy.epsilon_greedy, i_epoch)

        if self.policy.autoencoder:
            self.writer.add_scalar('Losses/Autoencoder Loss', ae_loss.detach().numpy(), i_epoch)
            self.writer.add_scalar('Losses/Weight a', self.policy.ae_coeff_loss.detach().numpy(), i_epoch)

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.target.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
