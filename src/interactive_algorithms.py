# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression

import os
import sys
import pickle
import time
import random
import numpy as np
import warnings
import itertools
warnings.filterwarnings("ignore")

import torch
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)
from tabulate import tabulate

from algorithms import ReinforceAlgorithm
from utils.masking_categorical import CategoricalMasked
from user_behavior import RealUser, SelectBestRewardUser

class DemonstrationReinforceAlgorithm(ReinforceAlgorithm):
    def __init__(self, nb_suggestions=2, **kwargs):
        super(DemonstrationReinforceAlgorithm, self).__init__(**kwargs)

        self.writer = SummaryWriter(comment=f"Interactive_Reinforce_experiment_{self.dataset}_{time.time()}")
        self.nb_suggestions = nb_suggestions
        self.interactive_indices = np.random.choice(a=self.batch_size, size=self.nb_suggestions)

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

                    action, log_prob, entropy, h_out, c_out, _ = self.select_action_interactively(state, h_in, c_in)

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
            if self.verbose:
                batch_max = final_rewards.max()
                if batch_max > self.logger['best_reward']:
                    i_best_reward = np.argmax(final_rewards)
                    self.logger.update({'best_expression': self.env.translations[i_best_reward],
                                        "best_reward": batch_max,
                                        "i_best_epoch": i_epoch})
                    print(f'Found {self.logger["best_expression"]} at epoch {self.logger["i_best_epoch"]} '
                          f'with reward {self.logger["best_reward"]}'
                          f'horizon {horizon[i_best_reward]}', flush=True)

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

    def select_action_interactively(self, state, h_in, c_in):
        action_logits, h_out, c_out, other_predictions = self.policy.forward(state, h_in, c_in)

        # create a categorical distribution over the list of probabilities of actions
        m = CategoricalMasked(logits=action_logits, masks=torch.BoolTensor(state['current_mask']))

        # and sample an action using the distribution
        action = m.sample()
        for i in self.interactive_indices:

            current_symbol = None
            for k, v in self.env.grammar.symbol_encoding.items():
                if np.array_equal(v, state["current_symbol"][i]):
                    current_symbol = k
                    break
            if current_symbol != "#":
                action[i] = self.suggest(i, self.env.translations[i], state["current_mask"][i], current_symbol)

        # compute log_probs
        log_probs = m.log_prob(action)
        entropy = m.entropy()

        return action, log_probs, entropy, h_out, c_out, other_predictions

    def suggest(self, i, translation, mask, current_symbol):
        print()
        print("______________________________________________")

        print(f"Individual n°{i}{translation}")

        print(f"Looking to replace symbol {current_symbol}")
        print('Possible rules : ')

        options = []
        for i in np.argwhere(mask[0]==1):
            rule = self.env.grammar.productions_list[i[0]]
            print(f"- rule n° {i[0]} {rule['raw']}")
            options.append(i[0])

        ask_for_user = True
        if len(options) == 1 :
            action = options[0]
            ask_for_user = False
        while ask_for_user:
            try:
                action = int(input(
                    f'What rule to select ? ({", ".join([str(o) for o in options[:-1]]) + " or " + str(options[-1])})'))
                if action not in options:
                    print("Invalid action, please select a correct one")
                    continue
                ask_for_user = False
            except Exception as e:
                print('Invalid action, please select a correct one')
                continue

        return action


class PreferenceReinforceGUI(ReinforceAlgorithm):
    def __init__(self, user=SelectBestRewardUser(),  x_label="x", interaction_type="from_start", **kwargs):
        super(PreferenceReinforceGUI, self).__init__(**kwargs)
        self.gui_data_path = os.path.join(self.writer.log_dir, 'gui_data')
        os.makedirs(self.gui_data_path, exist_ok=True)

        self.gui_answers = None
        self.user = user

        # params to store preferences
        self.final_rewards = None
        self.past_trajectories = None

        # Parameters used to combine REINFORCE algorithm with interactivity every n steps
        self.interaction_type = interaction_type

        self.n_reinforce_step = 0
        self.remaining_reinforce_iteration = self.n_reinforce_step
        self.apply_reinforce = False
        self.x_label = x_label

    def create_summary_writer(self):
        return SummaryWriter(log_dir=self.writer_logdir,
                                    comment=f"Preference_with_GUI_Reinforce_experiment_{self.dataset}_{time.time()}")

    @torch.inference_mode()
    def sample_episodes(self, i_epoch=0):
        if (i_epoch % self.user.interaction_frequency != 0) & self.user.reuse & (self.past_trajectories is not None):
            print(f'Epoch n° {i_epoch}: reuse past trajectories')
            suggested_trajectories = [{"action_ids": t} for t in self.past_trajectories]
            transitions, final_rewards, _ = self.simulate_trajectories(suggested_trajectories)
            return transitions, final_rewards

        print(f'Epoch n° {i_epoch}: sample new trajectories')
        batch = None
        final_rewards = None
        self.past_trajectories = [[] for _ in range(self.batch_size)]
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
                        self.past_trajectories[i] += [int(action[i][0])]
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
            if self.verbose:
                batch_max = final_rewards.max()
                if batch_max > self.logger['best_reward']:
                    i_best_reward = np.argmax(final_rewards)
                    self.logger.update({'best_expression': self.env.translations[i_best_reward],
                                        "best_reward": batch_max,
                                        "i_best_epoch": i_epoch})
                    print(f'Found {self.logger["best_expression"]} at epoch {self.logger["i_best_epoch"]} '
                          f'with reward {self.logger["best_reward"]}'
                          f'horizon {horizon[i_best_reward]}', flush=True)

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

        self.final_rewards = final_rewards
        return batch, final_rewards

    def optimize_model(self, batch, final_rewards, i_epoch):
        if (self.apply_reinforce & (self.remaining_reinforce_iteration <= 0)):
            print('Use Reinforce')
            super(PreferenceReinforceGUI, self).optimize_model(batch, final_rewards, i_epoch)
            self.remaining_reinforce_iteration -= 1

        else:
            self.optimize_model_with_preference(batch, final_rewards, i_epoch)
            self.remaining_reinforce_iteration = self.n_reinforce_step

    def optimize_model_with_preference(self, batch, final_rewards, i_epoch):
        top_epsilon_quantile = np.quantile(final_rewards, 1 - self.risk_eps)

        # Interactivity here : compare pairs of trajectories
        preferences_indices, preference_probs, simulated_rewards, simulated_transitions = \
            self.ask_for_preferences(top_epsilon_quantile, final_rewards, i_epoch)

        if simulated_transitions is not None:
            final_rewards += simulated_rewards
            batch += simulated_transitions

        self.optimize(batch, preferences_indices, preference_probs, top_epsilon_quantile, i_epoch)

    def ask_for_preferences(self, top_epsilon_quantile, final_rewards, i_epoch):
        #unique_indexes = np.array([self.env.translations.index(x) for x in set(self.env.translations)])
        top_indices = np.argwhere(final_rewards >= top_epsilon_quantile)[:, 0]
        #indices_to_compare = np.intersect1d(top_indices, unique_indexes)
        try:
            combinaisons_to_compare = random.choices(list(itertools.combinations(top_indices, 2)), k=5)
        except Exception as e:
            print(e)

        gui_infos = None
        if isinstance(self.user, RealUser):

            # save current population to file
            gui_infos = {"combinaisons": combinaisons_to_compare,
                         "translations": self.env.translations,
                         "x": self.env.X_test[self.x_label],
                         "predicted_values": self.env.get_predicted_values(),
                         "target_values": self.env.y_test,
                         "top_indices": top_indices,
                         "rewards": final_rewards,
                         'grammar': {'productions_list': self.env.grammar.productions_list,
                                     'productions_dict': self.env.grammar.productions_dict,
                                     "start_symbol": self.env.start_symbol}}
        else:
            gui_infos = {"combinaisons": combinaisons_to_compare,
                         "rewards": final_rewards,
                         "translations": self.env.translations}
        gui_answers = self.user.select_preference(gui_infos, i_epoch)

        self.gui_answers = gui_answers
        return self.use_preferences(gui_answers, final_rewards)

    def use_preferences(self, gui_answers, final_rewards):
        suggestions = gui_answers["suggest"]

        if isinstance(suggestions, dict):
            suggestions = [suggestions]

        pairs_ids = gui_answers["pairs"]['ids']
        answers = gui_answers["pairs"]['answers']

        if suggestions != []:
            with torch.inference_mode():
                simulated_transitions, simulated_rewards, simulated_translations = self.simulate_trajectories(suggestions)
        else :
            simulated_transitions, simulated_rewards, simulated_translations = None, [], None

        preferences_indices = []
        preference_probs = []

        for sim_i, sim_reward in enumerate(simulated_rewards):
            id_comparison = suggestions[sim_i]["comparison_with_id"]
            prob_comparison = np.exp(sim_reward) / (np.exp(sim_reward) + np.exp(final_rewards[id_comparison]))
            preference_probs = [prob_comparison]
            preferences_indices = [len(final_rewards) + sim_i]

        for i_top in gui_answers["classes"]['top']:
            for i_middle_to_low in gui_answers["classes"]['middle'] + gui_answers["classes"]['low']:
                if ([i_top, i_middle_to_low] not in pairs_ids) and ([i_middle_to_low, i_top] not in pairs_ids):
                    pairs_ids.append([i_top, i_middle_to_low])
                    answers.append("l")
        for i_middle in gui_answers["classes"]['middle']:
            for i_low in gui_answers["classes"]['low']:
                if ([i_middle, i_low] not in pairs_ids) and ([i_low, i_middle] not in pairs_ids):
                    pairs_ids.append([i_middle, i_low])
                    answers.append("l")

        for combinaison, answer in zip(pairs_ids, answers):
            id_left, id_right = combinaison
            expressions_data = [self.env.translations[id_left], self.env.translations[id_right]]
            rewards_data = [final_rewards[id_left], final_rewards[id_right]]
            prob_left = np.exp(final_rewards[id_left])/(np.exp(final_rewards[id_right]) + np.exp(final_rewards[id_left]))
            prob_right = np.exp(final_rewards[id_right])/(np.exp(final_rewards[id_right]) + np.exp(final_rewards[id_left]))

            if (answer == "right") or (answer == "r"):
                preferences_indices += [id_right]
                preference_probs += [prob_right]
            elif (answer == "left") or (answer == "l"):
                preferences_indices += [id_left]
                preference_probs += [prob_left]
            elif (answer == "both") or (answer == "b"):
                preferences_indices += [id_left, id_right]
                preference_probs += [1/2*prob_left, 1/2*prob_right]

        return preferences_indices, torch.Tensor(preference_probs), simulated_rewards, simulated_transitions

    def simulate_trajectories(self, suggested_trajectories):
        nb_suggestions = len(suggested_trajectories)
        h_in = self.init_type((1, nb_suggestions, self.env.hidden_size))
        c_in = self.init_type((1, nb_suggestions, self.env.hidden_size))

        state, suggestions_infos = self.env.simulate_reset(nb_suggestions)
        past_done = torch.zeros((nb_suggestions, 1))
        horizon = torch.ones((nb_suggestions, 1))

        transitions = [[] for _ in range(nb_suggestions)]
        for h in range(self.env.max_horizon):
            if suggested_trajectories:
                suggested_actions = torch.Tensor([
                    {True: suggestion['action_ids'][h%len(suggestion['action_ids'])], False: 0}[h < len(suggestion['action_ids'])]
                    for suggestion in suggested_trajectories])

            if self.env.observe_hidden_state:
                state['h'] = h_in.reshape((nb_suggestions, 1, self.env.hidden_size)).detach().numpy()
                state['c'] = c_in.reshape((nb_suggestions, 1, self.env.hidden_size)).detach().numpy()
            action, log_prob, entropy, h_out, c_out, _ = self.policy.select_action(state, h_in, c_in, suggested_actions)

            next_state, done, suggestions_infos = self.env.simulate_step(suggested_actions,
                                                                         nb_suggestions,
                                                                         suggestions_infos)

            for i in range(nb_suggestions):
                if past_done[i] != 1:
                    transition = [{k: v[i] for k, v in state.items()},
                                  h_in[0, i],
                                  c_in[0, i],
                                  action[i], past_done[i], done[i]]
                    transitions[i].append(transition)
                    if done[i] == 1:
                        horizon[i] = h

            # Update step information
            past_done = torch.Tensor(done)
            state = next_state
            h_in = torch.Tensor(h_out)
            c_in = torch.Tensor(c_out)
            if done.sum() == nb_suggestions:
                break

        #assert suggestions_infos["translations"] == self.env.translations
        final_rewards = self.env.simulate_reward(nb_suggestions, suggestions_infos["translations"])
        for i in range(nb_suggestions):
            for j in range(len(transitions[i])):
                transitions[i][j] += [final_rewards[i]]

        return transitions, final_rewards, suggestions_infos["translations"]

    def optimize(self, batch, preferences_indices, preference_probs, top_epsilon_quantile, i_epoch):
        def filter_combinaisons(b, indices, preferences):
            filtered_state = {k: [] for k in self.env.observation_space.spaces.keys()}
            filtered_h_in, filtered_c_in, filtered_action, filtered_done, filtered_rewards = [], [], [], [], []
            filtered_preferences = []
            for i_in_pref_list, i in enumerate(indices):
                for step in b[i]:
                    s, h, c, a, _, d, r = step
                    for k in self.env.observation_space.spaces.keys():
                        filtered_state[k].append(s[k])
                    filtered_h_in.append(h)
                    filtered_c_in.append(c)
                    filtered_action.append(a)
                    filtered_done.append(d)
                    filtered_rewards.append(r)
                    filtered_preferences.append(preferences[i_in_pref_list])

            for k in self.env.observation_space.spaces.keys():
                filtered_state[k] = torch.Tensor(filtered_state[k])

            if filtered_h_in != []:
                filtered_h_in = torch.vstack(filtered_h_in).unsqueeze(0)
                filtered_c_in = torch.vstack(filtered_c_in).unsqueeze(0)
                filtered_action = torch.vstack(filtered_action)
                filtered_done = torch.Tensor(filtered_done)
                filtered_rewards = torch.Tensor(filtered_rewards)
                filtered_preferences = torch.Tensor(filtered_preferences)
            return filtered_state, filtered_h_in, filtered_c_in, filtered_action, filtered_done, filtered_rewards,\
                   filtered_preferences

        # Filter top trajectories
        state, h_in, c_in, action, done, rewards, human_probs = filter_combinaisons(batch,
                                                                                    preferences_indices,
                                                                                    preference_probs)
        if h_in == []:
            return

        # reset gradients
        self.optimizer.zero_grad()

        # Perform forward pass
        action_logits, _, _, other_predictions = self.policy.forward(state, h_in, c_in)

        inputs_hat, score_estimations = other_predictions
        m = CategoricalMasked(logits=action_logits, masks=torch.BoolTensor(state['current_mask'].detach().numpy()))

        # compute log_probs
        log_probs = m.log_prob(action)[:, 0]
        entropy = m.entropy()

        # Compute loss
        policy_loss = - torch.mul(log_probs-torch.log(human_probs), rewards - top_epsilon_quantile).mean()

        entropy = m.entropy()
        loss = policy_loss.mean() - self.entropy_coeff * entropy.mean()

        # perform backprop
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.optimizer.step()

        self.writer.add_scalar('Losses/Loss', loss.detach().numpy(), i_epoch)
        self.writer.add_scalar('Losses/Policy Loss', policy_loss.sum().detach().numpy(), i_epoch)


class PreferenceReinforceAlgorithm(PreferenceReinforceGUI):
    def __init__(self, **kwargs):
        super(PreferenceReinforceAlgorithm, self).__init__(**kwargs)

        self.preferences = {}

    def create_summary_writer(self):
        return SummaryWriter(log_dir=self.writer_logdir,
                             comment=f"Preference_Reinforce_experiment_{self.dataset}_{time.time()}")

    def ask_for_preferences(self, top_epsilon_quantile, final_rewards, i_epoch):
        unique_indexes = np.array([self.env.translations.index(x) for x in set(self.env.translations)])
        top_indices = np.argwhere(final_rewards > top_epsilon_quantile)
        indices_to_compare = np.intersect1d(top_indices, unique_indexes)
        combinaisons_to_compare = random.choices(list(itertools.combinations(indices_to_compare, 2)), k=10)

        preferences_indices = []
        preference_probs = []
        for combination_number_i, combinaison in enumerate(combinaisons_to_compare):
            id_left, id_right = combinaison
            expressions_data = [self.env.translations[id_left], self.env.translations[id_right]]
            rewards_data = [self.final_rewards[id_left], self.final_rewards[id_right]]
            prob_left = np.exp(self.final_rewards[id_left])/(np.exp(self.final_rewards[id_right]) + np.exp(self.final_rewards[id_left]))
            prob_right = np.exp(self.final_rewards[id_right])/(np.exp(self.final_rewards[id_right]) + np.exp(self.final_rewards[id_left]))

            correct_answer = False
            while not correct_answer:
                print()
                print(f"Combination n°{combination_number_i}")
                print(tabulate([expressions_data, rewards_data],
                               headers=['Left Expression', 'Right Expression'],
                               tablefmt="presto"))
                input_text = 'Which one do you prefer ? \n' \
                             '- "right" (or "r"),\n' \
                             '- "left" (or "l"),\n' \
                             '- "both" (or "b"), meaning it\'s a tie \n' \
                             '- "none" (or"n"), meaning I can\'t tell \n' \
                             'Answer : '
                if (self.env.translations[id_right], self.env.translations[id_left]) in self.preferences.keys():
                    input_text = f'Past preference in history : {self.preferences[(self.env.translations[id_right], self.env.translations[id_left])]}\n' + input_text
                elif (self.env.translations[id_left], self.env.translations[id_right]) in self.preferences.keys():
                    input_text = f'Past preference in history : {self.preferences[(self.env.translations[id_left], self.env.translations[id_right])]}\n' + input_text
                answer = input(input_text)
                if self.final_rewards[id_right] > self.final_rewards[id_left]:
                    answer = 'r'
                elif self.final_rewards[id_right] < self.final_rewards[id_left]:
                    answer = "l"

                if (answer == "right") or (answer == "r"):
                    preferences_indices += [id_right]
                    preference_probs += [prob_right]
                    correct_answer = True
                elif (answer == "left") or (answer == "l"):
                    preferences_indices += [id_left]
                    preference_probs += [prob_left]
                    correct_answer = True
                elif (answer == "both") or (answer == "b"):
                    preferences_indices += [id_left, id_right]
                    preference_probs += [1/2*prob_left, 1/2*prob_right]
                    correct_answer = True
                elif (answer == "none") or (answer == "n"):
                    correct_answer = True
                else:
                    print("Incorrect answer")

            if self.env.translations[id_right] > self.env.translations[id_left]:
                self.preferences[(self.env.translations[id_right], self.env.translations[id_left])] = answer
            else:
                self.preferences[(self.env.translations[id_right], self.env.translations[id_left])] = answer

        return preferences_indices, torch.Tensor(preference_probs), [], []
