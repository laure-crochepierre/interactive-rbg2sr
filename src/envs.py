# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression
import gc
import re
import os

import dropbox
import pandas as pd
import numpy as np

from scipy import optimize
from numba import jit, float32
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Lasso

import gym
import gym.spaces as spaces

from utils.grammar_parser import ProbabilisticGrammar
from utils.constraints import Constraints


@jit(nopython=True)  # , float32(float32, float32)
def base_metric(y, yhat):
    return 1 / (1 + ((y - yhat) ** 2).mean())


class BatchSymbolicRegressionEnv(gym.Env):
    def __init__(self, grammar_file_path,
                 start_symbol,
                 train_data_path,
                 target,
                 test_data_path=None,
                 eval_params={},
                 metric=base_metric,
                 max_horizon=30,
                 min_horizon=4,
                 hidden_size=8,
                 batch_size=1,
                 normalize=False,
                 normalization_type="",
                 apply_constraints=False,
                 observe_brotherhood=False,
                 observe_parent=False,
                 observe_previous_actions=True,
                 observe_hidden_state=True,
                 observe_symbol=True,
                 observe_mask=True,
                 observe_depth=True,
                 outlier_heuristic=False,
                 constant_optimizer=True,
                 use_np=False,
                 ):

        # MDP related parameters
        self.max_horizon = max_horizon  # Maximal complexity of the solution
        self.min_horizon = min_horizon
        self.eval_params = eval_params  # dictionnary containing the parameters for expression evaluation
        self.batch_size = batch_size

        self.apply_constraints = apply_constraints
        self.observe_brotherhood = observe_brotherhood
        self.observe_parent = observe_parent
        self.observe_previous_actions = observe_previous_actions
        self.observe_hidden_state = observe_hidden_state
        self.observe_symbol = observe_symbol
        self.observe_mask = observe_mask
        self.observe_depth = observe_depth
        self.outlier_heuristic = outlier_heuristic

        self.target = target  # target variable to predict
        self.use_np = use_np
        # Load datasets
        self.X_train, self.X_test = None, None
        if test_data_path is not None:
            if ".feather" in train_data_path:
                if os.environ.get("DROPBOX_ACCESS_TOKEN") is not None:
                    raise AssertionError('DROPBOX_ACCESS_TOKEN environment var is set. '
                                         'You also have to choose csv files instead of feather')
                self.X_train = pd.read_feather(train_data_path)
                self.X_test = pd.read_feather(test_data_path)
            elif ".csv" in train_data_path:
                if os.environ.get("DROPBOX_ACCESS_TOKEN") is not None:

                    dbx = dropbox.Dropbox(os.environ.get('DROPBOX_ACCESS_TOKEN'))
                    _, train_bytes = dbx.files_download(train_data_path.replace('..', ''))
                    train_data = BytesIO(train_bytes.content)
                    self.X_train = pd.read_csv(train_data)
                    _, test_bytes = dbx.files_download(test_data_path.replace('..', ''))
                    test_data = BytesIO(test_bytes.content)
                    self.X_test = pd.read_csv(test_data)
                else:
                    self.X_train = pd.read_csv(train_data_path)
                    self.X_test = pd.read_csv(test_data_path)
            if normalize:
                if normalization_type == 'standard_scaler':
                    self.scaler = StandardScaler()
                    self.X_train.iloc[:, :] = self.scaler.fit_transform(self.X_train)
                    self.X_test.iloc[:, :] = self.scaler.transform(self.X_test)
                else:
                    mini = min(self.X_train.min().values.min(), self.X_test.min().values.min())
                    maxi = max(self.X_train.max().values.max(), self.X_test.max().values.max())
                    self.X_train = (self.X_train - mini) / (maxi - mini)
                    self.X_test = (self.X_test - mini) / (maxi - mini)
            self.y_train = self.X_train[target]
            self.X_train.drop(columns=[target], inplace=True)
            self.y_test = self.X_test[target]
            self.X_test.drop(columns=[target], inplace=True)
        else:
            if os.environ.get("DROPBOX_ACCESS_TOKEN") is not None:
                raise AssertionError('DROPBOX_ACCESS_TOKEN environment var is set. '
                                     'You also have to specify a test set on dropbox')
            if ".feather" in train_data_path:
                x = pd.read_feather(train_data_path).iloc[:20000]
            elif ".csv" in train_data_path:
                x = pd.read_csv(train_data_path).iloc[:20000]
            if normalize:
                self.scaler = StandardScaler()
                x.iloc[:, :] = self.scaler.fit_transform(x)
            y = x[target]
            x = x.drop(columns=[target], inplace=True)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y)

        self.columns = self.X_train.columns
        y_std = self.y_train.std()

        self.metric = base_metric

        # Load grammar from file
        self.start_symbol = start_symbol
        self.grammar_file_path = grammar_file_path
        if os.environ.get("DROPBOX_ACCESS_TOKEN") is not None:
            grammar_file_path = grammar_file_path.replace('../', "")
        self.grammar = ProbabilisticGrammar(grammar_file_path, start_symbol=start_symbol,
                                            dataset_n_vars=len(self.columns))

        if apply_constraints:
            self.constraints = Constraints(self.grammar, self.max_horizon, self.min_horizon)

        self.constant_optimizer = constant_optimizer

        self.masks = self.grammar.symbols_to_mask
        self.symbols = list(self.grammar.productions_dict.keys())

        self.n_actions = len(self.grammar.productions_list)
        self.n_symbols = len(self.grammar.symbols) + 1
        self.hidden_size = hidden_size

        # Define gym observation space
        space_dict = {}
        if self.observe_symbol:
            space_dict['current_symbol'] = spaces.MultiBinary(self.n_symbols)
        if self.observe_mask:
            space_dict['current_mask'] = spaces.MultiBinary(self.n_actions)
        if self.observe_depth:
            space_dict['current_depth'] = spaces.Box(low=0, high=1, shape=(1, 1))
        if self.observe_hidden_state:
            space_dict["h"] = spaces.MultiBinary(self.hidden_size)
            space_dict["c"] = spaces.MultiBinary(self.hidden_size)
        if self.observe_previous_actions:
            space_dict['past_actions'] = spaces.MultiBinary((self.max_horizon, self.n_actions + 1))
        if self.observe_brotherhood:
            space_dict['brother_action'] = spaces.MultiBinary((self.grammar.max_brother_symbols, self.n_actions + 1))
        if self.observe_parent:
            space_dict['parent_action'] = spaces.MultiBinary(self.n_actions + 1)

        self.observation_space = spaces.Dict(space_dict)

        # Define gym action space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_actions,), dtype=np.float32)

        self.queue = None
        self.translations = None
        self.past_actions = None
        self.past_actions_with_parent_infos = None
        self.current_parent_infos = None
        self.done = None
        self.i_step = None

        # set a caching function
        self.cache = {}
        for k, v in self.eval_params.items():
            vars()[k] = v

    def reset(self):

        self.done = np.zeros((self.batch_size, 1))

        if self.queue is not None:
            del self.queue, self.current_parent_infos, self.past_actions_with_parent_infos, self.translations
            gc.collect()
        self.queue = [[]] * self.batch_size
        self.current_parent_infos = [('#', -1)] * self.batch_size
        self.past_actions_with_parent_infos = [[]] * self.batch_size
        self.translations = [self.start_symbol] * self.batch_size
        self.i_step = 0

        self.past_actions = np.full((self.batch_size, self.max_horizon, self.grammar.n_discrete_actions + 1),
                                    self.grammar.action_encoding["#"])

        observation = {}
        if self.observe_symbol:
            observation['current_symbol'] = np.full((self.batch_size, 1, self.grammar.n_symbols + 1),
                                                    self.grammar.symbol_encoding[self.start_symbol])
        if self.observe_mask:
            observation['current_mask'] = np.full((self.batch_size, 1, self.grammar.n_discrete_actions),
                                                  self.grammar.symbols_to_mask[self.grammar.start_symbol])
        if self.observe_depth:
            observation["current_depth"] = np.zeros((self.batch_size, 1, 1))
        if self.observe_previous_actions:
            observation['past_actions'] = self.past_actions
        if self.observe_brotherhood:
            observation['brother_action'] = np.full((self.batch_size, self.grammar.max_brother_symbols,
                                                     self.grammar.n_discrete_actions + 1),
                                                    self.grammar.action_encoding["#"])
        if self.observe_parent:
            observation['parent_action'] = np.full((self.batch_size, 1, self.grammar.n_discrete_actions + 1),
                                                   self.grammar.action_encoding["#"])

        return observation

    def step(self, action_ids):
        new_symbols = np.zeros((self.batch_size, 1, self.n_symbols))
        new_masks = np.zeros((self.batch_size, 1, self.grammar.n_discrete_actions))
        new_parents = np.zeros((self.batch_size, 1, self.n_actions + 1))
        new_brothers = np.zeros((self.batch_size, self.grammar.max_brother_symbols, self.n_actions + 1))

        for i, action_id in enumerate(action_ids):
            if self.done[i]:
                new_symbols[i] = self.grammar.symbol_encoding["#"]
                new_masks[i] = self.grammar.symbols_to_mask[self.start_symbol]
                new_parents[i] = self.grammar.action_encoding["#"]
                new_brothers[i] = np.array([[self.grammar.action_encoding['#'][0]] * self.grammar.max_brother_symbols])
                continue

            action = self.grammar.productions_list[action_id]
            self.translations[i] = re.sub("<.+?>", action['raw'], self.translations[i], 1)

            # Store action in list
            self.past_actions_with_parent_infos[i] = self.past_actions_with_parent_infos[i] + [
                (action['raw'], self.current_parent_infos[i])]

            # Append descendant symbols to the queue.
            parent_action_info = (action['raw'], self.i_step)
            self.queue[i] = [(s, parent_action_info) for s in action["descendant_symbols"]] + self.queue[i]
            if not self.queue[i]:
                self.done[i] = 1
                if self.i_step <= self.min_horizon:
                    self.translations[i] = ""

                new_symbols[i] = self.grammar.symbol_encoding["#"]
                new_masks[i] = self.grammar.symbols_to_mask[self.start_symbol]
                new_parents[i] = self.grammar.action_encoding["#"]
                new_brothers[i] = np.array([[self.grammar.action_encoding['#'][0]] * self.grammar.max_brother_symbols])
                continue

            new_current_symbol, new_parent_infos = self.queue[i].pop(0)
            new_symbols[i] = self.grammar.symbol_encoding[new_current_symbol][0]
            new_parents[i] = self.grammar.action_encoding[new_parent_infos[0]]

            self.current_parent_infos[i] = new_parent_infos

            # get brothers
            if self.observe_brotherhood:
                brothers = [self.grammar.action_encoding[potential_brother_action][0]
                            for potential_brother_action, parent_info in self.past_actions_with_parent_infos[i]
                            if parent_info == new_parent_infos]

                brothers = brothers + [self.grammar.action_encoding['#'][0] for _ in
                                       range(self.grammar.max_brother_symbols - len(brothers))]
                new_brothers[i] = np.array(brothers)

            m = self.grammar.symbols_to_mask[new_current_symbol]
            if self.apply_constraints:
                c = self.constraints.init_constraint()
                queue_min_size = sum([min([p['distance_to_terminal']
                                           for p in self.grammar.productions_dict[q]]) for q, _ in self.queue[i]])

                c = self.constraints.make_min_max_constraint(c, new_current_symbol, self.i_step + queue_min_size)
                c = self.constraints.make_trig_constraint(c, new_current_symbol, self.translations[i])

                m = np.multiply(c, m)

                if m.sum() == 0:
                    m = self.grammar.symbols_to_mask[new_current_symbol]

            new_masks[i] = m

            # Update current symbol previously_selected_actions with the action input
            self.past_actions[i][self.i_step] = self.grammar.action_encoding[str(action['raw'])][0]

        self.i_step += 1
        observation = {}
        if self.observe_symbol:
            observation['current_symbol'] = new_symbols
        if self.observe_mask:
            observation['current_mask'] = new_masks
        if self.observe_depth:
            observation['current_depth'] = np.ones((self.batch_size, 1, 1)) / self.i_step
        if self.observe_previous_actions:
            observation['past_actions'] = self.past_actions
        if self.observe_brotherhood:
            assert len(new_brothers) == self.batch_size
            if new_brothers.shape == (self.batch_size, self.grammar.max_brother_symbols, 1,
                                      self.grammar.n_discrete_actions + 1):
                new_brothers = np.squeeze(new_brothers, axis=2)
            observation['brother_action'] = new_brothers
        if self.observe_parent:
            assert len(new_parents) == self.batch_size
            observation['parent_action'] = new_parents

        return observation, self.done

    def compute_final_reward(self):
        rewards = np.empty((self.batch_size,))
        for i, translation in enumerate(self.translations):
            rewards[i] = self.evaluate_on_data(i, translation)
        return rewards

    def evaluate_on_data(self, i_t, t):
        reward = 0
        if ("<" in t) or (t == ""):
            return reward
        else:
            def lasso_fit(columns_list):
                model = Lasso(alpha=0.01, max_iter=1000)
                model.fit(self.X_train[columns_list], self.y_train.values.reshape(-1, 1))
                return model.predict(self.X_train[columns_list])

            try:
                if self.constant_optimizer & ("const" in t):
                    constants = self.optimize_constants(t)
                    for i_constant, c in enumerate(constants):
                        t = t.replace('const', "{:.2f}".format(c), 1)
                        self.translations[i_t] = t
            except Exception as e:
                str(e)
            try:
                x = self.X_train
                if isinstance(x, pd.DataFrame) & self.use_np:
                    x = x.values
                y_pred = eval(t)
                y_pred[np.isnan(y_pred)] = 0
                if isinstance(y_pred, np.float64) or isinstance(y_pred, int):
                    return reward
                elif np.mean(np.abs(y_pred)) < 1e-10:
                    return reward
                else:
                    if isinstance(y_pred, pd.Series):
                        y_pred = y_pred.values
                    reward = self.metric(self.y_train.values, y_pred)

                if np.isnan(reward):
                    reward = 0
            except Exception as e:
                reward = 0
        return reward

    def optimize_constants(self, expression):
        global nb_const
        nb_const = 0

        def my_replacer(match):
            global nb_const
            res = f"{match.group()}{nb_const}"
            nb_const += 1
            return res

        expression = re.sub('(const)', my_replacer, expression)
        string_args = ", ".join(['const' + str(i) for i in range(expression.count("const"))])

        def f(const, expression, str_args, x_train, y_train):
            tmp_f = eval(f"lambda {str_args}, x, y: np.sum(({expression} - y)**2)")
            res = tmp_f(x=x_train, y=y_train, *const)
            return res

        initial_guess = np.random.rand(expression.count('const'))
        constants = optimize.minimize(f, initial_guess, method='BFGS', args=(expression,
                                                                             string_args,
                                                                             self.X_train,
                                                                             self.y_train))

        return constants['x']

    def get_predicted_values(self):
        predicted_data = np.empty((self.batch_size, len(self.y_test)))
        for i, translation in enumerate(self.translations):
            predicted_data[i] = self.get_y_pred_test(i, translation)
        return predicted_data

    def get_y_pred_test(self, i_t, t):
        if ("<" in t) or (t == ""):
            return np.zeros((len(self.y_test)))
        else:
            def lasso_fit(columns_list):
                model = Lasso(alpha=0.01, max_iter=1000)
                model.fit(self.X_test[columns_list], self.y_test.values.reshape(-1, 1))
                return model.predict(self.X_test[columns_list])

            try:
                if self.constant_optimizer & ("const" in t):
                    constants = self.optimize_constants(t)
                    for i_constant, c in enumerate(constants):
                        t = t.replace('const', "{:.2f}".format(c), 1)
                        self.translations[i_t] = t
            except Exception as e:
                str(e)
                # print(e)
            try:
                x = self.X_test
                if isinstance(x, pd.DataFrame) & self.use_np:
                    x = x.values
                y_pred = eval(t)
                y_pred[np.isnan(y_pred)] = 0
            except Exception as e:
                return np.zeros((len(self.y_test)))
        return y_pred

    def simulate_reset(self, nb_suggestions):
        suggestions_infos = {
            "done": np.zeros((nb_suggestions, 1)),
            "queue": [[]] * nb_suggestions,
            "current_parent_infos": [('#', -1)] * nb_suggestions,
            "past_actions_with_parent_infos": [[]] * nb_suggestions,
            "translations": [self.start_symbol] * nb_suggestions,
            "i_step": 0,
            'past_actions': np.full((nb_suggestions,
                                     self.max_horizon,
                                     self.grammar.n_discrete_actions + 1),
                                    self.grammar.action_encoding["#"])}

        observation = {}
        if self.observe_symbol:
            observation['current_symbol'] = np.full((nb_suggestions, 1, self.grammar.n_symbols + 1),
                                                    self.grammar.symbol_encoding[self.start_symbol])
        if self.observe_mask:
            observation['current_mask'] = np.full((nb_suggestions, 1, self.grammar.n_discrete_actions),
                                                  self.grammar.symbols_to_mask[self.grammar.start_symbol])
        if self.observe_depth:
            observation["current_depth"] = np.zeros((nb_suggestions, 1, 1))
        if self.observe_previous_actions:
            observation['past_actions'] = suggestions_infos['past_actions']
        if self.observe_brotherhood:
            observation['brother_action'] = np.full((nb_suggestions, self.grammar.max_brother_symbols,
                                                     self.grammar.n_discrete_actions + 1),
                                                    self.grammar.action_encoding["#"])
        if self.observe_parent:
            observation['parent_action'] = np.full((nb_suggestions, 1, self.grammar.n_discrete_actions + 1),
                                                   self.grammar.action_encoding["#"])

        return observation, suggestions_infos

    def simulate_step(self, action_ids, nb_suggestions, suggestions_infos):

        new_symbols = np.zeros((nb_suggestions, 1, self.n_symbols))
        new_masks = np.zeros((nb_suggestions, 1, self.grammar.n_discrete_actions))
        new_parents = np.zeros((nb_suggestions, 1, self.n_actions + 1))
        new_brothers = np.zeros((nb_suggestions, self.grammar.max_brother_symbols, self.n_actions + 1))

        for i, action_id in enumerate(action_ids):
            if suggestions_infos['done'][i]:
                new_symbols[i] = self.grammar.symbol_encoding["#"]
                new_masks[i] = self.grammar.symbols_to_mask[self.start_symbol]
                new_parents[i] = self.grammar.action_encoding["#"]
                new_brothers[i] = np.array([[self.grammar.action_encoding['#'][0]] * self.grammar.max_brother_symbols])
                continue
            action = self.grammar.productions_list[int(action_id)]
            suggestions_infos['translations'][i] = re.sub("<.+?>", action['raw'], suggestions_infos['translations'][i],
                                                          1)

            # Store action in list
            suggestions_infos['past_actions_with_parent_infos'][i] = \
            suggestions_infos['past_actions_with_parent_infos'][i] + \
            [(action['raw'], suggestions_infos['current_parent_infos'][i])]

            # Append descendant symbols to the queue.
            parent_action_info = (action['raw'], suggestions_infos['i_step'])
            suggestions_infos['queue'][i] = [(s, parent_action_info) for s in action["descendant_symbols"]] + \
                                            suggestions_infos['queue'][i]
            if not suggestions_infos['queue'][i]:
                suggestions_infos['done'][i] = 1
                if suggestions_infos['i_step'] <= self.min_horizon:
                    suggestions_infos['translations'][i] = ""

                new_symbols[i] = self.grammar.symbol_encoding["#"]
                new_masks[i] = self.grammar.symbols_to_mask[self.start_symbol]
                new_parents[i] = self.grammar.action_encoding["#"]
                new_brothers[i] = np.array([[self.grammar.action_encoding['#'][0]] * self.grammar.max_brother_symbols])
                continue

            new_current_symbol, new_parent_infos = suggestions_infos['queue'][i].pop(0)
            new_symbols[i] = self.grammar.symbol_encoding[new_current_symbol][0]
            new_parents[i] = self.grammar.action_encoding[new_parent_infos[0]]

            suggestions_infos['current_parent_infos'][i] = new_parent_infos

            # get brothers
            if self.observe_brotherhood:
                brothers = [self.grammar.action_encoding[potential_brother_action][0]
                            for potential_brother_action, parent_info in self.past_actions_with_parent_infos[i]
                            if parent_info == new_parent_infos]

                brothers = brothers + [self.grammar.action_encoding['#'][0] for _ in
                                       range(self.grammar.max_brother_symbols - len(brothers))]
                new_brothers[i] = np.array(brothers)

            m = self.grammar.symbols_to_mask[new_current_symbol]
            if self.apply_constraints:
                c = self.constraints.init_constraint()
                queue_min_size = sum([min([p['distance_to_terminal']
                                           for p in self.grammar.productions_dict[q]]) for q, _ in
                                      suggestions_infos['queue'][i]])

                c = self.constraints.make_min_max_constraint(c, new_current_symbol,
                                                             suggestions_infos['i_step'] + queue_min_size)
                c = self.constraints.make_trig_constraint(c, new_current_symbol, suggestions_infos['translations'][i])

                m = np.multiply(c, m)

                if m.sum() == 0:
                    m = self.grammar.symbols_to_mask[new_current_symbol]

            new_masks[i] = m

            # Update current symbol previously_selected_actions with the action input
            suggestions_infos['past_actions'][i][suggestions_infos['i_step']] = \
                self.grammar.action_encoding[str(action['raw'])][0]

        suggestions_infos['i_step'] += 1
        observation = {}
        if self.observe_symbol:
            observation['current_symbol'] = new_symbols
        if self.observe_mask:
            observation['current_mask'] = new_masks
        if self.observe_depth:
            observation['current_depth'] = np.ones((nb_suggestions, 1, 1)) / suggestions_infos['i_step']
        if self.observe_previous_actions:
            observation['past_actions'] = suggestions_infos['past_actions']
        if self.observe_brotherhood:
            assert len(new_brothers) == nb_suggestions
            if new_brothers.shape == (
                    nb_suggestions, self.grammar.max_brother_symbols, 1, self.grammar.n_discrete_actions + 1):
                new_brothers = np.squeeze(new_brothers, axis=2)
            observation['brother_action'] = new_brothers
        if self.observe_parent:
            assert len(new_parents) == nb_suggestions
            observation['parent_action'] = new_parents

        return observation, suggestions_infos['done'], suggestions_infos

    def simulate_reward(self, n_suggestions, translations):
        rewards = np.empty((n_suggestions,))
        for i, translation in enumerate(translations):
            rewards[i] = self.evaluate_on_data(i, translation)
        return rewards
