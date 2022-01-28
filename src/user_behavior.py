# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression

import os
import copy
import random
import time
import pickle
import dropbox

import numpy as np

from abc import ABC


class User(ABC):
    def __init__(self, reuse=False, interaction_frequency=1):
        self.default_answer = {"pairs": {"ids": [], "answers": []},
                               "classes": {"top": [], "middle": [], 'low': []},
                               "suggest": []}
        self.description = "Default User"
        self.preferences = None
        self.reuse = reuse
        self.interaction_frequency = interaction_frequency

    @staticmethod
    def select_preference(self, gui_infos, i_epoch):
        return NotImplemented


class RealUser(User):
    def __init__(self, gui_data_path, **kwargs):
        super(RealUser, self).__init__(**kwargs)
        self.gui_data_path = os.path.join(gui_data_path, "gui_data")
        self.description = "Real user. Preferences are queried with an interactive interface"
        self.type = "real"
        self.rules = None

        self.dbx = None
        if os.environ.get("DROPBOX_ACCESS_TOKEN") is not None:
            self.dbx = dropbox.Dropbox(os.environ.get('DROPBOX_ACCESS_TOKEN'))

    def select_preference(self, gui_infos, i_epoch):
        if (not self.reuse) or (i_epoch % self.interaction_frequency == 0):

            questions_path = os.path.join(self.gui_data_path, f"{i_epoch}.pkl")
            if self.dbx is None:
                pickle.dump(gui_infos, open(questions_path, 'wb'))
            else:
                self.dbx.files_upload(pickle.dumps(gui_infos), path=questions_path)  # save data to dropbox

            answer_path = os.path.join(self.gui_data_path, f"{i_epoch}_answers.pkl")
            answers_path_exist = False
            while not answers_path_exist:
                time.sleep(2)
                if self.dbx is None:
                    answers_path_exist = os.path.exists(answer_path)
                else:
                    try:
                        self.dbx.files_get_metadata(answer_path)
                        answers_path_exist = True
                    except Exception as e:
                        answers_path_exist = False

            gui_answers = None
            if self.dbx is None:
                gui_answers = pickle.load(open(answer_path, 'rb'))
            else:
                _, file_content = self.dbx.files_download(answer_path)
                gui_answers = pickle.loads(file_content.content)

            self.preferences = gui_answers
            if 'rules' in list(gui_answers.keys()):
                self.rules = gui_answers['rules']
        return self.preferences


class SelectBestRewardUser(User):
    def __init__(self, **kwargs):
        super(SelectBestRewardUser, self).__init__(**kwargs)
        self.description = "User who always select the solution with the highest reward"
        self.type = "best"

    def select_preference(self, gui_infos, i_epoch):
        if (not self.reuse) or (i_epoch % self.interaction_frequency == 0):
            simulated_answers = copy.deepcopy(self.default_answer)
            simulated_answers['pairs']['ids'] = gui_infos['combinaisons']

            rewards = gui_infos['rewards']
            answers = np.array(
                [{True: 'r', False: 'l'}[rewards[i1] < rewards[i2]] for i1, i2 in gui_infos['combinaisons']])
            simulated_answers['pairs']['answers'] = answers
            self.preferences = simulated_answers
        return self.preferences


class SelectRandomRewardUser(User):
    def __init__(self, **kwargs):
        super(SelectRandomRewardUser, self).__init__(**kwargs)
        self.description = "User who always select a random solution"
        self.type = "random"

    def select_preference(self, gui_infos, i_epoch):
        if (not self.reuse) or (i_epoch % self.interaction_frequency == 0):
            simulated_answers = copy.deepcopy(self.default_answer)
            simulated_answers['pairs']['ids'] = gui_infos['combinaisons']

            rewards = gui_infos['rewards']
            answers = np.array(
                [{True: 'r', False: 'l'}[random.uniform(0, 1) > 0.5] for i1, i2 in gui_infos['combinaisons']])
            simulated_answers['pairs']['answers'] = answers
            self.preferences = simulated_answers
        return self.preferences


class SelectRewardWithProbUser(User):
    def __init__(self, prob=0.6, **kwargs):
        super(SelectRewardWithProbUser, self).__init__(**kwargs)
        self.description = "User who always select a random solution"
        self.prob = prob
        self.type = f"random_prob_{prob}"

    def select_preference(self, gui_infos, i_epoch):
        if (not self.reuse) or (i_epoch % self.interaction_frequency == 0):
            simulated_answers = copy.deepcopy(self.default_answer)
            simulated_answers['pairs']['ids'] = gui_infos['combinaisons']

            rewards = gui_infos['rewards']
            answers = np.array([{True: 'r', False: 'l'}[np.logical_and(random.uniform(0, 1) > self.prob,
                                                                       rewards[i1] < rewards[i2])]
                                for i1, i2 in gui_infos['combinaisons']])
            simulated_answers['pairs']['answers'] = answers
            self.preferences = simulated_answers
        return self.preferences
