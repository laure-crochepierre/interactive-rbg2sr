# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression

import os
import json
import fire
from envs import BatchSymbolicRegressionEnv
from interactive_algorithms import PreferenceReinforceGUI
from policies import Policy
from user_behavior import RealUser


def launch_training(writer_logdir):
    # model definition
    params = json.load(open("params.json", 'rb'))
    params['env_kwargs']["grammar_file_path"] = os.path.join(params['folder_path'],
                                                             params['env_kwargs']["grammar_file_path"])
    params['env_kwargs']["train_data_path"] = os.path.join(params['folder_path'],
                                                           params['env_kwargs']["train_data_path"])
    params['env_kwargs']["test_data_path"] = os.path.join(params['folder_path'],
                                                          params['env_kwargs']["test_data_path"])
    model = PreferenceReinforceGUI(env_class=BatchSymbolicRegressionEnv,
                                   writer_logdir=writer_logdir,
                                   env_kwargs=params['env_kwargs'],
                                   policy_class=Policy,
                                   policy_kwargs=params['policy_kwargs'],
                                   dataset=params['dataset'],
                                   user_behavior=RealUser(),
                                   debug=1, **params['algo_kwargs'])
    model.train(params['n_epochs'])


if __name__ == "__main__":
    fire.Fire(launch_training)
