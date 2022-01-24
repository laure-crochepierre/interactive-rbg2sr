# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the interactive-RBG2SR an interactive approach to reinforcement based grammar guided symbolic regression

import os
import json
import time
import torch
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(1)

import fire
from envs import BatchSymbolicRegressionEnv
from interactive_algorithms import PreferenceReinforceGUI
from policies import Policy
from user_behavior import SelectBestRewardUser


def launch_training(interaction_frequency=2, reuse=False, writer_logdir=None):

    if writer_logdir is None:
        writer_logdir = f"../results/test/best/interaction_freq_{interaction_frequency}/{time.time()}"

    if not isinstance(reuse, bool):
        reuse = bool(reuse) == True

    print('reuse', reuse)

    # model definition
    params = json.load(open("params.json", 'rb'))
    params['env_kwargs']["grammar_file_path"] = os.path.join(params['folder_path'],
                                                             params['env_kwargs']["grammar_file_path"])
    params['env_kwargs']["train_data_path"] = os.path.join(params['folder_path'],
                                                           params['env_kwargs']["train_data_path"])
    params['env_kwargs']["test_data_path"] = os.path.join(params['folder_path'],
                                                          params['env_kwargs']["test_data_path"])
    params['env_kwargs']["use_np"] = True
    params['algo_kwargs']["learning_rate"] = 0.001
    params["n_epochs"] = 1000
    params['algo_kwargs']['risk_eps'] = 0.05

    user_params = {'reuse': reuse,
                   'interaction_frequency': interaction_frequency}
    model = PreferenceReinforceGUI(env_class=BatchSymbolicRegressionEnv,
                                   writer_logdir=writer_logdir,
                                   env_kwargs=params['env_kwargs'],
                                   policy_class=Policy,
                                   policy_kwargs=params['policy_kwargs'],
                                   dataset=params['dataset'],
                                   user=SelectBestRewardUser(**user_params),
                                   debug=1, **params['algo_kwargs'])
    model.train(params['n_epochs'])


if __name__ == "__main__":
    fire.Fire(launch_training)
