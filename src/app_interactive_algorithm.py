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


def launch_training(writer_logdir="./test", dataset_value="nguyen4", grammar_with_without_value="with",
                    frequency_value=5, interaction_type='from_start', reuse="yes"):
    reuse = reuse == "yes"
    # model definition
    try:
        params = json.load(open("params.json", 'rb'))
    except:
        params = json.load(open("src/params.json", 'rb'))

    params['dataset'] = dataset_value
    params['env_kwargs']["grammar_file_path"] = params[dataset_value]["grammar_file_path"]
    params['env_kwargs']["train_data_path"] = params[dataset_value]["train_data_path"]
    params['env_kwargs']["test_data_path"] = params[dataset_value]["test_data_path"]
    params['env_kwargs']["start_symbol"] = params[dataset_value]["start_symbol"]
    params['env_kwargs']["target"] = params[dataset_value]["target"]
    if "eval_params_path" in params[dataset_value].keys():
        params['env_kwargs']["eval_params_path"] = os.path.join(params['folder_path'],
                                                                params[dataset_value]["eval_params_path"])

    if grammar_with_without_value == "with":
        params['env_kwargs']["grammar_file_path"] = params['env_kwargs']["grammar_file_path"].replace('.bnf',
                                                                                                      "_with_const.bnf")

    params['env_kwargs']["grammar_file_path"] = os.path.join(params['folder_path'],
                                                             params['env_kwargs']["grammar_file_path"])
    params['env_kwargs']["train_data_path"] = os.path.join(params['folder_path'],
                                                           params['env_kwargs']["train_data_path"])
    params['env_kwargs']["test_data_path"] = os.path.join(params['folder_path'],
                                                          params['env_kwargs']["test_data_path"])
    params['env_kwargs']["dataset_name"] = params["dataset"]
    params['env_kwargs']["use_np"] = True
    if dataset_value == "case14":
        params['env_kwargs']["constant_optimizer"] = True
        params['risk_eps'] = params[dataset_value]['risk_eps']

    user_kwargs = {'reuse': reuse,
                   'interaction_frequency': frequency_value}

    model = PreferenceReinforceGUI(env_class=BatchSymbolicRegressionEnv,
                                   writer_logdir=writer_logdir,
                                   env_kwargs=params['env_kwargs'],
                                   policy_class=Policy,
                                   policy_kwargs=params['policy_kwargs'],
                                   dataset=params['dataset'],
                                   user=RealUser(gui_data_path=writer_logdir, **user_kwargs),
                                   x_label=params[dataset_value]['x_label'],
                                   debug=1, **params['algo_kwargs'])
    model.train(params['n_epochs'])


if __name__ == "__main__":
    fire.Fire(launch_training)
