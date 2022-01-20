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

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from multiprocessing import Pool

from envs import BatchSymbolicRegressionEnv
from interactive_algorithms import PreferenceReinforceGUI
from policies import Policy
from user_behavior import SelectRewardWithProbUser, SelectRandomRewardUser, SelectBestRewardUser

columns = ["name", "reuse", "interaction_frequency",
                          "mse", "mae", "r2", "nrmse", "result", 'time']
def launch_training(params):
    i, reuse, interaction_frequency, user, writer_logdir = params

    print(i, writer_logdir)

    np.random.seed(i)
    torch.random.manual_seed(i)

    # model definition
    params = json.load(open("params.json", 'rb'))
    params['env_kwargs']["grammar_file_path"] = os.path.join(params['folder_path'],
                                                             params['env_kwargs']["grammar_file_path"])
    params['env_kwargs']["train_data_path"] = os.path.join(params['folder_path'],
                                                           params['env_kwargs']["train_data_path"])
    params['env_kwargs']["test_data_path"] = os.path.join(params['folder_path'],
                                                          params['env_kwargs']["test_data_path"])
    params["n_epochs"] = 1000
    params['algo_kwargs']['risk_eps'] = 0.05
    model = PreferenceReinforceGUI(env_class=BatchSymbolicRegressionEnv,
                                   writer_logdir=f"{writer_logdir}/{time.time()}",
                                   env_kwargs=params['env_kwargs'],
                                   policy_class=Policy,
                                   policy_kwargs=params['policy_kwargs'],
                                   dataset=params['dataset'],
                                   user=user,
                                   debug=1, **params['algo_kwargs'])

    debut = time.time()
    try:
        model.train(params['n_epochs'])
    except Exception as e:
        print(e)
    duree = time.time() - debut

    var_y = model.env.y_test.var()
    nrmse = lambda y, yhat: mean_squared_error(y, yhat) / var_y
    metrics = [mean_squared_error, mean_absolute_error, r2_score, nrmse]

    f = eval(f'lambda x : {model.logger["best_expression"]}')
    y_pred = f(model.env.X_test)

    scores = [user.type, reuse, interaction_frequency]
    for m in metrics:
        try:
            scores += [m(model.env.y_test, y_pred)]
        except Exception as e:
            scores += [e]
    scores += [model.logger["best_expression"], duree]
    pd.DataFrame(data=[scores],
                 columns=columns).to_csv(f"{writer_logdir}/{time.time()}.csv")
    return scores


if __name__ == "__main__":
    nb_tests = 10
    scores = []

    combinaitions = []
    for i in range(nb_tests):
        for reuse in [True, False]:
            for interaction_frequency in [2, 5, 10, 15, 20]:
                user_params = {'reuse': reuse,
                               'interaction_frequency': interaction_frequency}
                for user in [SelectRewardWithProbUser(0.8, **user_params), SelectRewardWithProbUser(0.6, **user_params),
                             SelectRewardWithProbUser(0.4, **user_params), SelectRewardWithProbUser(0.2, **user_params),
                             SelectRandomRewardUser(**user_params), SelectBestRewardUser(**user_params)]:
                    logdir = f"../results/user_benchmark/reuse_{reuse}/{user.type}/freq_{interaction_frequency}/{i}"
                    combinaitions.append([i, reuse, interaction_frequency, user, logdir])

    with Pool(10) as p:
            scores = p.map(launch_training, combinaitions)

    pd.DataFrame(data=scores,
                 columns=columns).to_csv(f"../results/user_benchmark_{time.time()}.csv")
