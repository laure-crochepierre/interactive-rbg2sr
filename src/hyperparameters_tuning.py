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
from torch import set_num_threads
set_num_threads(1)
from torch.random import manual_seed as torch_manual_seed

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from multiprocessing import Pool

from envs import BatchSymbolicRegressionEnv
from interactive_algorithms import PreferenceReinforceGUI
from policies import Policy
from user_behavior import SelectRewardWithProbUser, SelectBestRewardUser, SelectByRegexUser

columns = ["name", "reuse", "interaction_frequency", "mse", "mae", "r2", "nrmse", "result", 'time']


def launch_training(params):
    i, reuse, interaction_frequency, user, writer_logdir = params
    user_params = {'reuse': reuse,
                   'interaction_frequency': interaction_frequency}
    user = SelectByRegexUser(**user_params)

    print(i, writer_logdir, flush=True)

    np.random.seed(i)
    torch_manual_seed(i)

    # model definition
    params = json.load(open("params.json", 'rb'))
    params['env_kwargs']["grammar_file_path"] = os.path.join(params['folder_path'],
                                                             params['env_kwargs']["grammar_file_path"])
    params['env_kwargs']["train_data_path"] = os.path.join(params['folder_path'],
                                                           params['env_kwargs']["train_data_path"])
    params['env_kwargs']["test_data_path"] = os.path.join(params['folder_path'],
                                                          params['env_kwargs']["test_data_path"])
    params['env_kwargs']["use_np"] = True

    params["n_epochs"] = 1000

    params['algo_kwargs']['verbose'] = 0

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
    if model.env.use_np:
        y_pred = f(model.env.X_test.values)
    else:
        y_pred = f(model.env.X_test)

    scores = [user.type, reuse, interaction_frequency]
    for m in metrics:
        try:
            scores += [m(model.env.y_test, y_pred)]
        except Exception as e:
            scores += [e]
    scores += [model.logger["best_expression"], duree]
    print(scores, flush=True)
    pd.DataFrame(data=[scores],
                 columns=columns).to_csv(f"{writer_logdir}/run_final_res_{time.time()}.csv")
    return scores


if __name__ == "__main__":
    nb_tests = 10
    scores = []

    combinaitions = []
    for i in range(nb_tests):
        for entropy_coeff in [0.001, 0.005, 0.0001, 0.0005]:
            for learning_rate in [0.1, 0.01, 0.001, 0.0001]:

                for risk_eps in [0.1, 0.01, 0.5]:
                    logdir = f"../results/hyperparameter_tuning_minmax_loss/entropy_{entropy_coeff}_lr_{learning_rate}_risk_eps_{risk_eps}_{i}"
                    combinaitions.append([i, entropy_coeff, learning_rate, risk_eps, logdir])
    print("Nb combinaisons", len(combinaitions), flush=True)

    with Pool(6) as p:
            scores = p.map(launch_training, combinaitions)

    pd.DataFrame(data=scores,
                 columns=columns).to_csv(f"../results/parameter_tuning_minmax_loss{time.time()}.csv")
