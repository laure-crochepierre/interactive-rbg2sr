import fire
import datetime
import pandas as pd
import numpy as np
import grid2op
from grid2op.Rules import DefaultRules
from grid2op.Parameters import Parameters

from tqdm import tqdm
from lightsim2grid import LightSimBackend
from sklearn.model_selection import train_test_split

n_episodes = 100
max_step = 100
date_features = ["year", "month", "day", "hour_of_day", "minute_of_hour",]
features = [ "prod_p", "prod_q", "prod_v", "load_p", "load_q", "load_v", 'a_or', 'a_ex', 'p_or', 'p_ex', 'q_or', 'q_ex', 'v_or', 'v_ex', 'rho', "line_status", "line_or_to_sub_pos", "line_or_to_sub_pos", "load_to_sub_pos", "gen_to_sub_pos", 'topo_vect']
n_1_features = [ 'p_or', 'p_ex', 'a_or', 'a_ex']
lines_of_interest = ["4_5_17"]
deco = ['3_8_16'] # 12_13_14

def main(run_type="dc"):
    if not run_type in ['ac', 'dc']:
        print('run_type must be "ac" or "dc')
        return
    env_name = "l2rpn_case14_sandbox"
    backend=LightSimBackend()

    param = Parameters()
    param.ENV_DC =  {'ac': False, 'dc': True}[run_type]
    param.FORECAST_DC =  {'ac': False, 'dc': True}[run_type]
    param.MAX_LINE_STATUS_CHANGED = 10e10
    param.MAX_SUB_CHANGED = 10e10
    param.NO_OVERFLOW_DISCONNECTION = True
    param.NB_TIMESTEP_COOLDOWN_LINE = 0
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    param.HARD_OVERFLOW_THRESHOLD = 10e10
    env = None
    if run_type == 'dc':
        env = grid2op.make(env_name, param=param)
    else:
        env = grid2op.make(env_name,  backend=backend, param=param)

    columns = [c for c in date_features]
    for f in ["prod_p", "prod_q", "prod_v"]:
        columns += [f"{f}_{p}" for p in env.name_gen]
    for f in ["load_p", "load_q", "load_v"]:
        columns += [f"{f}_{l}" for l in env.name_load]
    for f in ['a_or', 'a_ex', 'p_or', 'p_ex', 'q_or', 'q_ex', 'v_or', 'v_ex', 'rho', "line_status",
         "line_or_to_sub_pos", "line_or_to_sub_pos"]:
        columns += [f"{f}_{l}" for l in env.name_line]
    columns += [f"load_to_sub_pos_{l}" for l in env.name_load]
    columns += [f"gen_to_sub_pos_{l}" for l in env.name_gen]

    columns += [f"topo_vect_{l}" for l in env.name_load]
    columns += [f"topo_vect_{l}" for l in env.name_gen]
    columns += [f"topo_vect_{l}_or" for l in env.name_line]
    columns += [f"topo_vect_{l}_ex" for l in env.name_line]
    data = np.empty((n_episodes * max_step * 2, len(columns)))
    i_data = 0
    new_columns = [f for f in date_features]

    for n in tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        i_step = 0

        while not done:
            instance = [getattr(obs, f) for f in date_features]
            base_obs = obs
            try:
                act = env.action_space()
                act.line_set_status = [(l, 1) for l in env.name_line]
                obs, _, _, _ = base_obs.simulate(act, time_step=0)
            except Exception as e:
                print(e)
            for f in features:

                instance += getattr(obs, f).tolist()
                if i_data == 0:
                    if ('prod' in f) or ('gen' in f):
                        new_columns += [f"{f}_{p}" for p in env.name_gen]
                    elif 'load' in f :
                        new_columns += [f"{f}_{l}" for l in env.name_load]
                    elif 'topo_vect' in f:
                        new_columns += [f"topo_vect_{l}" for l in env.name_load]
                        new_columns += [f"topo_vect_{l}" for l in env.name_gen]
                        new_columns += [f"topo_vect_{l}_or" for l in env.name_line]
                        new_columns += [f"topo_vect_{l}_ex" for l in env.name_line]
                    else :
                        new_columns += [f"{f}_{l}" for l in env.name_line]
            assert(columns==new_columns)
            data[i_data] = np.array(instance)
            i_data += 1
            # Compute N-1
            for l_deco in deco:

                act = env.action_space()
                act.line_set_status = [{True:(l, -1), False:(l,1)}[l==l_deco] for l in env.name_line]
                n_1_obs, _, _, _ = base_obs.simulate(act, time_step=0)

                instance = [getattr(n_1_obs, f) for f in date_features]
                for f in features:
                    instance += getattr(n_1_obs, f).tolist()
                data[i_data] = np.array(instance)
                i_data += 1

            obs, reward, done, info = env.step(env.action_space())

            i_step += 1
            if i_step >= max_step:
                done = True

    df = pd.DataFrame(data=data, columns=new_columns)
    df.to_csv(f'simulate_deco_{"-".join(deco)}_data_n_{run_type}.csv', index=False)

    df.loc[:, 'datetime'] = df.apply(lambda row: datetime.datetime(year=int(row.year),
                                                                   month=int(row.month),
                                                                   day=int(row.day),
                                                                   hour=int(row.hour_of_day),
                                                                   minute=int(row.minute_of_hour)), axis=1)

    df_train, df_test = train_test_split(df)
    df_train.to_csv(f'simulate_deco_{"-".join(deco)}_data_train_n_{run_type}.csv', index=False)
    df_test.to_csv(f'simulate_deco_{"-".join(deco)}_data_test_n_{run_type}.csv', index=False)


if __name__ == "__main__":
    fire.Fire(main)

