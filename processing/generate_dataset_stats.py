import os
import pickle
import json

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import d4rl
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
games=["halfcheetah", "walker2d", "hopper"]
levels=["random", "medium", "medium-replay", "medium-expert", "expert"]
version=2

def get_dones(dataset):
    dones_float = np.zeros_like(dataset['rewards'])
    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] -
                            dataset['next_observations'][i]
                            ) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1
    return dones_float

def calculate_ep_return(rew, dones_float):
    terminal_indices = np.argwhere(dones_float)
    tru_indices = (terminal_indices + 1).squeeze()
    ep_rew = np.split(rew, tru_indices)
    ep_tot_rew = np.array([np.sum(x) for x in ep_rew])
    return ep_tot_rew

dd = {}
for game in games:
    for level in levels:
        env_name = f"{game}-{level}-v{version}"
        env= gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        rew = dataset["rewards"]
        dones_float = get_dones(dataset)
        ep_tot_rew = calculate_ep_return(rew, dones_float)
        mean_score = ep_tot_rew.mean()
        norm_score = env.get_normalized_score(mean_score)
        dd[f"{game}-{level}-v2"] = [len(dataset["rewards"]), mean_score, norm_score]

data = [[key] + v for key, v in dd.items()]
cols = ["dataset", "size", "average_reward", "average_norm_score"]
df = pd.DataFrame(data, columns=cols)
df.to_csv("temp/data.csv")

data_name = df.dataset.to_list()
data_score = (df.average_norm_score * 100).to_list()
baseline = {k: {"Dataset": v} for k, v in zip(data_name, data_score)}

save_dir = "scores"
os.makedirs(save_dir, exist_ok=True)
fname = os.path.join(save_dir, "dataset_baselines.json")
with open(fname, 'w') as fp:
    json.dump(baseline, fp)