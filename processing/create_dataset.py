import os
import pickle

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import d4rl
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_dones(dataset):
    dones_float = np.zeros_like(dataset['rewards'])
    for i in tqdm(range(len(dones_float) - 1)):
        if np.linalg.norm(dataset['observations'][i + 1] -
                            dataset['next_observations'][i]
                            ) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1
    return dones_float

def get_episode_returns(rew, dones):
    terminal_indices = np.argwhere(dones)
    tru_indices = (terminal_indices + 1).squeeze()
    ep_rew = np.split(rew, tru_indices)
    ep_tot_rew = np.array([np.sum(x) for x in ep_rew])
    return ep_tot_rew

def split_dataset(dataset, num_elements):
    dataset = {k: v[:num_elements] for k, v in dataset.items()}
    return dataset

games=["halfcheetah", "walker2d", "hopper"]
levels=["random", "medium", "expert"]
version=2

chose = 333000
dd = {}
game = games[2]
for game in games:
    for level in levels:
        env_name = f"{game}-{level}-v{version}"
        env= gym.make(env_name)
        dd[level] = d4rl.qlearning_dataset(env)

    new_dd = {level: split_dataset(dd[level], chose) for level in levels}
    keys = new_dd[levels[0]].keys()
    fixed_dataset = {k: np.concatenate([new_dd[level][k] for level in levels]) for k in keys}

    rew = fixed_dataset["rewards"]
    dones_float = get_dones(fixed_dataset)
    ep_tot_rew = get_episode_returns(rew, dones_float)

    split_rew = np.array([rew[s * chose: (s + 1) * chose].sum() for s in range(3)])

    split_ep_dones = {level: get_dones(new_dd[level]) for level in levels}
    split_ep_rew = [get_episode_returns(new_dd[level]["rewards"], split_ep_dones[level]).mean() for level in levels]

    print(f"Avg rew per split: {split_rew / chose}")
    print(f"Avg episode per split: {split_ep_rew}")

    plt.plot(rew)
    plt.xlim(0, len(rew))
    plt.tight_layout()
    plt.title(f"{game.title()}")
    plt.axvline(x = 333000, color='black', ls='--')
    plt.axvline(x = 2 * 333000, color='black', ls='--')
    plt.xlabel("Samples")
    plt.ylabel("Step Reward")
    plt.savefig(f"temp/mixed/{game}_rew.jpg", dpi=200, bbox_inches="tight")
    plt.close()

    # * UNCOMMENT TO SAVE TO DISK 
    # fname = f"/home/ssujit/.d4rl/datasets/{game}_mixed-v2.pkl"
    # with open(fname, 'wb') as handle:
    #     print(f"Saving dict to {fname}")
    #     pickle.dump(fixed_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(fname, 'rb') as handle:
    #     b = pickle.load(handle)
