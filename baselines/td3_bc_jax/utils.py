import numpy as np


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind],
        )

    def convert_D4RL(self, dataset):
        n = len(dataset["observations"])
        self.state[:n] = dataset["observations"]
        self.action[:n] = dataset["actions"]
        self.next_state[:n] = dataset["next_observations"]
        self.reward[:n] = dataset["rewards"].reshape(-1, 1)
        self.not_done[:n] = 1.0 - dataset["terminals"].reshape(-1, 1)
        self.size = n
        self.ptr = self.size

    def normalize_states(self, eps=1e-3):
        mean = self.state[:self.size].mean(0, keepdims=True)
        std = self.state[:self.size].std(0, keepdims=True) + eps
        self.state[:self.size] = (self.state[:self.size] - mean) / std
        self.next_state[:self.size] = (self.next_state[:self.size] - mean) / std
        return mean, std

class MDLReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), mixed=False):
        super().__init__(state_dim, action_dim, max_size)
        self.mixed = mixed
        self.reset()

    def reset(self):
        if self.mixed:
            print(f"Shuffling indices according to mixed profile.")
            chose = 333000
            a = [i * chose + np.random.permutation(chose) for i in range(3)]
            self.indices = np.concatenate(a)
            # assert self.indices.shape[0] == self.size
        else:
            self.indices = np.random.permutation(self.size)
        self.insert_index = 0

    def sample(self, batch_size: int, sample_last: bool = False, obs=0):
        assert self.insert_index != 0, "Have to set number of datapoints that we can sample from."
        assert self.insert_index <= self.size, "Insert index cant be larger than the dataset."
        start = 0
        if self.mixed:
            start = max(0, self.insert_index - 200000)
        ind = np.random.choice(self.indices[start: self.insert_index], size=batch_size)
        
        # pseudo_indices = np.random.randint(low=start, high=self.insert_index, size=batch_size)
        # ind = self.indices[pseudo_indices]

        if sample_last:
            assert obs > 0
            ind[-obs: ] = self.indices[self.insert_index - obs: self.insert_index]

        # ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind],
        )

def change_mdl(obs, gsteps, curr_index, train_iter):
    if train_iter % gsteps == 0:
        return curr_index + obs
    else:
        return curr_index

def get_obs_gsteps(strategy):
    if strategy == "naive":
        obs, gsteps = 1, 1
    elif "freq" in strategy:
        strategy = strategy.replace("freq", "")
        obs, gsteps = strategy.split("-")
        obs, gsteps = int(obs), int(gsteps)
    return obs, gsteps

def ceildiv(a, b):
    return -(a // -b)