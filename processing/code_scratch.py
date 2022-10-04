import wrappers
import d4rl
from dataset_utils import D4RLDataset
import gym
from common import normalize

env_name = "halfcheetah-medium-v2"
seed = 0
env = gym.make(env_name)
env = gym.make(env_name)
env = wrappers.EpisodeMonitor(env)
env = wrappers.SinglePrecision(env)
dataset = D4RLDataset(env)
print(dataset.rewards.min(), dataset.rewards.max(), dataset.rewards.mean(), dataset.rewards.std())
normalize(dataset)