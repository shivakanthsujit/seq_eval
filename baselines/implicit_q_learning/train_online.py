import os
from typing import Tuple

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import time
from common import ceildiv

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import wrappers
from dataset_utils import (Batch, MDLD4RLDataset, ReplayBuffer,
                           split_into_trajectories)
from evaluation import Logger, evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('id', 'baseline', 'Experiment name')
flags.DEFINE_string('logdir', 'logs', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2e6), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', 2000000,
                     'Replay buffer size (=max_steps if unspecified).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    # 'configs/antmaze_finetune_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    scaler = compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards /= scaler
    dataset.rewards *= 1000.0

    scaler_fn = lambda x: x * 1000 / scaler
    return scaler_fn


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, MDLD4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = MDLD4RLDataset(env)
    print("Before")
    print(dataset.rewards.min(), dataset.rewards.max(), dataset.rewards.mean(), dataset.rewards.std())

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        scaler_fn = lambda x: x - 1
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        scaler_fn = normalize(dataset)

    print("After")
    print(dataset.rewards.min(), dataset.rewards.max(), dataset.rewards.mean(), dataset.rewards.std())
    return env, dataset, scaler_fn


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

def main(_):
    env_name = FLAGS.env_name.split("-")[0]
    logdir = os.path.join(FLAGS.logdir, "mdl", f"online-{env_name}", f"iql_{FLAGS.id}", f"seed{FLAGS.seed}")
    print(f"Logging to {logdir}")
    summary_writer = Logger(logdir, 0)
    os.makedirs(logdir, exist_ok=True)

    env, dataset, scaler_fn = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    opt_decay_schedule = None,
                    **kwargs)
    eval_returns = []

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 FLAGS.replay_buffer_size or FLAGS.max_steps)

    train_start = int(1e4)
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < train_start:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation, )
        action = np.clip(action, -1, 1)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, scaler_fn(reward), mask,
                                float(done), next_observation)
        observation = next_observation

        summary_writer.step = i
        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                summary_writer.scalar(f'training/{k}', v)
            summary_writer.write()

        batch = replay_buffer.sample(FLAGS.batch_size)
        # if 'antmaze' in FLAGS.env_name:
        #     batch = Batch(observations=batch.observations,
        #                   actions=batch.actions,
        #                   rewards=batch.rewards - 1,
        #                   masks=batch.masks,
        #                   next_observations=batch.next_observations)
        if replay_buffer.size > train_start:
            update_info = agent.update(batch)
        else:
            update_info = None
        
        if i % FLAGS.log_interval == 0:
            summary_writer.scalar(f'replay_buffer_size', replay_buffer.size)
            if update_info is not None:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        summary_writer.scalar(f'training/{k}', v)
            summary_writer.write(True)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.scalar(f'evaluation/average_{k}s', v)
            summary_writer.write(True)

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(logdir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
