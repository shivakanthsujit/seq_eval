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
flags.DEFINE_integer('max_steps', int(1e5), 'Number of training steps.')
flags.DEFINE_string('mdl_strat', "naive", 'How to increase dataset size')
flags.DEFINE_integer('num_pretraining_steps', int(1e6),
                     'Number of pretraining steps.')
flags.DEFINE_integer('replay_buffer_size', 2000000,
                     'Replay buffer size (=max_steps if unspecified).')
flags.DEFINE_integer('init_dataset_size', None,
                     'Offline data size (uses all data if unspecified).')
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
    logdir = os.path.join(FLAGS.logdir, "mdl", f"finetune-{FLAGS.env_name}", f"iql_{FLAGS.id}", f"seed{FLAGS.seed}")
    print(f"Logging to {logdir}")
    summary_writer = Logger(logdir, 0)
    os.makedirs(logdir, exist_ok=True)

    env, dataset, scaler_fn = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    obs, gsteps = get_obs_gsteps(FLAGS.mdl_strat)
    elements = dataset.size
    max_steps = ceildiv(elements, obs) * gsteps

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    opt_decay_schedule = None,
                    **kwargs)
    initial_index = 5000
    dataset.insert_index = initial_index
    ps_sampled_indx = np.zeros(dataset.size)
    sampled_indx = np.zeros(dataset.size)
    eval_returns = []

    for i in tqdm.tqdm(range(1, max_steps + 1),
                       smoothing=0.1,
                       miniters=int(2e3),
                       disable=not FLAGS.tqdm):
        step = i // gsteps * obs
        ps_indx, indx, batch = dataset.sample(FLAGS.batch_size, i % gsteps == 0, obs)

        ps_sampled_indx[ps_indx] += 1
        sampled_indx[indx] += 1

        # if 'antmaze' in FLAGS.env_name:
        #     batch = Batch(observations=batch.observations,
        #                   actions=batch.actions,
        #                   rewards=batch.rewards - 1,
        #                   masks=batch.masks,
        #                   next_observations=batch.next_observations)

        update_info = agent.update(batch)

        dataset.insert_index = change_mdl(obs, gsteps, dataset.insert_index, i)
        dataset.insert_index = min(dataset.insert_index, dataset.size)

        summary_writer.step = step
        if step % FLAGS.log_interval == 0:
            summary_writer.scalar('training/insert_idx', dataset.insert_index)
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'training/{k}', v)
                else:
                    summary_writer.writer.add_histogram(f'training/{k}', v, step)
            summary_writer.write(True)

        if step % FLAGS.eval_interval == 0 or i == max_steps:
            perf = {}
            start = time.time()
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)
            perf["eval_time"] = time.time() - start


            for k, v in eval_stats.items():
                summary_writer.scalar(f'evaluation/average_{k}s', v)

            [summary_writer.scalar(f'perf/{k}', v) for k, v in perf.items()]

            summary_writer.write()

            eval_returns.append((step, eval_stats['return']))
            np.savetxt(os.path.join(logdir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 FLAGS.replay_buffer_size or FLAGS.max_steps)
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

    current_step = step

    observation, done = env.reset(), False
    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
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

        summary_writer.step = i + current_step
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
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.scalar(f'training/{k}', v)
                # else:
                #     summary_writer.add_histogram(f'training/{k}', v, i)
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
