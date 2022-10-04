import os
from pprint import pprint
import random
import time

from jaxrl.logger import Logger

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"]="1"

import numpy as np
from tqdm import tqdm
from absl import app, flags
from ml_collections import config_flags

from jaxrl.agents import SACLearner, AWACLearner, BCLearner
from jaxrl.utils import make_env, get_obs_gsteps, ceildiv, change_mdl
from jaxrl.datasets.d4rl_dataset import MDLD4RLDataset
from jaxrl.evaluation import evaluate

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_enum('dataset_name', 'd4rl', ['d4rl', 'awac', 'rl_unplugged'],
                  'Dataset name.')
flags.DEFINE_string('id', 'baseline', 'Experiment name')
flags.DEFINE_string('save_dir', './logs/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 5,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_float(
    'percentile', 100.0,
    'Dataset percentile (see https://arxiv.org/abs/2106.01345).')
flags.DEFINE_float('percentage', 100.0,
                   'Pencentage of the dataset to use for training.')
flags.DEFINE_string('mdl_strat', "naive", 'How to increase dataset size')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/bc_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    kwargs = dict(FLAGS.config)
    algo_name = kwargs.pop('algo')
    pprint(kwargs)
    run_name = os.path.join(FLAGS.env_name, f"{algo_name}_{FLAGS.id}", f"seed{FLAGS.seed}")
    logdir = os.path.join(FLAGS.save_dir, run_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")
    summary_writer = Logger(logdir, 0)

    video_save_folder = None if not FLAGS.save_video else os.path.join(
        logdir, 'video', 'eval')

    mixed=False
    env_name = FLAGS.env_name
    game = env_name.split("-")[0]
    if "mixed" in env_name:
        mixed = True
        temp_name = env_name.replace("mixed", "medium")
        env = make_env(temp_name, FLAGS.seed, video_save_folder)
    else:
        temp_name = env_name
        env = make_env(temp_name, FLAGS.seed, video_save_folder)
    dataset = MDLD4RLDataset(env, mixed=mixed, game=game)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    if FLAGS.percentage < 100.0:
        dataset.take_random(FLAGS.percentage)

    if FLAGS.percentile < 100.0:
        dataset.take_top(FLAGS.percentile)

    obs, gsteps = get_obs_gsteps(FLAGS.mdl_strat)
    elements = dataset.size
    max_steps = ceildiv(elements, obs) * gsteps

    kwargs = dict(FLAGS.config)
    algo = kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    if algo == 'sac':
        agent = SACLearner(FLAGS.seed,
                           env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'awac':
        agent = AWACLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == "bc":
        kwargs['num_steps'] = max_steps
        agent = BCLearner(FLAGS.seed,
                        env.observation_space.sample()[np.newaxis],
                        env.action_space.sample()[np.newaxis], **kwargs)
    else:
        raise NotImplementedError()

    print(f"Training {agent} algorithm")

    initial_index = 5000
    dataset.insert_index = initial_index
    eval_returns = []
    for i in tqdm(range(1, max_steps + 1),
                       smoothing=0.1,
                       miniters=int(2e3),
                       disable=not FLAGS.tqdm):
        step = i // gsteps * obs
        batch = dataset.sample(FLAGS.batch_size, i % gsteps == 0, obs)

        update_info = agent.update(batch)

        dataset.insert_index = change_mdl(obs, gsteps, dataset.insert_index, i)
        dataset.insert_index = min(dataset.insert_index, dataset.size)

        summary_writer.step = step
        if step % FLAGS.log_interval == 0 and i % gsteps == 0:
            for k, v in update_info.items():
                summary_writer.scalar(f'training/{k}', v)
            summary_writer.write(True)

        if (step % FLAGS.eval_interval == 0 and i % gsteps == 0) or i == max_steps:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)
            summary_writer.scalar(f'insert_index', dataset.insert_index )

            for k, v in eval_stats.items():
                summary_writer.scalar(f'evaluation/average_{k}s', v)
            summary_writer.write(True)

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(logdir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
