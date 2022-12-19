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

from jaxrl.agents import AWACLearner, SACLearner
from jaxrl.datasets import ReplayBuffer
from jaxrl.datasets.dataset_utils import make_env_and_dataset
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_enum('dataset_name', 'awac', ['d4rl', 'awac'], 'Dataset name.')
flags.DEFINE_string('save_dir', './logs/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer(
    'init_dataset_size', None,
    'Number of samples from the dataset to initialize the replay buffer.')
flags.DEFINE_integer('num_pretraining_steps', int(5e4),
                     'Number of pretraining steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/awac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    run_name = os.path.join(FLAGS.env_name, FLAGS.id, f"seed{FLAGS.seed}")
    logdir = os.path.join(FLAGS.save_dir, run_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")
    summary_writer = Logger(logdir, 0)

    if FLAGS.save_video:
        video_train_folder = os.path.join(logdir, 'video', 'train')
        video_eval_folder = os.path.join(logdir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed,
                                        FLAGS.dataset_name, video_train_folder)

    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

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
    else:
        raise NotImplementedError()

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 replay_buffer_size or FLAGS.max_steps)
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

    eval_returns = []
    observation, done = env.reset(), False

    # Use negative indices for pretraining steps.
    for i in tqdm(range(1 - FLAGS.num_pretraining_steps,
                             FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i >= 1:
            action = agent.sample_actions(observation)
            next_observation, reward, done, info = env.step(action)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(observation, action, reward, mask,
                                 float(done), next_observation)
            observation = next_observation
            summary_writer.step = FLAGS.num_pretraining_steps + info['total']['timesteps']
            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    summary_writer.scalar(f'training/{k}', v)
        else:
            info = {}
            info['total'] = {'timesteps': i}

        batch = replay_buffer.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            summary_writer.step = FLAGS.num_pretraining_steps + info['total']['timesteps']
            for k, v in update_info.items():
                summary_writer.scalar(f'training/{k}', v)
            summary_writer.write(True)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.scalar(f'evaluation/average_{k}s', v)
            summary_writer.write(True)

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(logdir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)