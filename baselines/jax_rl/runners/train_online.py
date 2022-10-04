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
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

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
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(2e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of environment steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/awac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    kwargs = dict(FLAGS.config)
    algo_name = kwargs.pop('algo')
    pprint(kwargs)
    env_name = FLAGS.env_name.split("-")[0]
    run_name = os.path.join(f"online-{env_name}", f"{algo_name}_{FLAGS.id}", f"seed{FLAGS.seed}")
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

    env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    # dataset = MDLD4RLDataset(env)
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
    elif algo == "bc":
        kwargs['num_steps'] = FLAGS.max_steps
        agent = BCLearner(FLAGS.seed,
                        env.observation_space.sample()[np.newaxis],
                        env.action_space.sample()[np.newaxis], **kwargs)
    else:
        raise NotImplementedError()

    print(f"Training {agent} algorithm")

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 int(2e6))
    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask,
                                float(done), next_observation)
        observation = next_observation
        
        summary_writer.step = i
        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                summary_writer.scalar(f'training/{k}', v)
        
        if replay_buffer.size > FLAGS.start_training:
            for _ in range(FLAGS.updates_per_step):
                batch = replay_buffer.sample(FLAGS.batch_size)
                update_info = agent.update(batch)
        else:
            update_info = None

        if i % FLAGS.log_interval == 0:
            summary_writer.scalar(f'replay_buffer_size', replay_buffer.size)
            summary_writer.scalar(f'replay_buffer/min', replay_buffer.rewards.min())
            summary_writer.scalar(f'replay_buffer/max', replay_buffer.rewards.max())
            summary_writer.scalar(f'replay_buffer/mean', replay_buffer.rewards.mean())
            if update_info is not None:
                for k, v in update_info.items():
                    summary_writer.scalar(f'training/{k}', v)
            summary_writer.write(True)

        if i % FLAGS.eval_interval == 0 or i == FLAGS.max_steps:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.scalar(f'evaluation/average_{k}s', v)
            summary_writer.write(True)

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(logdir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
