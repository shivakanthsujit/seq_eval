import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import jax
import jax.numpy as jnp
import flax

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import gym
import d4rl

import absl.app
import absl.flags
from tqdm import tqdm

from .conservative_sac import ConservativeSAC
from .replay_buffer import change_mdl, get_obs_gsteps, get_d4rl_dataset, subsample_batch, subsample_mdl_batch
from .jax_utils import batch_to_jax
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, ceildiv, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger


FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=1000,
    seed=42,
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=2000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    mdl=True,
    mdl_strat="naive",

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=wandb_logger.config.output_dir,
        # base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    mixed=False
    env_name = FLAGS.env
    game = env_name.split("-")[0]
    if "mixed" in env_name:
        mixed = True
        temp_name = env_name.replace("mixed", "medium")
        env = gym.make(temp_name)
    else:
        env = gym.make(env_name)
    eval_sampler = TrajSampler(env.unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env, mixed=mixed, game=game)
    
    print("Before")
    print(dataset['rewards'].min(), dataset['rewards'].max(), dataset['rewards'].mean(), dataset['rewards'].std())
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    print("After")
    print(dataset['rewards'].min(), dataset['rewards'].max(), dataset['rewards'].mean(), dataset['rewards'].std())
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]

    policy = TanhGaussianPolicy(
        observation_dim, action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset
    )
    qf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf)
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params['policy'])

    viskit_metrics = {}

    assert FLAGS.mdl
    obs, gsteps = get_obs_gsteps(FLAGS.mdl_strat)

    n_samples = dataset['observations'].shape[0]
    max_steps = ceildiv(n_samples, obs) * gsteps
    if mixed:
        print(f"Shuffling indices according to mixed profile.")
        chose = 333000
        a = [i * chose + np.random.permutation(chose) for i in range(3)]
        indices = np.concatenate(a)
        assert indices.shape[0] == n_samples
    else:
        indices = np.random.permutation(n_samples)
    sampled_indx = np.zeros(n_samples)
    insert_index = 5000
    train_iter = 0
    eval_freq = 10000
    log_freq = 5000
    start_time = time.time()
    for i in tqdm(range(max_steps), miniters=int(2e3)):
        epoch = i // gsteps
        step = i // gsteps * obs
        metrics = {'epoch': step, "insert_index": insert_index}

        train_iter += 1
        insert_index = change_mdl(obs, gsteps, insert_index, train_iter)
        insert_index = min(insert_index, n_samples)
        indx, batch = subsample_mdl_batch(dataset, indices, insert_index, FLAGS.batch_size, sample_last=i % gsteps == 0, obs=obs, mixed=mixed)
        sampled_indx[indx] += 1
        batch = batch_to_jax(batch)
        metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))

        with Timer() as eval_timer:
            if (step % eval_freq == 0 and i % gsteps == 0) or i == (max_steps - 1):
                trajs = eval_sampler.sample(
                    sampler_policy.update_params(sac.train_params['policy']),
                    FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                )
                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = time.time() - start_time
        metrics['eval_time'] = eval_timer()
        # metrics['epoch_time'] = train_timer() + eval_timer()
        if step % log_freq == 0 and i % gsteps == 0:
            wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(viskit_metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)
