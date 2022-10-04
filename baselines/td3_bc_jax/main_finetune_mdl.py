import numpy as np
import gym
import argparse
import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"]="1"

import d4rl
from logger import Logger
from tqdm import tqdm

import utils
import TD3_BC


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=5):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.0
    for _ in tqdm(range(eval_episodes), leave=False):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    tqdm.write("---------------------------------------")
    tqdm.write(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
    tqdm.write("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--logdir", default="logs")  # Policy name
    parser.add_argument("--id", default="baseline")  # Policy name
    parser.add_argument(
        "--env", default="hopper-medium-v0"
    )  # OpenAI gym environment name
    parser.add_argument(
        "--seed", default=0, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--eval_freq", default=5e3, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--max_timesteps", default=1e6, type=int
    )  # Max time steps to run environment
    parser.add_argument(
        "--save_model", action="store_true"
    )  # Save model and optimizer parameters
    parser.add_argument(
        "--load_model", default=""
    )  # Model load file name, "" doesn't load, "default" uses file_name
    # TD3
    parser.add_argument(
        "--expl_noise", default=0.1
    )  # Std of Gaussian exploration noise
    parser.add_argument(
        "--batch_size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument(
        "--policy_noise", default=0.2
    )  # Noise added to target policy during critic update
    parser.add_argument(
        "--noise_clip", default=0.5
    )  # Range to clip target policy noise
    parser.add_argument(
        "--policy_freq", default=2, type=int
    )  # Frequency of delayed policy updates
    # TD3 + BC
    parser.add_argument("--alpha", default=2.5)
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--mdl_strat", default="naive")
    args = parser.parse_args()

    file_name = f"model"
    print("---------------------------------------")
    print(f"Policy: {args.id}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    run_name = os.path.join(f"finetune-{args.env}", f"TD3BC_{args.id}", f"seed{args.seed}")
    logdir = os.path.join(args.logdir, run_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")
    summary_writer = Logger(logdir, 0)

    # if not os.path.exists():
    #     os.makedirs("./results")
    model_dir = os.path.join(logdir, "models")
    if args.save_model:
        os.makedirs(model_dir, exist_ok=True)

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    # torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # TD3 + BC
        "alpha": args.alpha,
    }

    # Initialize policy
    policy = TD3_BC.TD3_BC(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        mfile = os.path.join(model_dir, policy_file)
        policy.load(mfile)

    replay_buffer = utils.MDLReplayBuffer(state_dim, action_dim, int(2e6))
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    replay_buffer.reset()
    obs, gsteps = utils.get_obs_gsteps(args.mdl_strat)
    elements = replay_buffer.size
    max_timesteps = utils.ceildiv(elements, obs) * gsteps

    initial_index = 5000
    replay_buffer.insert_index = initial_index
    evaluations = []
    for t in tqdm(range(int(max_timesteps))):
        step = t // gsteps * obs

        batch = replay_buffer.sample(args.batch_size, t % gsteps == 0, obs)
        policy.train(batch)

        replay_buffer.insert_index = utils.change_mdl(obs, gsteps, replay_buffer.insert_index, t)
        replay_buffer.insert_index = min(replay_buffer.insert_index, replay_buffer.size)

        # Evaluate episode
        if step % args.eval_freq == 0 or t == int(max_timesteps - 1):
            tqdm.write(f"Time steps: {t+1}")
            eval_score = eval_policy(policy, args.env, args.seed, mean, std)
            evaluations.append(eval_score)
            save_dir = os.path.join(logdir, "results")
            np.save(save_dir, evaluations)
            summary_writer.step = step
            summary_writer.scalar(f'insert_index', replay_buffer.insert_index)
            summary_writer.scalar(f'eval/reward', eval_score)
            summary_writer.write(True)
            if args.save_model:
                mfile = os.path.join(model_dir, file_name)
                policy.save(mfile)

    tqdm.write(f"Begin online finetuning for {args.max_timesteps} timesteps")
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, int(2e6))
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        tqdm.write(f"Normalise states in buffer")
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    current_step = step
    state, done, ep_rew = env.reset(), False, 0
    for t in tqdm(range(int(args.max_timesteps))):
        step = t

        state = (np.array(state).reshape(1, -1) - mean) / std
        action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)

        next_state_norm = (np.array(next_state).reshape(1, -1) - mean) / std

        replay_buffer.add(state, action, next_state_norm, reward, done)
        state = next_state
        ep_rew += reward
        if done:
            summary_writer.scalar(f'train/reward', ep_rew)
            state, done, ep_rew = env.reset(), False, 0

        batch = replay_buffer.sample(args.batch_size)
        policy.train(batch)

        # Evaluate episode
        if step % args.eval_freq == 0 or t == int(args.max_timesteps - 1):
            tqdm.write(f"Time steps: {t+1}")
            eval_score = eval_policy(policy, args.env, args.seed, mean, std)
            evaluations.append(eval_score)
            save_dir = os.path.join(logdir, "results")
            np.save(save_dir, evaluations)
            summary_writer.step = step + current_step
            summary_writer.scalar(f'replay_buffer_size', replay_buffer.size)

            # summary_writer.scalar(f'replay/min', replay_buffer.reward.min())
            # summary_writer.scalar(f'replay/max', replay_buffer.reward.max())
            # summary_writer.scalar(f'replay/mean', replay_buffer.reward.mean())

            summary_writer.scalar(f'eval/reward', eval_score)
            summary_writer.write(True)
            if args.save_model:
                mfile = os.path.join(model_dir, file_name)
                policy.save(mfile)
