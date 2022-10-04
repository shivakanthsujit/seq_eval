from typing import Optional

import gym
from gym.wrappers import RescaleAction
from gym.wrappers.pixel_observation import PixelObservationWrapper

from jaxrl import wrappers


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name)
    else:
        domain_name, task_name = env_name.split('_')
        env = wrappers.DMCEnv(domain_name=domain_name,
                              task_name=task_name,
                              task_kwargs={'random': seed})

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = gym.wrappers.RecordVideo(env, save_folder)

    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == 'quadruped' else 0
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only,
                                      render_kwargs={
                                          'pixels': {
                                              'height': image_size,
                                              'width': image_size,
                                              'camera_id': camera_id
                                          }
                                      })
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def get_stats(name, x):
    x_metrics = {
                    f"{name}_min": x.min(),
                    f"{name}_max": x.max(),
                    f"{name}_mean": x.mean(),
                    f"{name}_std": x.std(),
                }
    return x_metrics

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