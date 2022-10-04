from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Dict

import flax.linen as nn
import gym
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats


class Logger:
    def __init__(self, logdir: Path, step: int):
        self._logdir = Path(logdir)
        self.writer = SummaryWriter(log_dir=str(self._logdir))
        self._scalars = defaultdict(list)
        self._images = {}
        self._videos = {}
        self._last_step = None
        self._last_time = None
        self.step = step

    def scalar(self, name, value):
        value = float(value)
        self._scalars[name].append(value)

    def video(self, name, value, log_to_tensorboard=True):
        self._videos[name] = [np.array(value), log_to_tensorboard]

    def write(self, fps=False):
        scalars = {k: np.mean(v) for k, v in self._scalars.items()}
        scalars = list(scalars.items())
        if len(scalars) == 0:
            return
        if fps:
            scalars.append(("perf/fps", self._compute_fps(self.step)))
        video_fps = 5
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": self.step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            prefix = "" if "/" in name else "scalars/"
            self.writer.add_scalar(prefix + name, np.mean(value), self.step)
        for name, value in self._videos.items():
            write_gif(name, value[0], self._logdir, video_fps)
            if value[1] is True:
                self.writer.add_video(name, value[0], self.step, video_fps)

        self._scalars = defaultdict(list)
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def close(self):
        self.writer.close()

def write_gif(name: str, frames: np.ndarray, logdir: Path, fps: int):
    B, T, C, H, W = frames.shape
    frames = frames.transpose((1, 3, 0, 4, 2)).reshape((T, H, B * W, C))
    frames = np.clip(frames * 255.0, 0, 255.0).astype(np.uint8)
    video_dir = logdir / "gifs"
    video_dir.mkdir(parents=True, exist_ok=True)
    filename = video_dir / f"{name.replace('/', '_')}.gif"
    write_gif_to_disk(frames, filename, fps)


def write_gif_to_disk(frames: np.ndarray, filename: str, fps: int = 5):
    from moviepy.editor import ImageSequenceClip

    try:
        clip = ImageSequenceClip(list(frames), fps=fps)
        clip.write_gif(filename, fps=fps, logger=None)
        tqdm.write(f"GIF saved to {filename}")
    except Exception as e:
        tqdm.write(frames.shape)
        tqdm.write("GIF Saving failed.", e)


def make_bar_plot(x, h, fname):
    assert len(h.shape) == 1
    assert x.shape == h.shape
    plt.plot(x, h)
    plt.xlim(x[0], x[-1])
    plt.ylim(0, h.max() * 1.1)
    plt.xlabel("Indices")
    plt.ylabel("Counts")
    plt.title("Frequency of sampling")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=120)
    plt.close()