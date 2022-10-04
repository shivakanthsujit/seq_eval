from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from tensorboardX import SummaryWriter
# import orjson as json
import rapidjson as json

from tqdm import tqdm

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

    def write(self, fps=False):
        scalars = {k: np.mean(v) for k, v in self._scalars.items()}
        scalars = list(scalars.items())
        if len(scalars) == 0:
            return
        if fps:
            scalars.append(("perf/fps", self._compute_fps(self.step)))

        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": self.step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            prefix = "" if "/" in name else "scalars/"
            self.writer.add_scalar(prefix + name, np.mean(value), self.step)

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