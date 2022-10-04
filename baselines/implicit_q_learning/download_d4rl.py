from collections import defaultdict
import d4rl
import gym

games=["halfcheetah", "walker2d", "hopper"]
levels=["random", "medium", "medium-expert", "medium-replay"]
version=2

dd = {}
lens = defaultdict(lambda: defaultdict())
for game in games:
    for level in levels:
        env_name = f"{game}-{level}-v{version}"
        env= gym.make(env_name)
        data = d4rl.qlearning_dataset(env)
        size = data["observations"].shape[0]
        lens[game][level] = size
        dd[env_name] = size
        print(f"{game} {level}: {size}")

print(dd)