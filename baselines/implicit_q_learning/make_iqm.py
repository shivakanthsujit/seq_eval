import csv
import os
import numpy as np
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

from collections import defaultdict
import glob
import json
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from tqdm import tqdm

def save_fig(fig, name, outdir):
    fname=os.path.join(outdir, f"{name}.png")
    print(f"Saving plot to {fname}")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    fname=os.path.join(outdir, f"{name}.pdf")
    print(f"Saving plot to {fname}")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    return fname


def decorate_axis(ax, wrect=10, hrect=10, labelsize="large"):
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
    # Pablos' comment
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))


def convert_to_matrix(score_dict):
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)

def load_last_in_jsonl(fname, yaxis):
    with open(fname, "r") as f:
        json_list = list(f)
    for json_str in json_list:
        if yaxis in json_str:
            result = json.loads(json_str)
    return result

def make_csv_data(scores):
    games_header = [[f"{game}_mean", f"{game}_std"] for game in ENVS]
    games_header = np.array(games_header).flatten()
    scores_header = ["alg"] + games_header.tolist()
    scores_stats = {k: [v.mean(0), v.std(0)] for k, v in scores.items()}
    scores_stats = {k: np.array([[m, s] for m, s in zip(*v)]) for k, v in scores_stats.items()}
    scores_stats = {k: v.flatten() for k, v in scores_stats.items()}
    scores_stats_data = [[k] + v.tolist() for k, v in scores_stats.items()]
    scores_stats_csv = [scores_header] + scores_stats_data
    return scores_stats_csv

def write_csv(csv_data, fname):
    with open(fname, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(csv_data)
    print(f"Saving csv to {fname}")

IQM = lambda x: metrics.aggregate_iqm(x)  # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0)  # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)

files = glob.glob("fixed_logs/all_logs/**/metrics.jsonl", recursive=True)
yaxis = "eval_returns"

# files = glob.glob("logs/**/metrics.jsonl", recursive=True)
# yaxis="evaluation/average_returns"

data = defaultdict(lambda: defaultdict(list))
for fname in tqdm(files):
    result = load_last_in_jsonl(fname, yaxis)
    parts = fname.split("/")
    seed = parts[-2]
    run = parts[-3]
    game = parts[-4]
    data[run][game].append(result[yaxis])

chosen_envs = [
        "halfcheetah-random-v2", "halfcheetah-medium-expert-v2", "halfcheetah-medium-replay-v2", "halfcheetah-medium-v2", 
        "hopper-random-v2", "hopper-medium-expert-v2", "hopper-medium-replay-v2", "hopper-medium-v2", "walker2d-random-v2", 
        "walker2d-medium-expert-v2", "walker2d-medium-replay-v2", "walker2d-medium-v2"
]

# chosen_envs = [
#         "finetune-halfcheetah-random-v2", "finetune-halfcheetah-medium-expert-v2", "finetune-halfcheetah-medium-replay-v2", "finetune-halfcheetah-medium-v2", 
#         "finetune-hopper-random-v2", "finetune-hopper-medium-expert-v2", "finetune-hopper-medium-replay-v2", "finetune-hopper-medium-v2", "finetune-walker2d-random-v2", 
#         "finetune-walker2d-medium-expert-v2", "finetune-walker2d-medium-replay-v2", "finetune-walker2d-medium-v2"
# ]

ENVS = sorted(chosen_envs)


def cut_data(x, seeds, envs):
    x = {k: x[k][:seeds] for k in x.keys()}
    return {k: x[k] for k in envs}

seeds = 5
outdir = "plots/agg"
algs = ["TD3BC_mdl_strat_freq1-1", "awac_mdl_strat_freq1-1", "bc_mdl_strat_freq1-1", "cql_mdl_strat_freq1-1", "iql_mdl_strat_freq1-1"]

# seeds = 3
# outdir = "plots/finetune"
# algs = ["TD3BC_mdl_strat_freq1-1_500k", "awac_mdl_strat_freq1-1_500k", "bc_mdl_strat_freq1-1_500k", "cql_mdl_strat_freq1-1_500k", "iql_mdl_strat_freq1-1_500k"]

# outdir = "plots/method"
# algs = ["iql_mdl_strat_freq1-1", "iql_mdl_baseline"]

os.makedirs(outdir, exist_ok=True)

env_scores = {alg: convert_to_matrix(cut_data(data[alg], seeds, ENVS)) for alg in algs}
normalized_env_scores = {alg: scores / 100 for alg, scores in env_scores.items()}

mean_scores_csv = make_csv_data(env_scores)
fname = os.path.join(outdir, "mean_scores.csv")
write_csv(mean_scores_csv, fname)

mean_norm_scores_csv = make_csv_data(normalized_env_scores)
fname = os.path.join(outdir, "mean_norm_scores.csv")
write_csv(mean_norm_scores_csv, fname)

# @title setup colors

colors = sns.color_palette("colorblind")
color_idxs = [0, 3, 4, 2, 1] + list(range(9, 4, -1))
COLOR_DICT = dict(zip(algs, [colors[idx] for idx in color_idxs]))

legend = {
    "TD3BC_mdl_strat_freq1-1": "TD3+BC",
    "awac_mdl_strat_freq1-1": "AWAC",
    "bc_mdl_strat_freq1-1": "BC",
    "cql_mdl_strat_freq1-1": "CQL",
    "iql_mdl_strat_freq1-1": "IQL",
    }

# legend = {
#     "TD3BC_mdl_strat_freq1-1_500k": "TD3+BC",
#     "awac_mdl_strat_freq1-1_500k": "AWAC",
#     "bc_mdl_strat_freq1-1_500k": "BC",
#     "cql_mdl_strat_freq1-1_500k": "CQL",
#     "iql_mdl_strat_freq1-1_500k": "IQL",
#     }

# legend = {
#     "iql_mdl_baseline": "Normal",
#     "iql_mdl_strat_freq1-1": "pMDL",
#     }

COLOR_DICT = dict(zip(list(legend.values()), [colors[idx] for idx in color_idxs]))
xlabel_y_coordinate = -0.2

aggregate_func = lambda x: np.array([MEDIAN(x), IQM(x), MEAN(x), OG(x)])
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    normalized_env_scores, aggregate_func, reps=50000)
scoreswlegend = {legend[k]: v for k, v in aggregate_scores.items()}
intervalswlegend = {legend[k]: v for k, v in aggregate_interval_estimates.items()}
fig, axes = plot_utils.plot_interval_estimates(
    scoreswlegend,
    intervalswlegend,
    metric_names = ['Median', 'IQM', 'Mean', 'Optimality Gap'],
    algorithms=list(legend.values()),
    colors=COLOR_DICT,
    xlabel_y_coordinate=xlabel_y_coordinate,
    xlabel='Max Normalized Score')
save_fig(fig, "agg", outdir)

aggregate_func = lambda x: np.array([MEDIAN(x), MEAN(x)])
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    normalized_env_scores, aggregate_func, reps=50000)
scoreswlegend = {legend[k]: v for k, v in aggregate_scores.items()}
intervalswlegend = {legend[k]: v for k, v in aggregate_interval_estimates.items()}
fig, axes = plot_utils.plot_interval_estimates(
    scoreswlegend,
    intervalswlegend,
    metric_names = ['Median', 'Mean'],
    algorithms=list(legend.values()),
    colors=COLOR_DICT,
    xlabel_y_coordinate=xlabel_y_coordinate,
    xlabel='Max Normalized Score')
save_fig(fig, "agg_mean_median", outdir)

aggregate_func = lambda x: np.array([IQM(x), OG(x)])
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    normalized_env_scores, aggregate_func, reps=50000)
scoreswlegend = {legend[k]: v for k, v in aggregate_scores.items()}
intervalswlegend = {legend[k]: v for k, v in aggregate_interval_estimates.items()}
fig, axes = plot_utils.plot_interval_estimates(
    scoreswlegend,
    intervalswlegend,
    metric_names = ['IQM', 'Optimality Gap'],
    algorithms=list(legend.values()),
    colors=COLOR_DICT,
    xlabel_y_coordinate=xlabel_y_coordinate,
    xlabel='Max Normalized Score')
save_fig(fig, "agg_iqm", outdir)