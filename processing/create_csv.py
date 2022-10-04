from collections import defaultdict
import glob
import json
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd

files = sorted(glob.glob("fixed_logs/all_logs/**/metrics.jsonl", recursive=True))

labels = "TD3BC_mdl_strat_freq1-1 TD3+BC awac_mdl_strat_freq1-1 AWAC bc_mdl_strat_freq1-1 BC cql_mdl_strat_freq1-1 CQL iql_mdl_strat_freq1-1 IQL"
labels = labels + " TD3BC_mdl_strat_freq1-1_500k TD3+BC awac_mdl_strat_freq1-1_500k AWAC bc_mdl_strat_freq1-1_500k BC cql_mdl_strat_freq1-1_500k CQL iql_mdl_strat_freq1-1_500k IQL"
labels = labels + " TD3BC_mdl_strat_freq1-1_slidingwindow TD3+BC awac_mdl_strat_freq1-1_slidingwindow AWAC bc_mdl_strat_freq1-1_slidingwindow BC cql_mdl_strat_freq1-1_slidingwindow CQL iql_mdl_strat_freq1-1_slidingwindow IQL"
labels = labels.split(" ")
legend = {}
for k, v in zip(labels[:-1: 2], labels[1: :2]):
    legend[k] = v

def load_jsonl(fname):
    with open(fname, "r") as f:
        json_list = list(f)
    records = []
    for json_str in json_list:
        result = json.loads(json_str)
        records.append(result)
    return pd.DataFrame(records)

data = defaultdict(lambda: defaultdict(list))
methods_set = set()
for fname in files:
    parts = list(Path(fname).parts)
    dataset = parts[-4]
    method = parts[-3]
    methods_set.add(method)
    record = load_jsonl(fname)
    data[dataset][method] = record.eval_returns.dropna().to_list()[-1]

scores = defaultdict(lambda: defaultdict(int))
for dataset, method_search in data.items():
    for method, score in method_search.items():
        scores[dataset][method] = np.mean(score)

def filter_data(dataset, dataset_search, method_search):
    dataset_set = set()
    method_set = set()
    for dataset, methods in scores.items():
        if dataset_search.search(dataset):
            dataset_set.add(dataset)
            for method in methods.keys():
                if method_search.search(method):
                    method_set.add(method)
    dataset_set = sorted(dataset_set)
    method_set = sorted(method_set)
    return dataset_set, method_set

def create_csv(scores, dataset_set, method_set):
    d4rl_data = [["-" for _ in method_set] for _ in dataset_set]
    for i, dataset in enumerate(dataset_set):
        for j, method in enumerate(method_set):
            d4rl_data[i][j] = scores[dataset][method]

    csv_header = ["Dataset"] + [legend[m] for m in method_set]
    csv_data = [[d] + np.round(d4rl_data[i], 2).tolist() for i, d in enumerate(dataset_set)]
    df = pd.DataFrame(csv_data, columns=csv_header)
    return df

save_dir = "temp/perf"
os.makedirs(save_dir, exist_ok=True)
name="d4rl"
dataset_search = re.compile("(^half|^hopper|^walker).*(random|medium)")
method_search = re.compile("mdl_strat_freq1-1")
dataset_set, method_set = filter_data(dataset, dataset_search, method_search)
df = create_csv(scores, dataset_set, method_set)
fname = os.path.join(save_dir, f"{name}.csv")
df.to_csv(fname, index=False)

name="finetune"
dataset_search = re.compile("finetune")
method_search = re.compile("mdl_strat_freq1-1_500k$")
dataset_set, method_set = filter_data(dataset, dataset_search, method_search)
df = create_csv(scores, dataset_set, method_set)
fname = os.path.join(save_dir, f"{name}.csv")
df.to_csv(fname, index=False)

name="mixed"
dataset_search = re.compile("mixed")
method_search = re.compile("slidingwindow")
dataset_set, method_set = filter_data(dataset, dataset_search, method_search)
df = create_csv(scores, dataset_set, method_set)
fname = os.path.join(save_dir, f"{name}.csv")
df.to_csv(fname, index=False)
