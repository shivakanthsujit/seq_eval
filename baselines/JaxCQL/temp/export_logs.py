from collections import defaultdict
import glob
import json
from pathlib import Path
import os

files = sorted(glob.glob("fixed_logs/export_logs/**/metrics.jsonl", recursive=True))

yaxis = "average_normalizd_return"
new_yaxis = "eval_returns"
for fname in files:
    with open(fname, "r") as f:
        json_list = list(f)
    new_json_list = []
    for json_str in json_list:
        result = json.loads(json_str)
        if yaxis in json_str:
            result[new_yaxis] = 100 * result[yaxis]
        new_json_list.append(result)
    parts = list(Path(fname).parts)
    parts[-5] = "uniform_name"
    new_fname = os.path.join(*parts)
    os.makedirs(os.path.dirname(new_fname), exist_ok=True)
    print(f"Writing to {new_fname}")
    with Path(new_fname).open("w") as f:
        for res in new_json_list:
            f.write(json.dumps(res) + "\n")
