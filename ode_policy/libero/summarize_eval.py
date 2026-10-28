"""Success rates of one or more training runs: epoch x suite, plus the four-suite mean.

Reads the result jsons the evaluators write next to each checkpoint:
``evalode_<suite>_ep<E><tag>.json`` (ODE policies; <tag> encodes non-default evaluator
flags, e.g. ``_r8c`` for the default protocol) and ``eval_<suite>_ep<E>.json`` (all others).
Each (run, tag) is one table; a CSV with every row is written alongside.

    python ode_policy/libero/summarize_eval.py il_outputs/ode_E1_s42 il_outputs/fm_dit_E1_s42
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
NAME = re.compile(r"^(?:evalode|eval)_(libero_object|libero_spatial|libero_goal|libero_10)_ep(\d+)(.*)\.json$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories, e.g. il_outputs/ode_E1_s42")
    p.add_argument("--out_csv", default="eval_summary.csv")
    args = p.parse_args()

    rows = []
    for run in args.runs:
        table = collections.defaultdict(dict)    # (tag) -> {(epoch, suite): success}
        for path in glob.glob(os.path.join(run, "*", "checkpoints", "*.json")):
            m = NAME.match(os.path.basename(path))
            if not m:
                continue
            suite, epoch, tag = m.group(1), int(m.group(2)), m.group(3)
            with open(path) as f:
                table[tag][(epoch, suite)] = json.load(f)["__overall__"]
        if not table:
            print(f"{run}: no evaluation results")
        for tag, res in sorted(table.items()):
            print(f"\n{run} {tag or ''}")
            print(f"{'epoch':>6}" + "".join(f"{s[7:]:>9}" for s in SUITES) + f"{'mean':>9}")
            for epoch in sorted({e for e, _ in res}):
                vals = [res.get((epoch, s)) for s in SUITES]
                mean = sum(vals) / 4 if all(v is not None for v in vals) else None
                cell = lambda v: f"{100 * v:9.1f}" if v is not None else f"{'-':>9}"
                print(f"{epoch:>6}" + "".join(cell(v) for v in vals) + cell(mean))
                rows.append([run, tag, epoch] + [v if v is not None else "" for v in vals]
                            + [mean if mean is not None else ""])
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "tag", "epoch"] + SUITES + ["mean"])
        w.writerows(rows)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
