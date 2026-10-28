"""Collect LIBERO-Plus per-task results into a policy x perturbation table.

Reads the JSONL that `libero_plus_eval_runner.py` appends as it runs, so a sweep
still in progress summarizes to what has finished so far — and shards of one suite
merge automatically, since every shard writes its own JSONL into the same
checkpoint directory.

    python ode_policy/libero/summarize_libero_plus.py --suite libero_10 --epoch 100

Success rates are marginals: an instance carrying two perturbations counts toward
both, which is how the paper reports them.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

DIMENSIONS = ("camera", "robot_init", "layout", "background", "light", "noise")
POLICY_ORDER = ["ode", "ode_dit", "fm_dit", "fm_unet", "a2a_dit", "ddpm_dit", "ddpm_unet", "act"]


def load(root, suite, epoch):
    """-> {policy: {dim: [0/1, ...]}} plus a per-policy instance count."""
    out, seen = {}, {}
    # <root>/<run>/<training suite>/checkpoints/: the training suite may be libero_4suite
    ckpt_dirs = os.path.join(root, "*", "*", "checkpoints")
    paths = (glob.glob(os.path.join(ckpt_dirs, f"plus_{suite}_ep{epoch}.jsonl"))
             + glob.glob(os.path.join(ckpt_dirs, f"plus_{suite}_ep{epoch}_shard*.jsonl")))
    for path in paths:
        policy = path.split(os.sep)[-4]
        if policy.startswith("_"):
            continue
        by_dim = out.setdefault(policy, collections.defaultdict(list))
        done = seen.setdefault(policy, set())
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from a killed job
                if r["task"] in done:
                    continue  # same instance in two shards / a resumed rerun
                done.add(r["task"])
                for d in r["dims"]:
                    by_dim[d].append(r["success"])
    return out, {p: len(s) for p, s in seen.items()}


def baseline(root, suite):
    """Standard-LIBERO best success rate, for the 'clean' column."""
    out = {}
    paths = (glob.glob(os.path.join(root, "*", "*", "checkpoints", f"eval_{suite}_ep*.json"))
             + glob.glob(os.path.join(root, "*", "*", "checkpoints", f"evalode_{suite}_ep*_r8c.json")))
    for path in paths:
        policy = path.split(os.sep)[-4]
        if policy.startswith("_"):
            continue
        m = re.search(r"ep(\d+)(_r8c)?\.json$", path)
        if not m:
            continue
        try:
            v = json.load(open(path))["__overall__"]
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        out[policy] = max(out.get(policy, 0.0), v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="il_outputs")
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--epoch", required=True, help="checkpoint epoch to summarise, e.g. 100")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    data, counts = load(args.root, args.suite, args.epoch)
    if not data:
        print(f"no LIBERO-Plus results for {args.suite} under {args.root}/")
        return 1
    base = baseline(args.root, args.suite)

    pols = [p for p in POLICY_ORDER if p in data] + sorted(set(data) - set(POLICY_ORDER))
    head = ["policy", "clean"] + list(DIMENSIONS) + ["perturbed_mean", "n_done"]
    rows = []
    for pol in pols:
        cells = [pol, f"{base.get(pol, float('nan')):.1%}" if pol in base else "—"]
        rates = []
        for d in DIMENSIONS:
            v = data[pol].get(d)
            if v:
                rates.append(sum(v) / len(v))
                cells.append(f"{rates[-1]:.1%}")
            else:
                cells.append("—")
        cells.append(f"{sum(rates) / len(rates):.1%}" if rates else "—")
        cells.append(str(counts.get(pol, 0)))
        rows.append(cells)

    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    print(f"\nLIBERO-Plus · {args.suite}  (one episode per instance, zero-shot)")
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    print("  ".join("-" * w[i] for i in range(len(head))))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print("\nn per dimension:  " + "  ".join(
        f"{d}={len(data[pols[0]].get(d, []))}" for d in DIMENSIONS))

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(head)
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
