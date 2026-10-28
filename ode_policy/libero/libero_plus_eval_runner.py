"""Evaluate a checkpoint on LIBERO-Plus, reporting success per perturbation type.

LIBERO-Plus (arXiv 2510.13626) is a drop-in replacement for the `libero` package
that keeps the original suite NAMES but replaces each suite's 10 tasks with a few
thousand perturbed instances. Protocol: train on standard LIBERO, evaluate here
zero-shot, one episode per task instance.

Run it from a conda env with LIBERO-plus installed instead of LIBERO — see
`README.md`. The checkpoints do NOT need retraining: they carry their own weights,
config and normalizer, and know nothing about the simulator.

    python ode_policy/libero/libero_plus_eval_runner.py \
        --ckpt il_outputs/ode/libero_4suite/checkpoints/100.ckpt \
        --task_suite libero_10

Which perturbation a task carries is encoded in its NAME; the package exposes no
API for it. `classify()` below parses those names. It is not guesswork: the
per-dimension totals it produces reproduce Table 7 of the paper exactly across all
four suites (camera 1599, robot_init 1550, language 1537, light 1142, background
1076, noise 1601, layout 1525 = `add` 829 + `levelN_sampleM` 696; total 10030).
`verify_taxonomy()` re-checks those totals and is run by `--verify_only`.

Language perturbation is EXCLUDED by default. Evaluating it would need the
rewritten instruction encoded live with the same CLIP tower the zarr records —
not implemented here.

Note that `task.language` is NOT usable as the lookup key for the others either:
LIBERO-plus builds the instruction from the FILENAME unless the name contains
"_language_", so a perturbed instance reports its suffix as part of the sentence
("... put the moka pot on it table 1"). Instances are resolved through
`base_task_name()` instead.

Progress is appended to a JSONL as each task finishes, and a rerun skips what that
file already holds. A suite is a few thousand episodes and a job can hit walltime;
resubmitting continues rather than starting over.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import sys

import numpy as np
import torch
from loguru import logger as log

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ode_policy.libero.libero_eval_runner import (  # noqa: E402
    SUITE_MAX_STEPS,
    _TaskEmbTable,
    _load_policy,
    _rollout_task,
)

# --------------------------------------------------------------------------- #
# perturbation taxonomy
# --------------------------------------------------------------------------- #
# A camera instance encodes its pose inline; this tuple is the UNPERTURBED pose,
# so `view_0_0_100_0_0` means "default camera" and `initstate_0` "default start".
# Both tokens appear together on every instance that carries either, which is why
# they cannot be told apart by presence alone — only by value.
_DEFAULT_VIEW = ("0", "0", "100", "0", "0")
_VIEW_RE = re.compile(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate_(\d+)")

DIMENSIONS = ("camera", "robot_init", "layout", "background", "light", "noise", "language")

# LIBERO-plus derives a task's language from its FILENAME unless the name contains
# "_language_" (see benchmark/__init__.py:grab_language_from_filename), so a
# perturbed instance reports its suffix as part of the instruction:
#   KITCHEN_SCENE3_turn_on_the_stove_..._table_1
#     -> "turn on the stove and put the moka pot on it table 1"
# That string is not in the checkpoint's task_emb table and every lookup fails.
# Cutting at the first perturbation token recovers the base task name, which the
# training zarr does record; verified to resolve all 2519 libero_10 instances.
_BASE_CUT = re.compile(r"_(?:view|language|table|tb|add|light|noise)_\d|_level\d+_sample\d+")


def base_task_name(task_name: str) -> str:
    """-> the stock-LIBERO task this perturbed instance was derived from."""
    m = _BASE_CUT.search(task_name)
    return task_name[:m.start()] if m else task_name

# Paper Table 7, summed over the four suites. verify_taxonomy() checks against it.
_PAPER_TOTALS = {"camera": 1599, "robot_init": 1550, "language": 1537, "light": 1142,
                 "background": 1076, "noise": 1601, "layout": 1525}
_PAPER_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def classify(task_name: str) -> set:
    """-> the set of perturbation dimensions this task instance carries.

    Instances can carry more than one (e.g. a moved camera *and* a new table
    texture), so a task counts toward every dimension it has; the per-dimension
    rates below are therefore marginals, matching how the paper reports them.
    """
    dims = set()
    if "_language_" in task_name:
        dims.add("language")
    if re.search(r"_noise_\d+", task_name):
        dims.add("noise")
    if re.search(r"_light_\d+", task_name):
        dims.add("light")
    if re.search(r"_add_\d+", task_name) or re.search(r"_level\d+_sample\d+", task_name):
        dims.add("layout")
    if re.search(r"_(?:table|tb)_\d+", task_name):
        dims.add("background")
    m = _VIEW_RE.search(task_name)
    if m:
        if m.group(1, 2, 3, 4, 5) != _DEFAULT_VIEW:
            dims.add("camera")
        if m.group(6) != "0":
            dims.add("robot_init")
    return dims


def verify_taxonomy() -> bool:
    """Re-derive the paper's per-dimension totals from the installed task map.

    A silent parser drift here would mislabel every number this script prints, so
    it is checked rather than trusted. Returns True when the totals match.
    """
    from libero.libero.benchmark import libero_task_map

    got = collections.Counter()
    for suite in _PAPER_SUITES:
        for name in libero_task_map[suite]:
            got.update(classify(name))
    ok = True
    log.info("perturbation taxonomy vs. paper Table 7:")
    for d in DIMENSIONS:
        want, have = _PAPER_TOTALS[d], got.get(d, 0)
        flag = "ok" if want == have else "MISMATCH"
        ok &= want == have
        log.info(f"  {d:<11} parsed {have:>5}   paper {want:>5}   {flag}")
    total = sum(got.values())
    log.info(f"  {'TOTAL':<11} parsed {total:>5}   paper {10030:>5}   "
             f"{'ok' if total == 10030 else 'MISMATCH'}")
    return ok and total == 10030


# --------------------------------------------------------------------------- #
# task selection
# --------------------------------------------------------------------------- #
def select_tasks(suite, dims_wanted: set, max_per_dim: int, shard: str, seed: int):
    """-> [(task_id, task, dims)] for the instances this job should run.

    Subsampling is stratified per dimension and deterministic in `seed`, so two
    policies compared against each other always run the identical instances.
    """
    rows = []
    for tid in range(suite.n_tasks):
        task = suite.get_task(tid)
        dims = classify(task.name)
        if not dims or not (dims & dims_wanted):
            continue  # unperturbed, or only dimensions we were not asked for
        if dims - dims_wanted:
            # e.g. a language rewrite that also moved the camera: running it would
            # book a language result under "camera". Drop it rather than mislabel.
            continue
        rows.append((tid, task, dims))

    if max_per_dim:
        rng = np.random.default_rng(seed)
        by_dim = collections.defaultdict(list)
        for r in rows:
            by_dim[tuple(sorted(r[2]))].append(r)
        kept = []
        for key in sorted(by_dim):
            group = by_dim[key]
            n = min(max_per_dim, len(group))
            idx = rng.choice(len(group), n, replace=False)
            kept += [group[i] for i in sorted(idx)]
        rows = sorted(kept, key=lambda r: r[0])

    if shard:
        i, n = (int(v) for v in shard.split("/"))
        rows = [r for k, r in enumerate(rows) if k % n == i - 1]
    return rows


# --------------------------------------------------------------------------- #
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="Checkpoint to evaluate (.ckpt)")
    p.add_argument("--task_suite", default="libero_10", choices=sorted(SUITE_MAX_STEPS))
    p.add_argument("--dims", default="camera,robot_init,layout,background,light,noise",
                   help="Comma-separated perturbation dimensions. 'language' needs live "
                        "instruction encoding and is not supported.")
    p.add_argument("--max_per_dim", type=int, default=0,
                   help="Stratified subsample: at most N instances per perturbation "
                        "combination. 0 = run everything.")
    p.add_argument("--shard", default="", help="'i/n' — run every n-th instance (1-based).")
    p.add_argument("--sample_seed", type=int, default=0,
                   help="Seed for --max_per_dim. Keep identical across policies.")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--eval_seed", type=int, default=10000)
    p.add_argument("--zarr_path", default=None)
    p.add_argument("--ode_replan_every", type=int, default=8,
                   help="ODE policies only: control steps executed per inference")
    p.add_argument("--ode_gain", type=float, default=1.0,
                   help="ODE only: tracker gain. 1 for virtual-target models; 3.91 for "
                        "measured-pose models")
    p.add_argument("--ode_reset_state", action="store_true",
                   help="ODE only: restart every chunk from the measured pose instead of "
                        "continuing from where the last integration ended (the default, "
                        "carry execution, is the best configuration)")
    p.add_argument("--ode_eps_sample", action="store_true",
                   help="ODE with a side head: draw the component and the width noise inside it "
                        "(multimodal tasks only; the default takes the component centre)")
    p.add_argument("--out_jsonl", default=None,
                   help="Per-task log, appended as the run goes. Default: next to --ckpt.")
    p.add_argument("--output_json", default=None, help="Summary. Default: next to --ckpt.")
    p.add_argument("--verify_only", action="store_true",
                   help="Check the name parser against the paper's totals and exit.")
    return p.parse_args()


def summarize(records, suite_name):
    """Per-dimension marginal success rates plus the pooled overall."""
    by_dim = collections.defaultdict(list)
    for r in records:
        for d in r["dims"]:
            by_dim[d].append(r["success"])
    out = {"__suite__": suite_name, "__n_tasks__": len(records),
           "__overall__": float(np.mean([r["success"] for r in records])) if records else 0.0}
    for d in DIMENSIONS:
        if by_dim.get(d):
            out[d] = {"success_rate": float(np.mean(by_dim[d])), "n": len(by_dim[d])}
    return out


def main():
    args = _parse_args()

    if args.verify_only:
        return 0 if verify_taxonomy() else 1
    if not args.ckpt:
        log.error("--ckpt is required (or pass --verify_only)")
        return 1

    dims_wanted = {d.strip() for d in args.dims.split(",") if d.strip()}
    unknown = dims_wanted - set(DIMENSIONS)
    if unknown:
        log.error(f"unknown dimension(s): {sorted(unknown)}; valid: {list(DIMENSIONS)}")
        return 1
    if "language" in dims_wanted:
        log.error(
            "The language dimension rewrites the instruction, which is not in the "
            "checkpoint's precomputed task_emb table — _TaskEmbTable would raise. "
            "Encoding it live with the zarr's own CLIP tower is not implemented."
        )
        return 1

    if not verify_taxonomy():
        log.error("Task-name parser disagrees with the paper's totals — the installed "
                  "LIBERO-plus differs from the one this script was written against. "
                  "Fix classify() before trusting any number below.")
        return 1

    device = torch.device(args.device)
    max_steps = args.max_steps or SUITE_MAX_STEPS[args.task_suite]
    log.info(f"Loading checkpoint: {args.ckpt}")
    policy, cfg = _load_policy(args.ckpt, device, prefer_ema=args.use_ema)

    with_wrist = "wrist_cam" in cfg.shape_meta.obs
    history_source = getattr(policy, "history_source", None)

    table = None
    if "task_emb" in cfg.shape_meta.obs:
        zarr_path = args.zarr_path or cfg.dataset_config.zarr_path
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(
                f"Checkpoint trained with task_emb but its training zarr {zarr_path} "
                f"is missing. Pass --zarr_path."
            )
        table = _TaskEmbTable(zarr_path)

    from libero.libero import benchmark as libero_bench
    from libero.libero import get_libero_path

    suite = libero_bench.get_benchmark_dict()[args.task_suite]()
    bddl_root = get_libero_path("bddl_files")

    rows = select_tasks(suite, dims_wanted, args.max_per_dim, args.shard, args.sample_seed)
    log.info(f"{len(rows)} task instances selected from {suite.n_tasks} "
             f"(dims={sorted(dims_wanted)}, max_per_dim={args.max_per_dim or 'all'}, "
             f"shard={args.shard or '1/1'})")

    ckpt_dir = pathlib.Path(args.ckpt).parent
    stem = pathlib.Path(args.ckpt).stem
    tag = f"plus_{args.task_suite}_ep{stem}" + (f"_shard{args.shard.replace('/', 'of')}"
                                                if args.shard else "")
    jsonl_path = args.out_jsonl or str(ckpt_dir / f"{tag}.jsonl")
    out_path = args.output_json or str(ckpt_dir / f"eval_{tag}.json")

    # Resume: a suite is thousands of episodes and a job can hit walltime.
    done = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line from a killed job
                done[r["task"]] = r
        log.info(f"resuming: {len(done)} instances already in {jsonl_path}")

    records = [done[t.name] for _, t, _ in rows if t.name in done]
    todo = [r for r in rows if r[1].name not in done]
    log.info(f"{len(todo)} to run, {len(records)} reused")

    fh = open(jsonl_path, "a", buffering=1)
    for k, (tid, task, dims) in enumerate(todo):
        bddl_name = getattr(task, "bddl_file_name", None) or getattr(task, "bddl_file")
        bddl_file = os.path.join(bddl_root, task.problem_folder, bddl_name)
        try:
            init_states = suite.get_task_init_states(tid)
        except Exception as e:
            log.warning(f"[{task.name}] no init states ({e}); random reset.")
            init_states = None

        task_emb = None
        if table is not None:
            # Look up by the BASE task name, never by task.language: LIBERO-plus
            # bakes the perturbation suffix into the instruction it reports.
            base = base_task_name(task.name)
            try:
                task_emb = table.lookup("", base)
            except KeyError:
                log.error(f"[{task.name}] base task {base!r} not in the task_emb "
                          f"table; skipping.")
                continue

        if hasattr(policy, "field"):
            # ODE policy: its output is a pose reference that has to be tracked, so it
            # runs through its own evaluator's rollout (chunked, 8 steps per inference)
            from ode_policy.libero.eval_ode_policy import _rollout_task as _ode_rollout
            sr, _, _ = _ode_rollout(
                policy, cfg, bddl_file, init_states, 1, max_steps, args.num_steps_wait,
                args.image_size, device, task_emb, with_wrist,
                replan_every=args.ode_replan_every, gain=args.ode_gain,
                carry_state=not args.ode_reset_state, eps_sample=args.ode_eps_sample)
        else:
            sr = _rollout_task(
                policy=policy, cfg=cfg, bddl_file=bddl_file, init_states=init_states,
                num_episodes=1, max_steps=max_steps, num_steps_wait=args.num_steps_wait,
                image_size=args.image_size, device=device, init_mode="fixed",
                eval_seed=args.eval_seed + tid * 10000, task_emb=task_emb,
                with_wrist=with_wrist, history_source=history_source,
                video_dir=None, video_fps=20, task_name=task.name,
            )
        rec = {"task": task.name, "dims": sorted(dims), "success": int(sr > 0.5)}
        fh.write(json.dumps(rec) + "\n")
        records.append(rec)
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            run = np.mean([r["success"] for r in records])
            log.info(f"  [{k + 1}/{len(todo)}] running success {run:.1%}")
    fh.close()

    summary = summarize(records, args.task_suite)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"=== {args.task_suite} · {args.ckpt} ===")
    for d in DIMENSIONS:
        if d in summary:
            log.info(f"  {d:<11} {summary[d]['success_rate']:>6.1%}  (n={summary[d]['n']})")
    log.info(f"  {'OVERALL':<11} {summary['__overall__']:>6.1%}  (n={summary['__n_tasks__']})")
    log.info(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
