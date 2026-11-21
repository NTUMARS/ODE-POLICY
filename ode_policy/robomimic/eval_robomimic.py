"""Evaluate an ODE or baseline checkpoint on robomimic tasks with the LIBERO rollout.

ode_policy/libero/eval_ode_policy.py's rollout with the env swapped for RobomimicEnv: the same pose tracker,
ODE carry state, gripper handling and execution metrics. Episodes start from env.reset() with seeds 0..n-1
(fresh resets, as robomimic and Diffusion Policy), horizon 700 steps for Tool Hang, 400 otherwise.

    python ode_policy/robomimic/eval_robomimic.py --ckpt il_outputs/ode/rm4/checkpoints/50.ckpt   # every task
    python ode_policy/robomimic/eval_robomimic.py --ckpt ... --task square --num_episodes 20
    # -> <ckpt dir>/evalrm_ep<E>_<task>.json  (success, __exec__: jerk and inference time)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import zarr
from loguru import logger as log

sys.path.insert(0, os.getcwd())
import ode_policy.libero.eval_ode_policy as E  # noqa: E402
from ode_policy.libero import exec_metrics  # noqa: E402
from ode_policy.libero.libero_eval_runner import _load_policy  # noqa: E402
from ode_policy.robomimic.robomimic_env import TASKS, RobomimicEnv  # noqa: E402


def task_embedding(cfg, task):
    """The task's CLIP embedding row in the training zarr (zeros for a single-task zarr)."""
    if "task_emb" not in cfg.shape_meta.obs:
        return None
    z = zarr.open(str(cfg.dataset_config.zarr_path), mode="r")
    names = list(z.attrs.get("task_names", []))
    if len(names) > 1:
        assert task in names, f"{task} not among the training tasks {names}"
        return np.asarray(z["meta/task_emb"][names.index(task)], np.float32)
    return np.zeros(int(cfg.shape_meta.obs.task_emb.shape[0]), np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default=None, help="default: every task of the training zarr")
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--image_size", type=int, default=240)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    import torch
    device = torch.device(args.device)
    policy, cfg = _load_policy(args.ckpt, device, prefer_ema=True)
    is_ode = hasattr(policy, "field")
    with_wrist = "wrist_cam" in cfg.shape_meta.obs
    names = list(zarr.open(str(cfg.dataset_config.zarr_path), mode="r").attrs.get("task_names", []))
    predict = policy.predict_action
    for task in [args.task] if args.task else names:
        hdf5, horizon, _ = TASKS[task]
        seeds = iter(range(10 ** 6))

        class Env(RobomimicEnv):
            def reset(self):
                np.random.seed(next(seeds))     # robosuite's placement samplers draw from np.random
                return super().reset()

        E.make_env = lambda bddl, camera_heights, camera_widths, with_wrist, control_freq, _h=hdf5: Env(
            _h, image_size=camera_heights, with_wrist=with_wrist, control_freq=control_freq)
        rec = exec_metrics.Recorder()
        exec_metrics.instrument(policy, rec, [E])
        rec.task = task
        # ODE: virtual-target poses, carried from chunk to chunk; baselines: the OSC delta commands
        sr, _, _ = E._rollout_task(policy, cfg, None, None, args.num_episodes, horizon, 0, args.image_size, device,
                                   task_embedding(cfg, task), with_wrist, replan_every=8, gain=1.0,
                                   carry_state=is_ode, action_space="pose" if is_ode else "delta")
        policy.predict_action = predict      # drop this task's timer
        rows, ex = exec_metrics.summarize(rec, 0)
        log.info(f"{task}: success {sr:.2%} ({args.num_episodes} episodes)")
        out = pathlib.Path(args.ckpt).parent / f"evalrm_ep{pathlib.Path(args.ckpt).stem}_{task}.json"
        json.dump({task: sr, "__overall__": sr, "__exec__": ex, "__episodes__": rows, "args": vars(args)},
                  open(out, "w"), indent=2)
        log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
