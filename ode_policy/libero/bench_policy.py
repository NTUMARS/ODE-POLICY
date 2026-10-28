"""Inference speed and motion smoothness, measured the same way for every policy.

Nothing here re-implements a rollout. The stock evaluator (fm_dit, a2a_dit, ...) and the
ODE evaluator are called unchanged; this script only
  * wraps the env so every command sent to the robot and every measured end-effector
    position is recorded, and
  * wraps policy.predict_action with a CUDA-synchronised timer,
so each policy runs under exactly the protocol its success numbers came from.

Speed
  call_ms      wall time of one predict_action call (encoder + head), mean / p50 / p95
  step_ms      total policy time / control steps executed -- cost per robot action
  encoder_ms   the observation encoder alone, on the same input (shared by all
               policies here, so head_ms = call_ms - encoder_ms is the head's cost)

Smoothness, over the policy-controlled steps of every episode (settle steps excluded)
  cmd_delta    mean |a_t - a_{t-1}| of the 6 arm channels of the OSC command, clipped to
               [-1, 1] as the controller does -- comparable across action spaces,
               since every policy ends up sending this same command
  boundary     cmd_delta at steps where a new inference started vs. all other steps;
               ratio > 1 means the motion jumps where chunks meet
  ee_jerk      RMS norm of the third derivative of the measured end-effector position
               (m/s^3) -- what the arm physically did
Every smoothness number is reported twice: over all episodes, and over successful
episodes only (suffix _succ). Failed episodes thrash, so comparing policies with
different success rates needs the second.

    python ode_policy/libero/bench_policy.py --ckpt il_outputs/ode_E1_s42/libero_4suite/checkpoints/100.ckpt \
        --task_suite libero_object --num_episodes 10
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from loguru import logger as log

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ode_policy.libero import eval_ode_policy as odev
from ode_policy.libero.pose import DT, TRACK_GAIN
from ode_policy.libero import exec_metrics
from ode_policy.libero import libero_eval_runner as stock
from ode_policy.utils.pytorch_util import dict_apply


def _instrument(policy, rec):
    exec_metrics.instrument(policy, rec, [stock, odev])


def _microbench(policy, obs, n=200):
    """Encoder-only and full-call latency on one fixed observation."""
    nobs = policy._normalize_obs(obs)
    To = policy.n_obs_steps
    enc_in = dict_apply(nobs, lambda x: x[:, :To].reshape(-1, *x.shape[2:]))

    def clock(fn):
        with torch.no_grad():
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(n):
                t = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
        return float(np.mean(ts) * 1000)

    return clock(lambda: policy.obs_encoder(enc_in)), clock(lambda: policy.predict_action(obs))


def _metrics(rec, settle, only_success=False):
    deltas, bound, inner, jerks, call_t, n_steps = [], [], [], [], [], 0
    for ep in rec.episodes:
        if only_success and not ep["success"]:
            continue
        cmd = np.clip(np.asarray(ep["cmd"])[settle:, :6], -1, 1)
        pos = np.asarray(ep["pos"])[settle:]
        if len(cmd) < 4:
            continue
        n_steps += len(cmd)
        d = np.abs(np.diff(cmd, axis=0)).mean(axis=1)          # d[k] = change INTO step k+1
        starts = {s - settle for s, _ in ep["calls"]}           # step index where a chunk began
        for k in range(1, len(cmd)):
            (bound if k in starts else inner).append(d[k - 1])
        deltas.extend(d)
        j = np.diff(pos, n=3, axis=0) / DT ** 3
        jerks.extend(np.linalg.norm(j, axis=1) ** 2)
        call_t.extend(t for _, t in ep["calls"])
    call_t = np.asarray(call_t, dtype=float) * 1000
    call_t = call_t[np.isfinite(call_t)]          # the warm-up call is not timed
    return {
        "call_ms_mean": float(call_t.mean()), "call_ms_p50": float(np.percentile(call_t, 50)),
        "call_ms_p95": float(np.percentile(call_t, 95)), "n_calls": int(len(call_t)),
        "step_ms": float(call_t.sum() / max(n_steps, 1)), "n_steps": int(n_steps),
        "cmd_delta": float(np.mean(deltas)),
        "cmd_delta_boundary": float(np.mean(bound)) if bound else float("nan"),
        "cmd_delta_inner": float(np.mean(inner)) if inner else float("nan"),
        "boundary_ratio": float(np.mean(bound) / np.mean(inner)) if bound and inner else float("nan"),
        "ee_jerk_rms": float(np.sqrt(np.mean(jerks))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task_suite", default="libero_object", choices=sorted(stock.SUITE_MAX_STEPS))
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--replan_every", type=int, default=8, help="ODE only")
    p.add_argument("--reset_state", action="store_true",
                   help="ODE only: restart every chunk from the measured pose instead of "
                        "continuing from the reference pose the last one reached")
    p.add_argument("--gain", type=float, default=TRACK_GAIN,
                   help="ODE only: tracker gain (1 for virtual-target models)")
    p.add_argument("--n_tasks", type=int, default=0, help="only the first N tasks (0 = all)")
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--output_json", default="")
    args = p.parse_args()
    args.carry_state = not args.reset_state

    device = torch.device("cuda:0")
    policy, cfg = stock._load_policy(args.ckpt, device, prefer_ema=True)
    is_ode = hasattr(policy, "field")
    with_wrist = "wrist_cam" in cfg.shape_meta.obs
    history_source = getattr(policy, "history_source", None)
    table = stock._TaskEmbTable(cfg.dataset_config.zarr_path) if "task_emb" in cfg.shape_meta.obs else None
    rec = exec_metrics.Recorder()
    _instrument(policy, rec)

    from libero.libero import benchmark as libero_bench
    from libero.libero import get_libero_path
    suite = libero_bench.get_benchmark_dict()[args.task_suite]()
    bddl_root = get_libero_path("bddl_files")
    max_steps = stock.SUITE_MAX_STEPS[args.task_suite]

    succ = []
    for tid in range(args.n_tasks or suite.n_tasks):
        task = suite.get_task(tid)
        bddl = os.path.join(bddl_root, task.problem_folder,
                            getattr(task, "bddl_file_name", None) or task.bddl_file)
        init_states = suite.get_task_init_states(tid)
        emb = table.lookup(getattr(task, "language", ""), task.name) if table else None
        if is_ode:
            sr, _, _ = odev._rollout_task(
                policy, cfg, bddl, init_states, args.num_episodes, max_steps,
                args.num_steps_wait, args.image_size, device, emb, with_wrist,
                replan_every=args.replan_every, gain=args.gain, carry_state=args.carry_state)
        else:
            sr = stock._rollout_task(
                policy, cfg, bddl, init_states, args.num_episodes, max_steps,
                args.num_steps_wait, args.image_size, device, "fixed", 0, emb,
                with_wrist, history_source)
        succ.append(sr)
        log.info(f"[{task.name}] success={sr:.0%}")

    res = {"ckpt": args.ckpt, "suite": args.task_suite, "kind": "ode" if is_ode else "chunk",
           "n_tasks": args.n_tasks or suite.n_tasks, "gain": args.gain if is_ode else None,
           "carry_state": bool(args.carry_state),
           "success": float(np.mean(succ)), "episodes": len(rec.episodes)}
    res.update(_metrics(rec, args.num_steps_wait))
    succ_only = _metrics(rec, args.num_steps_wait, only_success=True)
    res.update({f"{k}_succ": v for k, v in succ_only.items()
                if k.startswith(("cmd_delta", "boundary", "ee_jerk", "n_steps"))})
    res["n_success_episodes"] = int(sum(ep["success"] for ep in rec.episodes))
    enc_ms, call_ms = _microbench(policy, rec.sample_obs)
    res.update({"micro_encoder_ms": enc_ms, "micro_call_ms": call_ms,
                "micro_head_ms": call_ms - enc_ms,
                "params_M": sum(x.numel() for x in policy.parameters()) / 1e6,
                "encoder_params_M": sum(x.numel() for x in policy.obs_encoder.parameters()) / 1e6})
    out = args.output_json or str(pathlib.Path(args.ckpt).parent /
                                 f"bench_{args.task_suite}_ep{pathlib.Path(args.ckpt).stem}"
                                 f"{'_carry' if args.carry_state else ''}.json")
    json.dump(res, open(out, "w"), indent=2)
    log.info(json.dumps(res, indent=2))
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
