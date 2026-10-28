"""Execution-quality metrics recorded during any LIBERO evaluation, for every policy.

Both evaluators (ode_policy/libero/eval_ode_policy.py for the ODE policy, and
ode_policy/libero/libero_eval_runner.py for fm_dit / a2a_dit / ...) call
`instrument()` once after loading the policy. It wraps the environment so every command sent to
the robot and every measured end-effector position is recorded, and wraps
`policy.predict_action` with a CUDA-synchronised timer. Nothing in the rollouts changes.

Per episode (policy-controlled steps only; the settle steps at the start are excluded):
  steps        control steps until the episode ended (= completion time on a success, x 0.05 s)
  ee_jerk_rms  RMS norm of the 3rd derivative of the measured end-effector position (m/s^3)
  cmd_delta    mean |a_t - a_{t-1}| over the 6 arm channels of the OSC command (clipped to [-1, 1]);
               cmd_delta_inner over steps inside a chunk, cmd_delta_boundary over the first step
               of each new chunk (in-chunk vs. cross-chunk smoothness)
  boundary_ratio  cmd_delta at steps where a new inference started / at the other steps
                  (1 = chunks join as smoothly as the motion inside a chunk)
  call_ms      mean wall time of one predict_action call (encoder + head)
  step_ms      total inference time / control steps
The evaluators store the per-episode rows under "__episodes__" and their means (over all episodes
and over successful ones) under "__exec__" in the result json.
"""
from __future__ import annotations

import time

import numpy as np
import torch

DT = 0.05  # LIBERO control period


class Recorder:
    def __init__(self):
        self.episodes, self.cur, self.sample_obs, self.task = [], None, None, None
        self.warm = False

    def new_episode(self):
        # some rollouts reset twice before the first step; keep one record per episode
        if self.cur is not None and not self.cur["cmd"]:
            return
        self.cur = {"task": self.task, "cmd": [], "pos": [], "calls": [], "success": False}
        self.episodes.append(self.cur)


class EnvWrap:
    """Pass-through env that records commands and measured end-effector positions."""

    def __init__(self, env, rec):
        self._env, self._rec = env, rec

    def __getattr__(self, k):
        return getattr(self._env, k)

    def reset(self, *a, **k):
        self._rec.new_episode()
        return self._env.reset(*a, **k)

    def check_success(self):
        ok = self._env.check_success()
        if self._rec.cur is not None:
            self._rec.cur["success"] = bool(ok)   # the last call decides the episode
        return ok

    def step(self, action):
        out = self._env.step(action)
        if self._rec.cur is not None:
            self._rec.cur["cmd"].append(np.asarray(action, dtype=np.float64).copy())
            self._rec.cur["pos"].append(np.asarray(out[0]["robot0_eef_pos"], dtype=np.float64).copy())
        return out


def instrument(policy, rec, modules):
    """Time predict_action and wrap make_env in every module whose rollout creates envs."""
    orig = policy.predict_action

    def timed(*a, **k):
        if rec.sample_obs is None and a:
            rec.sample_obs = {kk: v.clone() for kk, v in a[0].items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if rec.cur is not None:
            # the first call of a process includes CUDA warm-up: keep its position (a chunk
            # boundary) but not its time
            dt = time.perf_counter() - t if rec.warm else float("nan")
            rec.warm = True
            rec.cur["calls"].append((len(rec.cur["cmd"]), dt))
        return out

    policy.predict_action = timed
    for mod in modules:
        make = mod.make_env
        mod.make_env = lambda *a, _m=make, **k: EnvWrap(_m(*a, **k), rec)


def episode_row(ep, settle):
    cmd = np.clip(np.asarray(ep["cmd"]).reshape(-1, 7)[settle:, :6], -1, 1)
    pos = np.asarray(ep["pos"]).reshape(-1, 3)[settle:]
    row = {"task": ep["task"], "success": bool(ep["success"]), "steps": int(len(cmd))}
    nan = float("nan")
    if len(cmd) >= 4:
        d = np.abs(np.diff(cmd, axis=0)).mean(axis=1)          # d[k] = change INTO step k+1
        starts = {s - settle for s, _ in ep["calls"]}
        bound = [d[k - 1] for k in range(1, len(cmd)) if k in starts]
        inner = [d[k - 1] for k in range(1, len(cmd)) if k not in starts]
        j = np.linalg.norm(np.diff(pos, n=3, axis=0) / DT ** 3, axis=1)
        row.update(cmd_delta=float(d.mean()), ee_jerk_rms=float(np.sqrt(np.mean(j ** 2))),
                   cmd_delta_inner=float(np.mean(inner)) if inner else nan,
                   cmd_delta_boundary=float(np.mean(bound)) if bound else nan,
                   boundary_ratio=float(np.mean(bound) / np.mean(inner)) if bound and inner else nan)
    else:
        row.update(cmd_delta=nan, ee_jerk_rms=nan, cmd_delta_inner=nan, cmd_delta_boundary=nan,
                   boundary_ratio=nan)
    calls = np.asarray([t for _, t in ep["calls"]], dtype=float) * 1000
    timed = calls[np.isfinite(calls)]
    row.update(n_calls=int(len(calls)), call_ms=float(timed.mean()) if len(timed) else nan,
               step_ms=float(timed.mean() * len(calls) / len(cmd)) if len(cmd) and len(timed) else nan)
    return row


def summarize(rec, settle):
    """-> (per-episode rows, {"all": means, "success": means over successful episodes})."""
    rows = [episode_row(ep, settle) for ep in rec.episodes]
    keys = ("steps", "ee_jerk_rms", "cmd_delta", "cmd_delta_inner", "cmd_delta_boundary",
            "boundary_ratio", "call_ms", "step_ms")

    def means(sel):
        out = {"n_episodes": len(sel)}
        for k in keys:
            v = np.asarray([r[k] for r in sel], dtype=float)
            v = v[np.isfinite(v)]
            out[k] = float(v.mean()) if len(v) else float("nan")
            out[k + "_std"] = float(v.std()) if len(v) else float("nan")
        return out

    return rows, {"all": means(rows), "success": means([r for r in rows if r["success"]])}
