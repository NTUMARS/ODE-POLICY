"""Evaluate a checkpoint on ManiSkill3 tasks: success rate plus execution quality.

The same loop for every method (CPU simulator, one env): n_obs_steps of history (the first frame repeated
at the start), one chunk of n_action_steps executed open loop, then observe again. Actions are
pd_joint_pos targets [7 arm joints, gripper].
  * ODE policies: the integrated path is sent unchanged; each chunk integrates on from where the last one
    ended, the first from the measured joints.
  * other policies: the first n_action_steps actions of the predicted chunk.
Test episodes use seeds 100000 + i (the demonstrations use seeds below 10000), at most MAX_STEPS[task]
control steps (about twice the median demo length); an episode succeeds the first time the env reports
success.

Execution quality: jerk = RMS norm of the third difference of the measured TCP position (m/s^3, 20 Hz,
successful episodes), call_ms = wall time of one predict_action call (CUDA-synchronised).

    python ode_policy/maniskill/eval_maniskill.py --ckpt il_outputs/ode/ms_8task/checkpoints/50.ckpt
    # -> <ckpt dir>/evalms_ep50.json (per-task results and their mean)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import zarr

sys.path.insert(0, os.getcwd())
from ode_policy.nav2d.eval_nav2d import load_policy  # noqa: E402

TEST_SEED, DT = 100000, 0.05
MAX_STEPS = {"PushCube-v1": 150, "PickCube-v1": 160, "StackCube-v1": 250, "PullCube-v1": 150,
             "LiftPegUpright-v1": 350, "PlaceSphere-v1": 230, "PullCubeTool-v1": 500, "StackPyramid-v1": 400,
             "PegInsertionSide-v1": 300, "PlugCharger-v1": 400}


def tcp_state(env):
    """Measured TCP position in the base frame, gripper opening and arm qpos (7)."""
    u = env.unwrapped
    pose = u.agent.robot.pose.inv() * u.agent.tcp.pose
    qpos = u.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float64)
    return pose.p[0].cpu().numpy().astype(np.float64), float(qpos[-2:].sum()), qpos[:7]


def rollout(env, policy, seed, n_obs, n_act, device, max_steps, task_vec):
    """task_vec: multi-task models also observe [goal in the base frame (3), task one-hot]."""
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed)                  # sampling noise (flow matching, diffusion) is the same per episode
    root = env.unwrapped.agent.robot.pose.p[0].cpu().numpy()
    if hasattr(policy, "reset"):
        policy.reset()                       # ODE: the gripper state starts from the open state
    is_ode = hasattr(policy, "field")
    kw = {"eps_center": True} if is_ode and getattr(policy, "side_k", 0) else {}

    def frame(o):
        p, g, q = tcp_state(env)
        img = o["sensor_data"]["base_camera"]["rgb"][0].permute(2, 0, 1).float() / 255.0
        st = np.r_[q, g]
        if task_vec is not None:
            goal = o["extra"]["goal_pos"][0].cpu().numpy() - root if "goal_pos" in o["extra"] else np.zeros(3)
            st = np.r_[st, goal, task_vec]
        return {"head_cam": img, "agent_pos": torch.tensor(st, dtype=torch.float32)}, p

    f0, p0 = frame(obs)
    hist, tcp, times, cache = [f0] * n_obs, [p0], [], []
    a0 = f0["agent_pos"][None, :7].to(device) if is_ode else None
    succ, steps = False, 0
    for steps in range(1, max_steps + 1):
        if not cache:
            od = {k: torch.stack([h[k] for h in hist[-n_obs:]])[None].to(device) for k in hist[-1]}
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = (policy.predict_action(od, a0=a0, **kw) if is_ode else policy.predict_action(od))["action"][0, :n_act]
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - t0))
            if is_ode:
                a0 = out[-1:, :7].clone()
            cache = list(out.cpu().numpy().astype(np.float64))
        a = cache.pop(0)
        if not np.isfinite(a).all():           # the integrated path ran away: the episode fails
            break
        obs, _, _, _, info = env.step(np.r_[a[:7], np.clip(a[7], -1, 1)].astype(np.float32))
        f, p = frame(obs)
        hist.append(f); tcp.append(p)
        if bool(info["success"][0]):
            succ = True
            break
    j = np.diff(np.array(tcp), n=3, axis=0) / DT ** 3
    jerk = float(np.sqrt(np.mean(np.sum(j ** 2, axis=1)))) if len(j) else float("nan")
    return succ, steps, jerk, times


def eval_task(task, policy, n_obs, n_act, device, n_test, task_vec):
    env = gym.make(task, obs_mode="rgb", control_mode="pd_joint_pos", render_mode=None, sim_backend="physx_cpu",
                   reward_mode="sparse", max_episode_steps=MAX_STEPS[task] + 10)
    rows = []
    for i in range(n_test):
        succ, steps, jerk, times = rollout(env, policy, TEST_SEED + i, n_obs, n_act, device, MAX_STEPS[task], task_vec)
        # the first call of the first episode includes CUDA warm-up
        rows.append(dict(seed=TEST_SEED + i, success=succ, steps=steps, jerk=jerk,
                         call_ms=float(np.mean(times[1:] if i == 0 and len(times) > 1 else times))))
    env.close()
    ok = [r for r in rows if r["success"]]
    res = {"task": task, "success": len(ok) / len(rows),
           "steps_succ": float(np.mean([r["steps"] for r in ok])) if ok else None,
           "jerk_succ": float(np.nanmean([r["jerk"] for r in ok])) if ok else None,
           "call_ms": float(np.mean([r["call_ms"] for r in rows])), "__episodes__": rows}
    print(f"  {task}: success {res['success']:.2f}  jerk {res['jerk_succ']}  call {res['call_ms']:.1f} ms", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default=None, help="single-task models: the task; multi-task: optional, one task only")
    p.add_argument("--n_test", type=int, default=100, help="test episodes per task")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, n_obs, n_act = load_policy(args.ckpt, device)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False, pickle_module=__import__("dill"))
    attrs = zarr.open(str(payload["cfg"].dataset_config.zarr_path), mode="r").attrs
    tasks = list(attrs.get("tasks", [])) or [attrs.get("task", args.task)]
    todo = [args.task] if args.task and args.task in tasks else tasks
    per = {}
    for t in todo:
        vec = np.eye(len(tasks))[tasks.index(t)] if attrs.get("tasks") else None
        per[t] = eval_task(t, policy, n_obs, n_act, device, args.n_test, vec)
    keys = ("success", "steps_succ", "jerk_succ", "call_ms")
    mean = {k: float(np.nanmean([np.nan if per[t][k] is None else per[t][k] for t in todo])) for k in keys}
    mean["success_std_tasks"] = float(np.std([per[t]["success"] for t in todo]))
    res = {"ckpt": args.ckpt, "tasks": todo, "n_test": args.n_test, "mean": mean, "per_task": per}
    ep = os.path.splitext(os.path.basename(args.ckpt))[0]
    tag = "_" + todo[0].split("-")[0] if len(todo) == 1 and len(tasks) > 1 else ""
    out = os.path.join(os.path.dirname(args.ckpt), f"evalms_ep{ep}{tag}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"{args.ckpt}: success {mean['success']:.3f} (" + " ".join(f"{t.split('-')[0]} {per[t]['success']:.2f}" for t in todo)
          + f")  jerk {mean['jerk_succ']:.1f}  call {mean['call_ms']:.1f} ms -> {out}", flush=True)


if __name__ == "__main__":
    main()
