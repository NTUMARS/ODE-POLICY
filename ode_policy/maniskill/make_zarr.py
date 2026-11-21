"""Render ManiSkill3 motion-planning demos (gen_demos.py) into this repo's zarr layout, in joint space.

Each demo is replayed from its first state with the same controller (pd_joint_pos) to render the 128 px
base camera; demos that no longer succeed are dropped. Two zarrs share the images and meta:

    <out>      state  = [arm qpos (7), gripper opening (1)]
               action = [arm joint targets issued at t (7), gripper command (+1 open / -1 closed)]
               goal   = goal position in the robot base frame (3; 0 for tasks without one)
               (fm_dit and the other baselines)
    <out>_ode  action = [joint targets in force at t, i.e. issued at t-1 (7; row 0 = the measured qpos),
                         gripper state: every switch ramped over --grip_ramp steps, row 0 = the reset state]
               (the ODE policy integrates from row 0 through the next targets; the gripper is an ODE state)

    python ode_policy/maniskill/make_zarr.py --task PickCube-v1   # -> data_policy/ms_PickCube.zarr{,_ode}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import zarr
from mani_skill.trajectory import utils as trajectory_utils

SRC = "data_policy/raw/maniskill/{task}/motionplanning/trajectory"


def reset_to(env, ep, h5traj):
    """As ManiSkill's replay tool with --use-first-env-state."""
    kw = dict(ep.get("reset_kwargs", {}))
    seed = kw.get("seed", ep["episode_seed"])
    kw["seed"] = seed[0] if isinstance(seed, list) else seed
    env.reset(**kw)
    env.unwrapped.set_state_dict(trajectory_utils.dict_to_list_of_dicts(h5traj["env_states"])[0])


def replay(env, ep, tr):
    """Re-run one demo; per step the image, qpos and goal before the step, and whether it succeeded."""
    reset_to(env, ep, tr)
    u = env.unwrapped
    obs = u.get_obs()
    root = u.agent.robot.pose.p[0].cpu().numpy()
    acts = tr["actions"][:]
    imgs, qpos, goal, succ = [], [], [], False
    for a in acts:
        imgs.append(obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy())
        qpos.append(u.agent.robot.get_qpos()[0].cpu().numpy())
        goal.append(obs["extra"]["goal_pos"][0].cpu().numpy() - root if "goal_pos" in obs["extra"] else np.zeros(3))
        obs, _, _, _, info = env.step(a)
        succ = succ or bool(info["success"][0])
    return np.moveaxis(np.array(imgs), -1, 1), np.array(qpos), np.array(goal), acts.astype(np.float32), succ


def ramp(g, n, start):
    """Gripper state for the ODE: each switch of the command g becomes a linear ramp over n steps."""
    r = g.copy()
    for k in np.flatnonzero(g[1:] != g[:-1]) + 1:
        m = min(n, len(g) - k)
        r[k:k + m] = g[k - 1] + (g[k] - g[k - 1]) * np.arange(1, m + 1) / n
    r[0] = start
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=200, help="episodes to keep (the first that replay successfully)")
    ap.add_argument("--grip_ramp", type=int, default=5)
    ap.add_argument("--src", default=SRC, help="demo path without .h5/.json; {task} is filled in")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"data_policy/ms_{args.task.split('-')[0]}.zarr"
    src = args.src.format(task=args.task)
    meta = json.load(open(src + ".json"))
    assert meta["env_info"]["env_kwargs"]["control_mode"] == "pd_joint_pos", meta["env_info"]["env_kwargs"]
    eps = {f"traj_{e['episode_id']}": e for e in meta["episodes"]}
    f = h5py.File(src + ".h5", "r")
    keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
    env = gym.make(args.task, obs_mode="rgb", control_mode="pd_joint_pos", render_mode=None,
                   sim_backend="physx_cpu", reward_mode="sparse")
    img, state, act, ode, goal, ends, seeds, tried = [], [], [], [], [], [], [], 0
    for k in keys:
        if len(ends) == args.n:
            break
        tried += 1
        im, q, g, a, succ = replay(env, eps[k], f[k])
        if not succ:
            print(f"{k}: replay did not succeed, skipped", flush=True)
            continue
        arm, grip = q[:, :7], q[:, 7:9].sum(1, keepdims=True)
        img.append(im); goal.append(g); state.append(np.c_[arm, grip]); act.append(a)
        # gripper row 0 = the reset state (open / closed from the measured opening), as at deployment
        gs = ramp(a[:, 7].copy(), args.grip_ramp, 1.0 if grip[0, 0] > 0.04 else -1.0)
        ode.append(np.c_[np.concatenate([arm[:1], a[:-1, :7]]), gs])
        ends.append(len(a)); seeds.append(eps[k]["episode_seed"])
    env.close()
    assert ends, "no demo replayed successfully"
    img, ends = np.concatenate(img), np.cumsum(ends)
    state, act, ode, goal = (np.concatenate(x).astype(np.float32) for x in (state, act, ode, goal))
    print(f"{len(ends)}/{tried} episodes kept, {len(act)} steps", flush=True)

    comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    out_ode = out.rstrip("/") + "_ode"
    for p in (out, out_ode):
        if os.path.exists(p):
            shutil.rmtree(p)
    root = zarr.open(out, mode="w")
    d, m = root.create_group("data"), root.create_group("meta")
    d.create_dataset("head_camera", data=img, chunks=(100, *img.shape[1:]), compressor=comp)
    d.create_dataset("state", data=state, chunks=(1000, state.shape[1]), compressor=comp)
    d.create_dataset("action", data=act, chunks=(1000, act.shape[1]), compressor=comp)
    d.create_dataset("goal", data=goal, chunks=(1000, 3), compressor=comp)
    m.create_dataset("episode_ends", data=ends.astype(np.int64), compressor=None)
    m.create_dataset("episode_seed", data=np.array(seeds, np.int64), compressor=None)
    root.attrs.update(task=args.task, control="joint", pose_dim=7, grip_ramp=args.grip_ramp)
    oroot = zarr.open(out_ode, mode="w")
    od = oroot.create_group("data")
    od.create_dataset("state", data=state, chunks=(1000, state.shape[1]), compressor=comp)
    od.create_dataset("action", data=ode, chunks=(1000, ode.shape[1]), compressor=comp)
    os.symlink(os.path.abspath(os.path.join(out, "data", "head_camera")), os.path.join(out_ode, "data", "head_camera"))
    os.symlink(os.path.abspath(os.path.join(out, "meta")), os.path.join(out_ode, "meta"))
    oroot.attrs.update(root.attrs.asdict())
    print(f"-> {out}, {out_ode}", flush=True)


if __name__ == "__main__":
    main()
