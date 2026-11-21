"""Merge per-task ManiSkill3 zarrs (make_zarr.py) into one multi-task training set.

One model per method is trained on all tasks and tested on each (as on LIBERO). The task identity
enters as an ordinary low-dim observation:
    state  = [arm qpos (7), gripper opening (1), goal in the base frame (3), task one-hot (K)]
    action = as in make_zarr.py (stock and _ode copies)

    python ode_policy/maniskill/build_multitask.py     # the 8 tasks -> data_policy/ms_8task.zarr{,_ode}
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import zarr

TASKS = ["PushCube-v1", "PickCube-v1", "StackCube-v1", "PullCube-v1", "LiftPegUpright-v1",
         "PlaceSphere-v1", "PullCubeTool-v1", "StackPyramid-v1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--n_per_task", type=int, default=200)
    ap.add_argument("--out", default="data_policy/ms_8task.zarr")
    args = ap.parse_args()

    parts = []
    for t in args.tasks:
        z = zarr.open(f"data_policy/ms_{t.split('-')[0]}.zarr", mode="r")
        zo = zarr.open(f"data_policy/ms_{t.split('-')[0]}.zarr_ode", mode="r")
        ends = z["meta/episode_ends"][:]
        n = min(args.n_per_task, len(ends))
        stop = int(ends[n - 1])
        parts.append(dict(task=t, z=z, stop=stop, ends=ends[:n], state=z["data/state"][:stop],
                          act=z["data/action"][:stop], ode=zo["data/action"][:stop], goal=z["data/goal"][:stop],
                          seeds=z["meta/episode_seed"][:n]))
        print(f"{t}: {n} episodes, {stop} steps", flush=True)

    K = len(parts)
    state, act, ode, ends, seeds, task_id, off = [], [], [], [], [], [], 0
    for k, p in enumerate(parts):
        oh = np.zeros((p["stop"], K), np.float32); oh[:, k] = 1
        state.append(np.c_[p["state"], p["goal"], oh])
        act.append(p["act"]); ode.append(p["ode"])
        ends.append(p["ends"] + off); off += p["stop"]
        seeds.append(p["seeds"]); task_id.append(np.full(p["stop"], k, np.int64))
    state, act, ode = (np.concatenate(x).astype(np.float32) for x in (state, act, ode))
    ends, seeds, task_id = np.concatenate(ends), np.concatenate(seeds), np.concatenate(task_id)
    print(f"merged: {len(ends)} episodes, {len(act)} steps, state dim {state.shape[1]}", flush=True)

    comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    out_ode = args.out.rstrip("/") + "_ode"
    for p_ in (args.out, out_ode):
        if os.path.exists(p_):
            shutil.rmtree(p_)
    root = zarr.open(args.out, mode="w")
    d, m = root.create_group("data"), root.create_group("meta")
    H, W = parts[0]["z"]["data/head_camera"].shape[2:]
    img = d.create_dataset("head_camera", shape=(len(act), 3, H, W), dtype=np.uint8, chunks=(100, 3, H, W),
                           compressor=comp)
    o = 0
    for p in parts:                                   # copy the images task by task, in blocks
        for a in range(0, p["stop"], 2000):
            b = min(a + 2000, p["stop"])
            img[o + a:o + b] = p["z"]["data/head_camera"][a:b]
        o += p["stop"]
    d.create_dataset("state", data=state, chunks=(1000, state.shape[1]), compressor=comp)
    d.create_dataset("action", data=act, chunks=(1000, act.shape[1]), compressor=comp)
    d.create_dataset("task_id", data=task_id, chunks=(1000,), compressor=comp)
    m.create_dataset("episode_ends", data=ends.astype(np.int64), compressor=None)
    m.create_dataset("episode_seed", data=seeds.astype(np.int64), compressor=None)
    root.attrs.update(tasks=[p["task"] for p in parts], control="joint", pose_dim=7)
    oroot = zarr.open(out_ode, mode="w")
    od = oroot.create_group("data")
    od.create_dataset("state", data=state, chunks=(1000, state.shape[1]), compressor=comp)
    od.create_dataset("action", data=ode, chunks=(1000, ode.shape[1]), compressor=comp)
    os.symlink(os.path.abspath(os.path.join(args.out, "data", "head_camera")), os.path.join(out_ode, "data", "head_camera"))
    os.symlink(os.path.abspath(os.path.join(args.out, "meta")), os.path.join(out_ode, "meta"))
    oroot.attrs.update(root.attrs.asdict())
    print(f"-> {args.out}, {out_ode}", flush=True)


if __name__ == "__main__":
    main()
