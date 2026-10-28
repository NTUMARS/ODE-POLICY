"""nav2d data for the history-based baselines (A2A, MARS): low-dim state = last velocity.

A2A and MARS start their flow from the history of actions and, by default, read that history
from `agent_pos` (history_source="agent_pos"), where the state is the last velocity command.
The nav2d zarr stores the agent POSITION as state (for fm_dit and the ODE), so these two
methods get a copy in that convention:

    data/state     = last velocity: the action of the previous step, zeros at an episode start
    data/action    = velocity (unchanged)
    data/agent_pos = agent position (MARS's KNN key, knn_state_key=agent_pos)
    data/head_camera, meta/   symlinked from the source zarr

The evaluator feeds such policies the last EXECUTED velocity (zeros before the first step).

    python ode_policy/nav2d/make_vel_zarr.py     # -> data_policy/nav2d_vel_200.zarr
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import zarr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data_policy/nav2d_pos_200.zarr")
    p.add_argument("--out", default="data_policy/nav2d_vel_200.zarr")
    args = p.parse_args()

    src = zarr.open(args.src, mode="r")
    ends = src["meta/episode_ends"][:]
    pos = src["data/state"][:].astype(np.float32)
    act = src["data/action"][:].astype(np.float32)
    last = np.zeros_like(act)
    last[1:] = act[:-1]
    last[np.r_[0, ends[:-1]]] = 0.0

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    root = zarr.open(args.out, mode="w")
    data = root.create_group("data")
    comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    for k, v in (("state", last), ("action", act), ("agent_pos", pos)):
        data.create_dataset(k, data=v, chunks=(1000, v.shape[1]), compressor=comp)
    os.symlink(os.path.abspath(os.path.join(args.src, "data", "head_camera")),
               os.path.join(args.out, "data", "head_camera"))
    os.symlink(os.path.abspath(os.path.join(args.src, "meta")), os.path.join(args.out, "meta"))
    print(f"{len(ends)} episodes, {len(act)} steps -> {args.out}")


if __name__ == "__main__":
    main()
