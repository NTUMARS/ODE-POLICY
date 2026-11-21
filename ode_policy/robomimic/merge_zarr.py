"""Merge single-task robomimic zarrs (make_zarr.py, same image size) into one multi-task zarr.

Episodes are concatenated in the given order; data/task_id holds the task index per step and meta/task_emb
the CLIP text embedding of each task's instruction, so the policies are conditioned on the task as on LIBERO.

    python ode_policy/robomimic/merge_zarr.py --out data_policy/rm4_ph.zarr --src tool_hang=data_policy/tool_hang_ph240.zarr \
        square=data_policy/square_ph240.zarr can=data_policy/can_ph240.zarr lift=data_policy/lift_ph240.zarr
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np
import zarr

sys.path.insert(0, os.getcwd())
from ode_policy.libero.text_encoder import encode_instructions  # noqa: E402
from ode_policy.robomimic.robomimic_env import TASKS  # noqa: E402

TEXT_MODEL = "openai/clip-vit-base-patch32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="task=zarr")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    specs = [s.split("=", 1) for s in args.src]
    srcs = [zarr.open(p, mode="r") for _, p in specs]
    S = srcs[0]["data/head_camera"].shape[-1]
    assert all(z["data/head_camera"].shape[-1] == S for z in srcs), "all sources need the same image size"

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    root = zarr.open(args.out, mode="w")
    comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    cams = {c: root.create_dataset(f"data/{c}", shape=(0, 3, S, S), chunks=(64, 3, S, S), dtype="u1", compressor=comp)
            for c in ("head_camera", "wrist_camera")}
    states, actions, task_id, ends, n_tot, t0 = [], [], [], [], 0, time.time()
    for k, ((task, path), z) in enumerate(zip(specs, srcs)):
        n = int(z["meta/episode_ends"][-1])
        for c, arr in cams.items():
            for a in range(0, n, 4096):
                arr.append(z[f"data/{c}"][a:min(a + 4096, n)])
        states.append(z["data/state"][:]); actions.append(z["data/action"][:])
        task_id.append(np.full(n, k, np.int64)); ends.append(z["meta/episode_ends"][:] + n_tot)
        n_tot += n
        print(f"{task}: {len(ends[-1])} demos, {n} steps ({time.time() - t0:.0f}s)", flush=True)

    names = [t for t, _ in specs]
    instr = [TASKS[t][2] for t in names]
    root.create_dataset("data/state", data=np.concatenate(states).astype(np.float32), chunks=(4096, 8))
    root.create_dataset("data/action", data=np.concatenate(actions).astype(np.float32), chunks=(4096, 7))
    root.create_dataset("data/task_id", data=np.concatenate(task_id), chunks=(4096,))
    root.create_dataset("meta/episode_ends", data=np.concatenate(ends))
    root.create_dataset("meta/task_emb", data=encode_instructions(instr, model_name=TEXT_MODEL).astype(np.float32))
    root.attrs.update(task_names=names, task_instructions=instr, text_model=TEXT_MODEL)
    print(f"wrote {args.out}: {len(names)} tasks, {sum(len(e) for e in ends)} demos, {n_tot} steps")


if __name__ == "__main__":
    main()
