"""robomimic proficient-human demos -> a LIBERO-format zarr, images rendered by replaying the states.

robomimic's low-dim datasets carry no images; as its dataset_states_to_obs.py, this sets the simulator to
every recorded state and renders. Every frame is encoded with the LIBERO conventions
(libero_env.build_policy_obs), so training data and live observations go through one function.

    data/head_camera, data/wrist_camera  (N, 3, S, S) uint8   agentview, robot0_eye_in_hand
    data/state   (N, 8)  [eef pos 3, eef quat 4 (xyzw), gripper opening 1]
    data/action  (N, 7)  the demo OSC command (delta pos 3, delta rot 3, gripper +1 close / -1 open)
    data/task_id (N,), meta/episode_ends, meta/task_emb (1, 512) zeros (merge_zarr.py adds CLIP embeddings)

    python ode_policy/robomimic/make_zarr.py --task square --out data_policy/square_ph240.zarr
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import h5py
import numpy as np
import zarr

sys.path.insert(0, os.getcwd())
from ode_policy.libero.libero_env import build_policy_obs  # noqa: E402
from ode_policy.robomimic.robomimic_env import TASKS, RobomimicEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None, help="first n demos (default all 200)")
    ap.add_argument("--image_size", type=int, default=240)
    args = ap.parse_args()
    hdf5 = TASKS[args.task][0]

    f = h5py.File(hdf5, "r")
    demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[:args.n]
    env = RobomimicEnv(hdf5, image_size=args.image_size, with_wrist=True)
    env.reset()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    root = zarr.open(args.out, mode="w")
    S = args.image_size
    comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    head = root.create_dataset("data/head_camera", shape=(0, 3, S, S), chunks=(64, 3, S, S), dtype="u1", compressor=comp)
    wrist = root.create_dataset("data/wrist_camera", shape=(0, 3, S, S), chunks=(64, 3, S, S), dtype="u1", compressor=comp)
    states, actions, ends, n_tot, t0 = [], [], [], 0, time.time()
    for i, k in enumerate(demos):
        d = f["data"][k]
        st, act = d["states"][:], d["actions"][:].astype(np.float32)
        hs, ws, prev_q = [], [], None
        for t in range(len(st)):
            po, prev_q = build_policy_obs(env.set_init_state(st[t]), prev_q, with_wrist=True)
            hs.append(po["head_cam"][0]); ws.append(po["wrist_cam"][0]); states.append(po["agent_pos"][0])
        head.append(np.stack(hs)); wrist.append(np.stack(ws)); actions.append(act)
        n_tot += len(st)
        ends.append(n_tot)
        if i % 10 == 0 or i == len(demos) - 1:
            print(f"{i + 1}/{len(demos)} demos, {n_tot} steps, {time.time() - t0:.0f}s", flush=True)
    env.close()
    root.create_dataset("data/state", data=np.stack(states).astype(np.float32), chunks=(4096, 8))
    root.create_dataset("data/action", data=np.concatenate(actions), chunks=(4096, 7))
    root.create_dataset("data/task_id", data=np.zeros(n_tot, np.int64), chunks=(4096,))
    root.create_dataset("meta/episode_ends", data=np.array(ends, np.int64))
    root.create_dataset("meta/task_emb", data=np.zeros((1, 512), np.float32))
    root.attrs.update(task_names=[args.task], task_instructions=[TASKS[args.task][2]], source=hdf5)
    print(f"wrote {args.out}: {len(demos)} demos, {n_tot} steps")


if __name__ == "__main__":
    main()
