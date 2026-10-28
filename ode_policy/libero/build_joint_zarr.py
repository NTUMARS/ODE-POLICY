"""Concatenate the four LIBERO suites into one training set, `libero_4suite_50demo.zarr`.

Same layout as the per-suite zarrs from hdf5_to_zarr.py, so every policy trains on it
unchanged: data/{head_camera, wrist_camera, state, action, task_id}, meta/{episode_ends,
task_emb} and the root attrs. task_id is offset per suite (object 0-9, spatial 10-19,
goal 20-29, libero_10 30-39) and meta/task_emb is stacked in that order, so
task_emb[task_id] still picks the right instruction embedding. Evaluation looks the
embedding up by instruction text, which the concatenated attrs carry for all 40 tasks.

Images are copied chunk by chunk (about 13 GB on disk); the ODE variant is then
    python ode_policy/libero/build_ode_zarr.py --suite libero_4suite
which symlinks the images and rewrites only `action`.

    python ode_policy/libero/build_joint_zarr.py
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import zarr

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
COPY_ROWS = 2000   # steps per copy batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_policy")
    ap.add_argument("--out", default="libero_4suite_50demo.zarr")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dst = os.path.join(args.data_root, args.out)
    if os.path.exists(dst):
        if not args.force:
            print(f"{dst} exists; pass --force to rebuild")
            return 0
        shutil.rmtree(dst)

    srcs = [zarr.open(os.path.join(args.data_root, f"{s}_50demo.zarr"), mode="r") for s in SUITES]
    out = zarr.open(dst, mode="w")
    data, meta = out.create_group("data"), out.create_group("meta")

    # root attrs: the per-task lists are concatenated, the rest copied from the first suite
    attrs = dict(srcs[0].attrs)
    attrs["suite"] = "libero_4suite"
    attrs["task_names"] = sum((list(z.attrs["task_names"]) for z in srcs), [])
    attrs["task_instructions"] = sum((list(z.attrs["task_instructions"]) for z in srcs), [])
    attrs["source_suites"] = SUITES
    out.attrs.update(attrs)

    n_task = [z["meta"]["task_emb"].shape[0] for z in srcs]
    task_offset = np.concatenate([[0], np.cumsum(n_task)[:-1]])
    n_step = [z["data"]["head_camera"].shape[0] for z in srcs]
    step_offset = np.concatenate([[0], np.cumsum(n_step)[:-1]])

    meta.create_dataset("task_emb", data=np.concatenate([np.asarray(z["meta"]["task_emb"]) for z in srcs]),
                        chunks=(sum(n_task), srcs[0]["meta"]["task_emb"].shape[1]), dtype="float32")
    meta.create_dataset("episode_ends", data=np.concatenate(
        [np.asarray(z["meta"]["episode_ends"]) + off for z, off in zip(srcs, step_offset)]),
        chunks=(10000,), dtype="int64")

    data.create_dataset("task_id", data=np.concatenate(
        [np.asarray(z["data"]["task_id"]) + off for z, off in zip(srcs, task_offset)]),
        chunks=srcs[0]["data"]["task_id"].chunks, dtype="int64")
    for name in ("state", "action"):
        data.create_dataset(name, data=np.concatenate([np.asarray(z["data"][name]) for z in srcs]),
                            chunks=srcs[0]["data"][name].chunks, dtype="float32")

    total = sum(n_step)
    for cam in ("head_camera", "wrist_camera"):
        if not all(cam in z["data"] for z in srcs):
            print(f"skip {cam}: not in every suite")
            continue
        a0 = srcs[0]["data"][cam]
        arr = data.create_dataset(cam, shape=(total, *a0.shape[1:]), chunks=a0.chunks,
                                  dtype=a0.dtype, compressor=a0.compressor)
        pos = 0
        for s, z in zip(SUITES, srcs):
            src = z["data"][cam]
            for i in range(0, src.shape[0], COPY_ROWS):
                blk = src[i:i + COPY_ROWS]
                arr[pos:pos + len(blk)] = blk
                pos += len(blk)
            print(f"  {cam}: {s} copied ({pos}/{total})", flush=True)
    print(f"wrote {dst}: {total} steps, {len(meta['episode_ends'])} episodes, "
          f"{sum(n_task)} tasks, suites {SUITES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
