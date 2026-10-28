"""Data-efficiency training sets (paper experiment E3): the first K demonstrations of every task.

  data_policy/libero_4suite_<K>demo.zarr          same layout as libero_4suite_50demo.zarr (images copied)
  data_policy/libero_4suite_<K>demo_ode_vt.zarr   ODE virtual-target variant, what
                                                  `build_ode_zarr.py --target virtual` makes from the K-demo zarr:
                                                  everything symlinked except `action`

"First K" = the K lowest episode indices of each task_id in libero_4suite_50demo.zarr, i.e. the order hdf5_to_zarr.py
wrote the demos. The ODE action is computed with build_ode_zarr.py's virtual-target formula and checked against the
matching rows of libero_4suite_50demo_ode_vt.zarr (the formula only looks within an episode, so they must be identical).

    python ode_policy/libero/build_kdemo_zarr.py 5 10 20 30
"""
import argparse
import os
import pathlib
import shutil
import sys

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ode_policy.libero.libero_conventions import DELTA_POS_SCALE, DELTA_ROT_SCALE  # noqa: E402
from ode_policy.libero.pose import pose_signal  # noqa: E402

DATA = "data_policy"
SRC = f"{DATA}/libero_4suite_50demo.zarr"
SRC_VT = f"{DATA}/libero_4suite_50demo_ode_vt.zarr"
BLOCK = 2000   # output rows per image copy batch (a multiple of the 100-row chunks: aligned writes)


def virtual_target(state, act, ends):
    """build_ode_zarr.py --target virtual, verbatim: row t = target of the command issued at t-1."""
    state, act = state.astype(np.float64), act.astype(np.float64)
    cmd = np.clip(act[:, :6], -1.0, 1.0)
    pos_t = state[:, :3] + DELTA_POS_SCALE * cmd[:, :3]
    rot_t = R.from_rotvec(DELTA_ROT_SCALE * cmd[:, 3:6]) * R.from_quat(state[:, 3:7])
    vt = pose_signal(pos_t, rot_t.as_quat())
    sig = pose_signal(state[:, :3], state[:, 3:7])
    shifted = np.empty_like(sig)
    for a, b in zip(np.concatenate([[0], ends[:-1]]), ends):
        shifted[a] = sig[a]
        shifted[a + 1:b] = vt[a:b - 1]
    grip = np.sign(act[:, 6])
    grip[grip == 0] = 1.0
    return np.concatenate([shifted, grip[:, None]], axis=1).astype(np.float32)


def build(k):
    dst, dst_vt = f"{DATA}/libero_4suite_{k}demo.zarr", f"{DATA}/libero_4suite_{k}demo_ode_vt.zarr"
    if os.path.exists(dst) and os.path.exists(dst_vt):
        print(f"{dst} exists, skipped")
        return
    src = zarr.open(SRC, mode="r")
    ends = np.asarray(src["meta"]["episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    ep_task = np.asarray(src["data"]["task_id"])[starts]
    keep = np.concatenate([np.flatnonzero(ep_task == t)[:k] for t in np.unique(ep_task)])
    keep.sort()
    assert len(keep) == k * len(np.unique(ep_task)), "fewer than K demos for some task"
    rows = np.concatenate([np.arange(starts[i], ends[i]) for i in keep])
    new_ends = np.cumsum(ends[keep] - starts[keep])

    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    out = zarr.open(tmp, mode="w")
    out.attrs.update(dict(src.attrs, demos_per_task=k,
                          derived_from="libero_4suite_50demo.zarr, first K episodes of every task_id"))
    data, meta = out.create_group("data"), out.create_group("meta")
    meta.create_dataset("task_emb", data=np.asarray(src["meta"]["task_emb"]), chunks=src["meta"]["task_emb"].chunks,
                        dtype="float32")
    meta.create_dataset("episode_ends", data=new_ends, chunks=(10000,), dtype="int64")
    for name in ("task_id", "state", "action"):
        a = src["data"][name]
        data.create_dataset(name, data=np.asarray(a)[rows], chunks=a.chunks, dtype=a.dtype, compressor=a.compressor)
    for cam in ("head_camera", "wrist_camera"):
        if cam not in src["data"]:
            continue
        a = src["data"][cam]
        arr = data.create_dataset(cam, shape=(len(rows), *a.shape[1:]), chunks=a.chunks, dtype=a.dtype,
                                  compressor=a.compressor)
        for p in range(0, len(rows), BLOCK):
            idx = rows[p:p + BLOCK]
            cuts = np.flatnonzero(np.diff(idx) != 1) + 1         # contiguous source runs inside this block
            arr[p:p + len(idx)] = np.concatenate([a[r[0]:r[-1] + 1] for r in np.split(idx, cuts)])
        print(f"  K={k}: {cam} copied ({len(rows)} rows)", flush=True)
    shutil.rmtree(dst, ignore_errors=True)
    os.rename(tmp, dst)

    # ODE virtual-target variant (as build_ode_zarr.py: symlinks + rewritten action)
    base = zarr.open(dst, mode="r")
    acts = virtual_target(np.asarray(base["data"]["state"]), np.asarray(base["data"]["action"]), new_ends)
    ref = np.asarray(zarr.open(SRC_VT, mode="r")["data"]["action"])[rows]
    assert np.array_equal(acts, ref), f"virtual target differs from the 50-demo rows (max {np.abs(acts - ref).max()})"
    tmp = dst_vt + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(f"{tmp}/data")
    os.makedirs(f"{tmp}/meta")
    for grp, name in [("data", "head_camera"), ("data", "wrist_camera"), ("data", "state"), ("data", "task_id"),
                      ("meta", "episode_ends"), ("meta", "task_emb")]:
        if os.path.exists(f"{dst}/{grp}/{name}"):
            os.symlink(os.path.abspath(f"{dst}/{grp}/{name}"), f"{tmp}/{grp}/{name}")
    for f in (".zgroup", ".zattrs"):
        for base in ("", "data", "meta"):
            if os.path.exists(os.path.join(dst, base, f)):
                shutil.copy(os.path.join(dst, base, f), os.path.join(tmp, base, f))
    zarr.open(tmp, mode="a")["data"].create_dataset("action", data=acts, chunks=(4096, 7), dtype="float32",
                                                    overwrite=True)
    shutil.rmtree(dst_vt, ignore_errors=True)
    os.rename(tmp, dst_vt)
    print(f"wrote {dst} and {dst_vt}: {len(keep)} episodes ({k} per task), {len(rows)} steps", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int, nargs="+", help="demonstrations per task")
    ap.add_argument("--data_root", default="data_policy")
    args = ap.parse_args()
    DATA = args.data_root
    SRC, SRC_VT = f"{DATA}/libero_4suite_50demo.zarr", f"{DATA}/libero_4suite_50demo_ode_vt.zarr"
    for k in args.k:
        build(k)
