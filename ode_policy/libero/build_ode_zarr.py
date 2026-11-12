"""Write a training set whose `action` field is the end-effector pose, for the ODE policy.

The ODE policy learns  da/dt = f(c, s, a)  where `a` is the pose the arm should be at.
It is trained by integrating f from the current pose and matching the integrated path
against the demonstrated one, so it needs the pose itself -- never a finite-difference
velocity, which would inherit the demo's step-to-step jitter.

--grip_ramp 5 (the default) ramps every gripper switch over 5 steps in the ODE copy and writes
<name>_gr5.zarr, the data of the default policy (gripper integrated as an ODE state, grip_ode);
--grip_ramp 1 keeps the raw +-1 command (the paper's classification-head setting, <name>.zarr).

--target virtual (the default, used for every result in the paper) is described below;
--target measured writes the measured pose instead (-> <suite>_50demo_ode.zarr, evaluate
with --gain 3.91). Row t of `action` (7 numbers):
    [0:6]  measured pose at t: 3 position (m) + rotation vector relative to CANONICAL_QUAT
    [6]    the demo's gripper command at t (+1 close / -1 open)

With the stock window (8 history + 8 future, current index 7) a sample therefore holds
the start pose a0 = row 7 and the eight poses the ODE must pass through, rows 8..15.
Pose, not raw OSC deltas, because the pose is smooth and the deltas are not
(position steps compress ~5x better than the commanded deltas), and pose -> OSC is a
verified closed-loop map (see ode_policy/libero/pose.py).

Images are symlinked rather than copied, so this costs megabytes, not gigabytes.

--target virtual writes the VIRTUAL target (-> <suite>_50demo_ode_vt.zarr): the
pose the demonstrator's OSC command pointed at, r = p + DELTA_POS_SCALE * cmd (rotation
R = exp(DELTA_ROT_SCALE * cmd_rot) * R_now, world frame). The measured pose is the
RESULT of a command; in contact (a drawer resisting) the arm moves far less than it was
told to, and that gap is the push. r keeps it, so with the tracker gain at 1,
(r - p) / scale reproduces the demo command exactly. Row t holds the target of the
command issued at t-1, i.e. the one to aim at from the pose measured at t; row 0 of an
episode is its first measured pose.

    python ode_policy/libero/build_ode_zarr.py --suite libero_4suite                 # virtual
    python ode_policy/libero/build_ode_zarr.py --suite libero_object --target measured

--target delta writes the raw OSC command itself (-> <suite>_50demo_ode_delta.zarr),
the action fm_dit / a2a_dit learn: row t = command issued at t-1 (zeros at an episode's
first row), so the ODE integrates from the last command sent and its output is sent to
the controller as is. A control for the virtual target: same information, no pose
anchor.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

import numpy as np
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scipy.spatial.transform import Rotation as R

from ode_policy.libero.pose import pose_signal
from ode_policy.libero.libero_conventions import DELTA_POS_SCALE, DELTA_ROT_SCALE

ACTION_DIM = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--data_root", default="data_policy")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--target", default="virtual", choices=["measured", "virtual", "delta"])
    ap.add_argument("--grip_ramp", type=int, default=5,
                    help="ramp each gripper open/close switch over this many steps in the ODE copy, "
                         "so the gripper is a continuous state for grip_ode (the default policy); "
                         "1 = the raw +-1 command (the paper's classification-head setting)")
    args = ap.parse_args()

    src = os.path.join(args.data_root, f"{args.suite}_50demo.zarr")
    suffix = {"measured": "_ode", "virtual": "_ode_vt", "delta": "_ode_delta"}[args.target]
    if args.grip_ramp > 1:
        suffix += f"_gr{args.grip_ramp}"      # the default policy's data; --grip_ramp 1 = the paper's
    dst = os.path.join(args.data_root, f"{args.suite}_50demo{suffix}.zarr")
    if os.path.exists(dst):
        if not args.force:
            print(f"{dst} exists; pass --force to rebuild")
            return 0
        shutil.rmtree(dst)

    root = zarr.open(src, mode="r")
    state = np.asarray(root["data"]["state"]).astype(np.float64)
    act = np.asarray(root["data"]["action"]).astype(np.float64)
    ends = np.asarray(root["meta"]["episode_ends"])

    if args.target == "virtual":
        cmd = np.clip(act[:, :6], -1.0, 1.0)            # the OSC controller clips the same way
        pos_t = state[:, :3] + DELTA_POS_SCALE * cmd[:, :3]
        rot_t = R.from_rotvec(DELTA_ROT_SCALE * cmd[:, 3:6]) * R.from_quat(state[:, 3:7])
        vt = pose_signal(pos_t, rot_t.as_quat())          # target of the command issued at t
        sig = pose_signal(state[:, :3], state[:, 3:7])
        starts0 = np.concatenate([[0], ends[:-1]])
        shifted = np.empty_like(sig)
        for a, b in zip(starts0, ends):                    # row t <- command issued at t-1
            shifted[a] = sig[a]
            shifted[a + 1:b] = vt[a:b - 1]
        sig = shifted
    elif args.target == "delta":
        cmd = np.clip(act[:, :6], -1.0, 1.0)
        sig = np.empty_like(cmd)
        for a, b in zip(np.concatenate([[0], ends[:-1]]), ends):   # row t <- command at t-1
            sig[a] = 0.0
            sig[a + 1:b] = cmd[a:b - 1]
    else:
        sig = pose_signal(state[:, :3], state[:, 3:7])
    grip = np.sign(act[:, 6])
    grip[grip == 0] = 1.0
    if args.grip_ramp > 1:
        # each switch becomes a linear ramp over grip_ramp steps starting at the switch;
        # episodes start open (-1). The command the policy sends stays discrete.
        n = args.grip_ramp
        for a, b in zip(np.concatenate([[0], ends[:-1]]), ends):
            c = grip[a:b].copy()
            for k in np.flatnonzero(c[1:] != c[:-1]) + 1:
                m = min(n, len(c) - k)
                grip[a + k:a + k + m] = c[k - 1] + (c[k] - c[k - 1]) * np.arange(1, m + 1) / n
            grip[a] = -1.0
    acts = np.concatenate([sig, grip[:, None]], axis=1).astype(np.float32)

    os.makedirs(os.path.join(dst, "data"), exist_ok=True)
    os.makedirs(os.path.join(dst, "meta"), exist_ok=True)
    for grp, name in [("data", "head_camera"), ("data", "wrist_camera"),
                      ("data", "state"), ("data", "task_id"),
                      ("meta", "episode_ends"), ("meta", "task_emb")]:
        s = os.path.join(src, grp, name)
        if os.path.exists(s):
            os.symlink(os.path.abspath(s), os.path.join(dst, grp, name))
    for f in (".zgroup", ".zattrs"):
        for base in ("", "data", "meta"):
            s = os.path.join(src, base, f)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(dst, base, f))

    out = zarr.open(dst, mode="a")
    out["data"].create_dataset("action", data=acts, chunks=(4096, ACTION_DIM),
                               dtype="float32", overwrite=True)
    out.attrs["grip_ramp"] = int(args.grip_ramp)

    # per-step motion, within episodes only -- sets the scale of f's output
    starts = np.concatenate([[0], ends[:-1]])
    step = np.concatenate([np.diff(sig[a:b], axis=0) for a, b in zip(starts, ends)])
    print(f"wrote {dst}  ({len(acts)} steps, {len(ends)} episodes)")
    print(f"  pose range min {np.round(sig.min(0), 3)}")
    print(f"             max {np.round(sig.max(0), 3)}")
    print(f"  |step| p50/p99 per axis: {np.round(np.percentile(np.abs(step), 50, 0), 4)}"
          f" / {np.round(np.percentile(np.abs(step), 99, 0), 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
