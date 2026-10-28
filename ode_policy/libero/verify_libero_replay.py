"""Replay recorded demo actions in the live env; check conventions and encoding.

This is the strongest policy-independent check on the pipeline. It answers two
separate questions, and keeping them separate is the whole point:

**1. Is the observation encoding identical on both sides?**
   Measured at ``t = 0`` only — right after ``set_init_state`` and before any
   action is applied. At that instant there is no simulation divergence, so any
   pose difference between the live env (``libero_env.build_policy_obs``) and the
   converter (``hdf5_to_zarr``) is a genuine encoding mismatch: a wrong
   quaternion hemisphere rule, a different gripper formula, a swapped field.
   Those are silent at training time and pin eval success near zero.

**2. Does the action convention actually drive the robot?**
   Measured by whether replaying the demo's own actions reproduces its success.
   If it does not, then the delta-vs-absolute choice, the controller mode, the
   init-state handoff, or the success checker is wrong.

The two must not be conflated. Replaying open-loop through MuJoCo does *not*
reproduce a demo exactly — contacts are stiff and error accumulates — so a long
trajectory can drift out of the success region, and its late-trajectory pose
error can be large, while every convention is perfectly correct. That is why the
encoding check reads ``t = 0`` and the trajectory-wide numbers are informational.

Usage:
    python ode_policy/libero/verify_libero_replay.py \
        --raw_dir data_raw/libero/libero_spatial \
        --num_tasks 3 --num_demos 2
"""
from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np

from ode_policy.libero.libero_conventions import (
    canonicalize_quat_sequence,
    gripper_qpos_to_state,
)
from ode_policy.libero.libero_env import build_policy_obs, make_env

STATE_NAMES = ["px", "py", "pz", "qx", "qy", "qz", "qw", "grip"]

# Encoding check, at t=0 where no divergence exists yet. Anything above this is a
# real mismatch between the converter and the live env, not simulation noise.
T0_POSE_TOL = 0.02

# LIBERO's own dataset-regeneration pipeline drops demos that fail replay, so a
# rate — not all-or-nothing — is the right criterion, and the rate that counts as
# healthy depends on the horizon. Measured on this install (n=6..20 per suite):
#   libero_spatial 6/6, libero_object 6/6, libero_goal 5/6, libero_10 14/20 (70%,
#   with failures concentrated in the longer demos: median 343 vs 279 steps).
# Thresholds sit below those so a real convention break (which would collapse the
# rate toward 0) still trips them.
DEFAULT_MIN_RATE = 0.75
SUITE_MIN_RATE = {"libero_10": 0.55, "libero_90": 0.55}


def _min_rate_for(raw_dir: str) -> float:
    name = os.path.basename(os.path.normpath(raw_dir))
    return SUITE_MIN_RATE.get(name, DEFAULT_MIN_RATE)

# The gripper channel is high-gain: state = 1 - 2*width/0.04, so a 2 mm finger
# difference becomes 0.1 in state and a one-frame timing offset at the open/close
# transition momentarily reaches ~0.5. Only its mean is diagnostic of encoding.
GRIP_MEAN_TOL = 0.20


def _demo_state(obs_group, T):
    """Reproduce the converter's (T, 8) state for one demo."""
    from scipy.spatial.transform import Rotation as R

    pos = np.asarray(obs_group["ee_pos"], dtype=np.float32)
    quat = R.from_rotvec(np.asarray(obs_group["ee_ori"])).as_quat().astype(np.float32)
    grip = np.asarray(obs_group["gripper_states"], dtype=np.float32)
    return np.concatenate(
        [pos, canonicalize_quat_sequence(quat), gripper_qpos_to_state(grip)[:, None]],
        axis=-1,
    ).astype(np.float32)[:T]


def _bddl_path(h5file, bddl_root):
    """Rebase the hdf5's recorded bddl path onto this LIBERO install."""
    rel = h5file["data"].attrs["bddl_file_name"]
    if isinstance(rel, bytes):
        rel = rel.decode()
    parts = str(rel).replace("\\", "/").split("/")
    return os.path.join(bddl_root, parts[-2], parts[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--num_tasks", type=int, default=3)
    p.add_argument("--num_demos", type=int, default=2)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--min_rate", type=float, default=None,
                   help="Minimum fraction of demos that must replay successfully. "
                        "Default depends on the suite (long-horizon suites replay "
                        "less reliably); see SUITE_MIN_RATE.")
    args = p.parse_args()
    if args.min_rate is None:
        args.min_rate = _min_rate_for(args.raw_dir)

    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.hdf5")))[: args.num_tasks]
    if not files:
        raise FileNotFoundError(f"No hdf5 under {args.raw_dir}")

    from libero.libero import get_libero_path

    bddl_root = get_libero_path("bddl_files")
    t0_errs, traj_mean, traj_max = [], [], []
    outcomes = []  # (success, length)

    for h5_path in files:
        with h5py.File(h5_path, "r") as f:
            bddl_file = _bddl_path(f, bddl_root)
            if not os.path.exists(bddl_file):
                print(f"[skip] bddl not found: {bddl_file}")
                continue

            env = make_env(bddl_file, args.image_size, args.image_size)
            for dk in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[: args.num_demos]:
                demo = f["data"][dk]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                ref = _demo_state(demo["obs"], len(actions))

                env.reset()
                raw_obs = env.set_init_state(np.asarray(demo.attrs["init_state"]))
                live, prev_q = build_policy_obs(raw_obs, None)

                e0 = np.abs(live["agent_pos"][0] - ref[0])
                t0_errs.append(e0)
                errs = [e0]

                for t, a in enumerate(actions):
                    raw_obs, _r, done, _i = env.step(a)
                    live, prev_q = build_policy_obs(raw_obs, prev_q)
                    if t + 1 < len(ref):
                        errs.append(np.abs(live["agent_pos"][0] - ref[t + 1]))
                    if env.check_success() or done:
                        break

                ok = bool(env.check_success())
                E = np.stack(errs)
                traj_mean.append(E.mean(axis=0))
                traj_max.append(E.max(axis=0))
                outcomes.append((ok, len(actions)))
                print(f"  {os.path.basename(h5_path)[:38]:38s} {dk:8s} success={ok!s:5s} "
                      f"len={len(actions):4d}  t0|pose|={e0[:7].max():.4f}  "
                      f"traj|pose|={E[:, :7].max():.4f}")
            env.close()

    if not outcomes:
        print("nothing replayed")
        return 1

    t0 = np.stack(t0_errs)
    tmean = np.mean(traj_mean, axis=0)
    tmax = np.max(traj_max, axis=0)
    succeeded = sum(int(s) for s, _ in outcomes)
    total = len(outcomes)
    rate = succeeded / total

    print("\nper-channel error (t0 = encoding only; traj = encoding + replay divergence):")
    for i, n in enumerate(STATE_NAMES):
        print(f"    {n:5s} t0_max={t0[:, i].max():.4f}   traj_mean={tmean[i]:.4f}   "
              f"traj_max={tmax[i]:.4f}")

    ok = True

    # --- 1. encoding, judged where no divergence exists yet -------------------
    t0_pose = t0[:, :7].max()
    if t0_pose > T0_POSE_TOL:
        print(f"\n[FAIL] at t=0 the live pose differs from the converter's by {t0_pose:.4f} "
              f"(> {T0_POSE_TOL}). The two sides encode the state differently — check "
              "libero_conventions.py against both of its callers.")
        ok = False
    else:
        print(f"\n[PASS] observation encoding matches at t=0 (max pose diff {t0_pose:.4f}).")

    if tmean[7] > GRIP_MEAN_TOL:
        print(f"[FAIL] gripper differs by {tmean[7]:.4f} on average (> {GRIP_MEAN_TOL}) — "
              "check the gripper convention.")
        ok = False
    else:
        print(f"[PASS] gripper encoding agrees (mean {tmean[7]:.4f}; transient max "
              f"{tmax[7]:.4f} is open/close timing, not a mismatch).")

    # --- 2. action convention, judged by reproduced success -------------------
    fail_len = [n for s, n in outcomes if not s]
    succ_len = [n for s, n in outcomes if s]
    if fail_len:
        print(f"       replayed OK: median length "
              f"{int(np.median(succ_len)) if succ_len else 0}; failed: median length "
              f"{int(np.median(fail_len))}.")
    if rate < args.min_rate:
        print(f"[FAIL] only {succeeded}/{total} demos replayed successfully ({rate:.0%} < "
              f"{args.min_rate:.0%}) — check the action convention and controller mode.")
        ok = False
    else:
        print(f"[PASS] {succeeded}/{total} demos replayed successfully ({rate:.0%}).")

    print("\nRESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
