"""End-effector pose representation and the pose -> OSC tracking law used by the ODE policy.

Pose signal: 3 position (m) + rotation vector of the orientation RELATIVE to a fixed
reference. The absolute orientation sits near 180 deg (gripper pointing down), where the
rotation vector is singular and wraps; relative to CANONICAL_QUAT the whole swing is
~17 deg median and smooth. The reference is fixed (not each episode's first frame) so the
training target and the deployed policy agree across episodes.

Quaternions are (x, y, z, w), as LIBERO stores them. Rotation commands are in the world
frame.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from ode_policy.libero.libero_conventions import DELTA_POS_SCALE, DELTA_ROT_SCALE

DT = 0.05  # LIBERO runs at 20 Hz

# mean demo orientation; also hard-coded in ode_policy/policies/ode/ode_image_policy.py
CANONICAL_QUAT = np.array([0.99959104, -0.00150985, -0.02851977, 0.00144439])
POSE_REF = R.from_quat(CANONICAL_QUAT)

# Pose error -> OSC command gain, as a multiple of the naive 1/DELTA_*_SCALE.
# For the VIRTUAL target (build_ode_zarr.py --target virtual, the default) the gain is 1:
# (r - p) / DELTA_*_SCALE reproduces the demo command exactly.
TRACK_GAIN = 1.0
# For the MEASURED pose (--target measured) the arm only partly reaches the commanded
# target within a step, so the pose it should LAND on must be commanded harder. Fit on 80
# libero_object demos: commanded action = 78.1 x achieved position step (R^2 0.945), i.e.
# 3.91x; rotation gives the same factor (R^2 0.833) independently.
MEASURED_TRACK_GAIN = 3.91


def pose_signal(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """(T,3) position, (T,4) xyzw quaternion -> (T,6) pose signal."""
    rel = (R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)) * POSE_REF.inv()).as_rotvec()
    return np.concatenate([np.asarray(pos, dtype=np.float64), rel], axis=1)


def live_pose(raw_obs):
    return (np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64),
            R.from_quat(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float64)))


def track_action(p_ref, R_ref, p_now, R_now, grip, gain=TRACK_GAIN):
    """One closed-loop OSC command that aims the end effector at (p_ref, R_ref)."""
    ep = np.asarray(p_ref, dtype=np.float64) - np.asarray(p_now, dtype=np.float64)
    er = (R_ref * R_now.inv()).as_rotvec()
    dpos = ep / DELTA_POS_SCALE * gain
    drot = er / DELTA_ROT_SCALE * gain
    return np.clip(np.concatenate([dpos, drot, [grip]]), -1.0, 1.0).astype(np.float32)
