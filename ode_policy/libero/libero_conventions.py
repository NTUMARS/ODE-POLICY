"""The single source of truth for LIBERO state encoding.

Both the offline converter (``hdf5_to_zarr.py``) and the live env wrapper
(``libero_env.py``) must encode observations *identically*. When they drift, the
policy sees one distribution in training and another at eval — no crash, no
warning, just a success rate pinned near zero. Keeping the arithmetic in one
module is what structurally prevents that drift, so put any new convention here
rather than in either caller.

Gripper
-------
LIBERO's ``gripper_states`` / robosuite's ``robot0_gripper_qpos`` hold the two
finger joint positions as a **mirrored pair** — ``[+x, -x]``, each finger roughly
in ``[0.002, 0.040]`` m. Their signed mean is therefore ~0 regardless of how open
the gripper is; a converter that averages them produces a channel containing
nothing but residual noise. The informative quantity is the finger *separation*,
so we average the **magnitudes**. That also stays correct for dumps that store
both fingers positive.

Quaternion hemisphere
---------------------
``q`` and ``-q`` are the same rotation, so a sign convention is needed or the
normalizer sees one physical pose as two far-apart clusters. The obvious rule —
"make ``w >= 0``" — fails badly on LIBERO: the gripper points down, i.e. a ~180°
rotation, so ``w ≈ 0`` (measured range ±0.08) and the sign of ``w`` is
effectively a coin flip. Episodes then land in opposite hemispheres and the eef
``x`` channel spans ±1 for what is one and the same orientation.

Instead we fix the sign by the component with the **largest magnitude** (``x``,
≈0.999, for LIBERO's home pose). That is stable across episodes and across the
train/eval boundary, and needs no stored reference. Within an episode the sign
then propagates by dot-product continuity.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Panda finger travel, per finger, in meters.
GRIPPER_MAX_WIDTH = 0.04


def gripper_qpos_to_state(qpos: np.ndarray) -> np.ndarray:
    """Finger joint positions -> opening scalar in [-1, +1]; +1 closed, -1 open.

    Accepts ``(T, F)`` or ``(F,)``. Uses mean |qpos| over the fingers, which is
    the per-finger opening for both the mirrored ``[+x, -x]`` layout LIBERO uses
    and the both-positive layout some dumps use.

    The sign matches the demo's gripper *command* channel (+1 close / -1 open) so
    state and action move together.
    """
    q = np.abs(np.asarray(qpos, dtype=np.float32))
    single = q.ndim == 1
    if single:
        q = q[None]
    width = q[:, :2].mean(axis=1) if q.shape[1] >= 2 else q[:, 0]
    normed = np.clip(width / GRIPPER_MAX_WIDTH, 0.0, 1.0)  # 0=closed, 1=open
    state = (1.0 - 2.0 * normed).astype(np.float32)
    return state[0] if single else state


def canonical_quat_sign(q: np.ndarray) -> np.ndarray:
    """Fix a single xyzw quaternion's sign by its largest-magnitude component.

    Stable for LIBERO, where ``w ≈ 0`` makes the usual ``w >= 0`` rule arbitrary.
    """
    q = np.asarray(q, dtype=np.float32).reshape(4).copy()
    if q[int(np.argmax(np.abs(q)))] < 0:
        q = -q
    return q


def align_quat(q_now: np.ndarray, q_prev: Optional[np.ndarray]) -> np.ndarray:
    """Put ``q_now`` in the same hemisphere as ``q_prev``.

    With no ``q_prev`` (episode start) fall back to :func:`canonical_quat_sign`,
    which is exactly what the converter does for each demo's first frame.
    """
    q = np.asarray(q_now, dtype=np.float32).reshape(4).copy()
    if q_prev is None:
        return canonical_quat_sign(q)
    if float(np.dot(q, np.asarray(q_prev, dtype=np.float32).reshape(4))) < 0.0:
        q = -q
    return q


def canonicalize_quat_sequence(quat_seq: np.ndarray) -> np.ndarray:
    """Sign-canonicalize a ``(T, 4)`` xyzw sequence: anchor frame 0, then continue."""
    q = np.asarray(quat_seq, dtype=np.float32).copy()
    q[0] = canonical_quat_sign(q[0])
    for t in range(1, q.shape[0]):
        if float(np.dot(q[t], q[t - 1])) < 0.0:
            q[t] *= -1.0
    return q


def image_to_nchw_uint8(rgb_hwc: np.ndarray) -> np.ndarray:
    """Undo LIBERO's OpenGL vertical flip; ``(H, W, 3)`` -> ``(3, H, W)`` uint8."""
    return np.ascontiguousarray(np.moveaxis(rgb_hwc[::-1, :, :], -1, 0).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Measured delta
# --------------------------------------------------------------------------- #
# LIBERO's OSC_POSE controller maps a normalized command in [-1, 1] onto these
# physical increments per control step (env_args.controller_configs.output_max):
#     position  ±0.05 m      orientation  ±0.5 rad
# Dividing a *measured* pose increment by the same scales expresses it in the
# controller's own units, so measured deltas and recorded action commands live in
# one space and share the action normalizer.
DELTA_POS_SCALE = 0.05
DELTA_ROT_SCALE = 0.5

# Which frame the orientation increment is expressed in.
#   "world" : q_t = dq ⊗ q_{t-1}   (left-multiply; robosuite applies OSC rotation
#                                   increments about the base frame)
#   "body"  : q_t = q_{t-1} ⊗ dq   (right-multiply)
# Determined empirically against the recorded commands — see
# verify_libero_delta.py, which reports the correlation for both.
DELTA_ROT_FRAME = "world"


def _quat_mul(a, b):
    """Hamilton product of xyzw quaternions; a, b are (..., 4) numpy arrays."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def _quat_conj(q):
    out = np.asarray(q, dtype=np.float32).copy()
    out[..., :3] *= -1.0
    return out


def quat_to_axis_angle(q):
    """xyzw unit quaternion (..., 4) -> axis-angle vector (..., 3), angle in [0, pi]."""
    q = np.asarray(q, dtype=np.float32)
    # q and -q are the same rotation; pick w >= 0 so the angle is the short way round.
    q = np.where(q[..., 3:4] < 0, -q, q)
    v = q[..., :3]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(n, q[..., 3:4])
    return (angle * v / np.maximum(n, 1e-8)).astype(np.float32)


def measured_delta_from_poses(pos, quat, grip=None, frame=DELTA_ROT_FRAME):
    """Per-frame *measured* motion, in the controller's normalized action units.

    ``md[i]`` describes the motion that carried the arm from frame ``i-1`` to
    frame ``i`` — i.e. what actually happened, as opposed to what was commanded.
    ``md[0]`` has no predecessor and is replicated from ``md[1]``.

    Args:
        pos:  (T, 3) end-effector positions, meters.
        quat: (T, 4) end-effector orientations, xyzw, hemisphere-continuous.
        grip: (T,) optional gripper scalar; fills the last channel so the result
              is action-shaped. Policies that treat the gripper as a noise dim
              overwrite it anyway.
        frame: "world" or "body", see DELTA_ROT_FRAME.

    Returns:
        (T, 7) float32 ``[dpos(3)/0.05, drot_axis_angle(3)/0.5, gripper(1)]``.
    """
    pos = np.asarray(pos, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    T = pos.shape[0]

    dpos = np.zeros_like(pos)
    dpos[1:] = pos[1:] - pos[:-1]

    if frame == "world":
        dq = _quat_mul(quat[1:], _quat_conj(quat[:-1]))
    elif frame == "body":
        dq = _quat_mul(_quat_conj(quat[:-1]), quat[1:])
    else:
        raise ValueError(f"frame must be 'world' or 'body', got {frame!r}")
    drot = np.zeros_like(pos)
    drot[1:] = quat_to_axis_angle(dq)

    md = np.zeros((T, 7), dtype=np.float32)
    md[:, 0:3] = dpos / DELTA_POS_SCALE
    md[:, 3:6] = drot / DELTA_ROT_SCALE
    if grip is not None:
        md[:, 6] = np.asarray(grip, dtype=np.float32).reshape(-1)
    if T > 1:
        md[0] = md[1]  # no predecessor for frame 0
    return md


def measured_delta_from_poses_torch(agent_pos, frame=DELTA_ROT_FRAME):
    """Batched torch version of :func:`measured_delta_from_poses`.

    Args:
        agent_pos: (B, T, 8) raw ``[pos(3), quat(4 xyzw), gripper(1)]``.
        frame: see DELTA_ROT_FRAME.

    Returns:
        (B, T, 7) float32, same units and ``md[0] = md[1]`` convention as the
        numpy version. The last channel carries the gripper *state*; policies that
        treat the gripper as a noise dim overwrite it.
    """
    import torch

    pos = agent_pos[..., 0:3]
    quat = agent_pos[..., 3:7]
    grip = agent_pos[..., 7:8]

    dpos = torch.zeros_like(pos)
    dpos[:, 1:] = pos[:, 1:] - pos[:, :-1]

    a = quat[:, 1:] if frame == "world" else quat[:, :-1]
    b = quat[:, :-1] if frame == "world" else quat[:, 1:]
    if frame == "world":
        b = torch.cat([-b[..., :3], b[..., 3:]], dim=-1)   # conj(q_{t-1})
    elif frame == "body":
        a = torch.cat([-a[..., :3], a[..., 3:]], dim=-1)   # conj(q_{t-1})
    else:
        raise ValueError(f"frame must be 'world' or 'body', got {frame!r}")

    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    dq = torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)

    # Short way round: q and -q are the same rotation.
    dq = torch.where(dq[..., 3:4] < 0, -dq, dq)
    v = dq[..., :3]
    n = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(n, dq[..., 3:4])
    drot = torch.zeros_like(pos)
    drot[:, 1:] = angle * v / n.clamp(min=1e-8)

    md = torch.cat([dpos / DELTA_POS_SCALE, drot / DELTA_ROT_SCALE, grip], dim=-1)
    if md.shape[1] > 1:
        md = torch.cat([md[:, 1:2], md[:, 1:]], dim=1)  # frame 0 has no predecessor
    return md
