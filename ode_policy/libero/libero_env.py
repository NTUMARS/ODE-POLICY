"""Thin wrapper around LIBERO's OffScreenRenderEnv.

Every convention here must stay in lock-step with ``hdf5_to_zarr.py`` — a
mismatch between how demos were encoded and how live observations are built is
silent (no crash, no warning) and shows up only as a flat 0% success rate.

Invariants
----------
* Env runs OSC_POSE at its default ``control_delta=True``. ``env.step`` takes a
  7-dim delta command ``[d_pos(3), d_axis_angle(3), gripper_cmd(1)]`` in the
  controller's normalized [-1, 1] range (scaled to 5cm / 0.5rad max per step).
* ``agent_pos`` = (1, 8) float32 ``[pos(3), quat(4 xyzw, hemisphere-continuous), gripper(1)]``
* policy action = (7,) float32, same layout as ``env.step``'s input.
* Gripper scalar, quaternion sign, and the image flip all come from
  ``libero_conventions.py``, which the offline converter imports too — that
  shared module is what keeps train and eval encodings from drifting apart.

Why delta rather than absolute targets: robosuite 1.4.1's ``control_delta=False``
mode drifts (~20mm/step in -Y even when the commanded goal equals the current
pose). Delta mode is also how LIBERO's own demos were collected, so train and
test share one action space.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, Optional, Tuple

import numpy as np

from ode_policy.libero.libero_conventions import (
    align_quat,
    gripper_qpos_to_state,
    image_to_nchw_uint8,
)


def _egl_device_works(idx: int) -> bool:
    """Can mujoco initialize an EGL display on this device index?

    Probed in a subprocess because mujoco decides at *import* time and raises on
    failure — there is no way to retry inside one interpreter.
    """
    env = dict(os.environ, MUJOCO_EGL_DEVICE_ID=str(idx), PYOPENGL_PLATFORM="egl")
    try:
        return subprocess.run(
            [sys.executable, "-c", "import mujoco.egl"],
            env=env, capture_output=True, timeout=120,
        ).returncode == 0
    except Exception:
        return False


def _configure_egl_device() -> None:
    """Pick a working EGL render device when the scheduler hands out GPU UUIDs.

    robosuite's ``create_initialized_egl_device_display`` reads
    ``CUDA_VISIBLE_DEVICES`` and calls ``int()`` on it. Some schedulers set that variable to
    a GPU *UUID* (``GPU-6497402e-...``), so the first env build dies with
    ``ValueError: invalid literal for int()``. The failure only shows up in batch
    jobs: interactively the variable is usually unset, so the same code path runs
    clean and the bug stays hidden until a compute node picks it up.

    ``MUJOCO_EGL_DEVICE_ID`` wins over ``CUDA_VISIBLE_DEVICES`` in that lookup, so
    setting it sidesteps the parse — but it must name a device that actually
    initializes. Note the EGL device list is NOT the CUDA device list: it can
    include entries that no GPU backs, and mujoco's own default is to try every
    one until a display comes up. Setting the variable disables that search, so
    we do the search ourselves and pin the index that won.

    ``CUDA_VISIBLE_DEVICES`` is deliberately left alone; torch keeps the GPU the
    scheduler assigned.
    """
    if os.environ.get("MUJOCO_EGL_DEVICE_ID"):
        return
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd or cvd.split(",")[0].isdigit():
        return  # unset or numeric: robosuite parses it fine, leave mujoco's search alone

    for idx in range(8):
        if _egl_device_works(idx):
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(idx)
            # robosuite ALSO asserts `MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`
            # (binding_utils.py) — a plain substring test, which is meaningless
            # against a UUID. It passes only when the UUID's hex happens to
            # contain that digit, so jobs succeeded or failed by luck of the draw:
            # of four GPUs handed out here, three UUIDs contained "0" and the
            # fourth did not. Rewrite the variable to the same index so both
            # checks agree.
            #
            # Safe at this point: torch selected its device when the policy was
            # loaded, long before make_env runs, and CUDA_VISIBLE_DEVICES is only
            # read at process start. Only robosuite, importing now, sees this.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            print(f"[libero-env] EGL device {idx}; rewrote CUDA_VISIBLE_DEVICES "
                  f"{cvd} -> {idx} for robosuite (torch already holds its GPU)")
            return
    raise RuntimeError(
        f"No EGL device could be initialized (CUDA_VISIBLE_DEVICES={cvd}). "
        "Headless rendering is unavailable on this node — check that MUJOCO_GL=egl "
        "and that the driver exposes the EGL PLATFORM_DEVICE extension."
    )


def make_env(bddl_file: str, camera_heights: int = 128, camera_widths: int = 128,
             with_wrist: bool = False, control_freq: int = 20):
    """Instantiate a LIBERO off-screen env with the default OSC_POSE controller
    (control_freq: control steps per second; LIBERO's demos are 20 Hz)."""
    _configure_egl_device()
    from libero.libero.envs import OffScreenRenderEnv

    kwargs = dict(
        bddl_file_name=bddl_file,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
    )
    if control_freq != 20:
        # the episode limit is in control steps: keep the same 50 s as at 20 Hz
        kwargs.update(control_freq=control_freq, horizon=1000 * control_freq // 20)
    if with_wrist:
        # Order matters only in that both names must be present; LIBERO's demos
        # were collected with exactly this pair.
        kwargs["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    return OffScreenRenderEnv(**kwargs)


def build_policy_obs(
    raw_obs: Dict[str, np.ndarray],
    prev_state_quat: Optional[np.ndarray] = None,
    with_wrist: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Convert one raw env observation into the policy-facing dict.

    Returns ``(policy_obs, canon_quat)``. Feed ``canon_quat`` back as
    ``prev_state_quat`` on the next call so the quaternion never flips sign
    mid-episode — the converter enforces the same continuity over demos, and a
    flip would put the live state far outside the range the normalizer was fit on.
    """
    img = image_to_nchw_uint8(raw_obs["agentview_image"])
    pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    quat = align_quat(raw_obs["robot0_eef_quat"], prev_state_quat)
    grip = float(gripper_qpos_to_state(
        np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    ))

    agent_pos = np.concatenate([pos, quat, np.array([grip], dtype=np.float32)], axis=-1)
    policy_obs = {
        "head_cam": img[None],                             # (1, 3, H, W) uint8
        "agent_pos": agent_pos[None].astype(np.float32),   # (1, 8) float32
    }
    if with_wrist:
        # robosuite names the eye-in-hand view after the robot; the hdf5 dumps
        # call the same camera `eye_in_hand_rgb`.
        wrist_raw = raw_obs.get("robot0_eye_in_hand_image")
        if wrist_raw is None:
            raise KeyError(
                "Policy expects a wrist_cam observation but the env produced no "
                "'robot0_eye_in_hand_image'. Build the env with that camera enabled "
                "(make_env(..., with_wrist=True))."
            )
        policy_obs["wrist_cam"] = image_to_nchw_uint8(wrist_raw)[None]
    return policy_obs, quat


def policy_action_to_env_action(
    policy_action: np.ndarray,
    current_state_quat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Near-identity passthrough for the 7-dim delta policy output.

    The policy is trained on LIBERO's raw demo actions, so its output is already
    in the env's command space. Only the gripper channel is clamped — flow/
    diffusion samplers can overshoot past ±1 and robosuite does not clip it.
    """
    del current_state_quat  # kept for call-site compatibility with the absolute variant
    assert policy_action.shape[-1] == 7, f"expected 7-dim delta action, got {policy_action.shape}"
    action = policy_action.astype(np.float32).copy()
    action[6] = float(np.clip(action[6], -1.0, 1.0))
    return action
