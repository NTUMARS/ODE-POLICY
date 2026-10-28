"""Observation and action adapters between the R1 Lite client's 14-DoF format and the policy.

The client's vectors are always 14-D [L1..L6, L_grip, R1..R6, R_grip] with three cameras
(cam_head, cam_left_wrist, cam_right_wrist). What the policy reads depends on what it was
trained on (its shape_meta):

  * single arm (state_dim 7): agent_pos = the right half [R1..R6, R_grip], cameras
    head_cam + wrist_cam; its 7-D action rows are expanded to 14-D by echoing the client's
    current left arm + left gripper (the left arm holds still);
  * two arms (state_dim 14): agent_pos = the whole 14-D vector (the training zarr uses the
    wire order), cameras head_cam + wrist_cam + left_wrist_cam; its 14-D action rows are
    sent as they are.

wrist_cam is the RIGHT wrist (the name predates the two-arm tasks).
"""

from __future__ import annotations

import base64
import logging
from collections import deque
from typing import Any, Sequence

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

WIRE_DIM = 14
# policy camera obs key -> the client's image key
CLIENT_CAMERAS = {"head_cam": "cam_head", "wrist_cam": "cam_right_wrist", "left_wrist_cam": "cam_left_wrist"}


def policy_state_slice(state_dim: int) -> slice:
    """The part of the client's 14-D state a policy reads, and its action rows write."""
    if state_dim == 7:
        return slice(7, 14)     # right arm + right gripper (single-arm tasks)
    if state_dim == WIRE_DIM:
        return slice(0, 14)     # both arms: the training layout is the wire order
    raise ValueError(f"agent_pos must be 7-D (right arm) or 14-D (both arms), got {state_dim}")


class ActionAdapter:
    """(N, 7) right-arm or (N, 14) two-arm action chunk -> the (N, 14) format the R1 Lite
    client expects: [L1..L6, L_grip, R1..R6, R_grip].

    Pure function — no internal state.
    """

    @staticmethod
    def adapt(
        action: np.ndarray,
        left_echo: np.ndarray,
        n_action_steps: int,
    ) -> np.ndarray:
        if action.shape[0] != n_action_steps:
            raise ValueError(
                f"action has {action.shape[0]} rows, expected {n_action_steps}"
            )
        if action.ndim == 2 and action.shape[1] == WIRE_DIM:
            return action.astype(np.float32, copy=True)          # two arms: already the wire layout
        if action.shape[1] != 7:
            raise ValueError(f"action must have 7 or 14 cols, got {action.shape[1]}")
        if left_echo.shape != (7,):
            raise ValueError(f"left_echo must be shape (7,), got {left_echo.shape}")

        out = np.zeros((n_action_steps, 14), dtype=np.float32)
        out[:, 0:7] = left_echo.astype(np.float32)[None, :]   # broadcast L1..L6 + L_grip
        out[:, 7:13] = action[:, 0:6].astype(np.float32)      # R1..R6
        out[:, 13] = action[:, 6].astype(np.float32)          # R_grip
        return out


class ObservationAdapter:
    """Convert a single client observation dict into the obs_dict shape that the
    policy's `predict_action` expects, while maintaining the rolling `n_obs_steps`
    history buffer.

    state_dim / camera_keys: what the policy was trained on (7 + head_cam, wrist_cam for a
    single arm; 14 + head_cam, wrist_cam, left_wrist_cam for two arms). Only the cameras
    the policy reads are required in a request.

    State machine:
      * `reset()` clears the history deque
      * first call to `adapt()` after `reset()` fills the deque by padding
        the same frame forward (matches `DefaultEvalRunner._stack_last_n_obs`)
      * subsequent calls push exactly one new frame; deque drops the oldest
    """

    def __init__(
        self,
        n_obs_steps: int,
        image_size: int,
        device: torch.device,
        state_dim: int = 7,
        camera_keys: Sequence[str] = ("head_cam", "wrist_cam"),
    ) -> None:
        self.n_obs_steps = n_obs_steps
        self.image_size = image_size
        self.device = device
        self.state_slice = policy_state_slice(state_dim)
        unknown = [k for k in camera_keys if k not in CLIENT_CAMERAS]
        if unknown:
            raise ValueError(f"policy cameras {unknown} have no client image (known: {CLIENT_CAMERAS})")
        self.camera_keys = tuple(camera_keys)
        # Each entry is ({camera key: chw}, agent_pos).
        self.buf: deque = deque(maxlen=n_obs_steps)
        self.warmed: bool = False
        self._prompt_logged: bool = False

    def reset(self) -> None:
        self.buf.clear()
        self.warmed = False
        self._prompt_logged = False

    def _decode(self, b64_png: str) -> np.ndarray:
        """base64 PNG → float32 CHW RGB in [0, 1], resized to image_size."""
        try:
            raw = base64.b64decode(b64_png)
        except Exception as e:
            raise ValueError(f"image is not valid base64: {e}") from e
        arr = np.frombuffer(raw, dtype=np.uint8)
        try:
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR uint8 or None
        except cv2.error as e:
            raise ValueError(f"cv2.imdecode failed: {e}") from e
        if img is None:
            raise ValueError("cv2.imdecode failed: bytes are not a valid PNG/JPEG")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(
                img,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return np.ascontiguousarray(img)

    def adapt(self, msg: dict[str, Any]) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        # 1. Validate state ─────────────────────────────────────────────
        state = msg.get("state")
        if state is None:
            raise ValueError("request is missing 'state' field")
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.shape != (14,):
            raise ValueError(f"state must have shape (14,), got {state_arr.shape}")

        # 2. Validate images ────────────────────────────────────────────
        images = msg.get("images") or {}
        for key in self.camera_keys:
            if CLIENT_CAMERAS[key] not in images:
                raise ValueError(f"request is missing images.{CLIENT_CAMERAS[key]}")

        # 3. Log prompt once if present ─────────────────────────────────
        prompt = msg.get("prompt")
        if prompt and not self._prompt_logged:
            log.info(
                "received prompt=%r; this server does not use language conditioning, ignoring.",
                prompt,
            )
            self._prompt_logged = True

        # 4. Extract proprio: left echo (7) + the policy's agent_pos ────
        left_echo = state_arr[0:7].copy()                          # L1..L6 + L_grip
        agent_pos = state_arr[self.state_slice].copy()             # (7,) right arm or (14,) both

        # 5. Decode images ──────────────────────────────────────────────
        frames = {key: self._decode(images[CLIENT_CAMERAS[key]]) for key in self.camera_keys}

        # 6. Push into history; pad-forward on first frame ──────────────
        entry = (frames, agent_pos)
        # Pad-forward shares the same entry across all slots; safe because
        # the arrays inside are never mutated after this point (we only
        # np.stack them on read).
        if not self.warmed:
            for _ in range(self.n_obs_steps):
                self.buf.append(entry)
            self.warmed = True
        else:
            self.buf.append(entry)

        # 7. Stack into tensors and move to device ──────────────────────
        obs_dict = {
            key: torch.from_numpy(np.stack([e[0][key] for e in self.buf], axis=0))   # (T,3,H,W)
            .unsqueeze(0).to(self.device, non_blocking=True)
            for key in self.camera_keys
        }
        obs_dict["agent_pos"] = torch.from_numpy(np.stack([e[1] for e in self.buf], axis=0)).unsqueeze(0).to(
            self.device, non_blocking=True)                         # (1,T,D)
        return obs_dict, left_echo


def dummy_obs_dict(inference, image_size: int = 256, state: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    """Random camera frames + agent_pos shaped like ObservationAdapter's output for `inference`
    (its camera_keys / state_dim / n_obs_steps / device) -- for smoke tests and latency checks."""
    dev, T = inference.device, inference.n_obs_steps
    obs = {k: torch.rand(1, T, 3, image_size, image_size, device=dev)
           for k in getattr(inference, "camera_keys", ("head_cam", "wrist_cam"))}
    d = getattr(inference, "state_dim", 7)
    ap = torch.zeros(1, T, d, device=dev) if state is None else \
        torch.as_tensor(np.asarray(state, dtype=np.float32), device=dev).expand(1, T, d).clone()
    obs["agent_pos"] = ap
    return obs
