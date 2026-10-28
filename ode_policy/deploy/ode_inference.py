"""Serve the ODE policy the way it was trained: observe every tick, re-integrate every
`replan_every` ticks from the carried state, execute the integrated chunk in order.

The robot client (`main_r1lite_closedloop.py`) queries the server at 15 Hz and executes
row 0 of every (8, 14) reply. The ODE policy, however, must run CHUNKED (docs/PORTING.md
"keep aligned": integrate 8 steps per observation, execute all 8, carry the end state into the next
integration; every-step replanning is "not aligned" and loses >10 points on LIBERO).
Both are satisfied without touching the client:

  * every request still updates the 8-frame observation history (it evolves at the
    training fps, as in the LIBERO evaluator where obs_deque is appended every env step);
  * on ticks 0, 8, 16, ... the ODE integrates a fresh chunk of `n_action_steps` targets
    from a_state (the last target sent; the live joints on the first chunk of an episode);
  * on every tick the reply is the cached chunk shifted so that row 0 is THIS tick's
    pre-planned target (the tail is held constant), so the client's "execute row 0"
    walks through the chunk exactly as eval_ode_policy.py does with --replan_every 8
    --carry_state.

Gripper: the policy's gripper head outputs +1 (close) / -1 (open); the R1 Lite wants a
width command in mm, so the sign is mapped to --gripper_close / --gripper_open.

Layout: the policy's rows are [joints(P), grippers(G)]; the reply rows are in the layout of
agent_pos, the slice of the client's 14-D state the policy reads -- single arm [R1..R6,
R_grip] (P=6, G=1, the right half of the wire vector), two arms [L1..L6, L_grip, R1..R6,
R_grip] (P=12, G=2, the whole wire vector). The joints go to the policy's state_pose_index
columns, the grippers to the remaining ones, in order.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch

from ode_policy.deploy.policy_inference import load_checkpoint_policy, policy_io

PerGripper = Union[float, Sequence[float]]


def _per_gripper(v: PerGripper, n: int, name: str) -> np.ndarray:
    """A scalar or one value per gripper -> (n,) float64."""
    a = np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()
    if a.size == 1:
        return np.repeat(a, n)
    if a.size != n:
        raise ValueError(f"{name}: expected 1 or {n} values (one per gripper), got {a.size}")
    return a

log = logging.getLogger(__name__)


class ODEChunkedInference:
    """Same interface as `PolicyInference` (n_obs_steps / n_action_steps / horizon / device /
    reset_episode / infer), plus the chunk schedule and the carried state."""

    def __init__(self, policy, n_obs_steps: int, n_action_steps: int, horizon: int, device: torch.device,
                 replan_every: Optional[int] = None, gripper_close_cmd: PerGripper = 0.0,
                 gripper_open_cmd: PerGripper = 100.0, gripper_ramp_ticks: PerGripper = 0, carry_state: bool = True,
                 camera_keys: Sequence[str] = ("head_cam", "wrist_cam")) -> None:
        self.policy = policy
        self.device = torch.device(device)
        self.n_obs_steps, self.n_action_steps, self.horizon = int(n_obs_steps), int(n_action_steps), int(horizon)
        self.pose_dim = int(getattr(policy, "pose_dim"))
        self.n_grippers = int(getattr(policy, "n_grippers", 1))
        # agent_pos / reply layout: the joints at the policy's live-pose columns, the grippers in the rest
        self.state_dim = self.pose_dim + self.n_grippers
        idx = getattr(policy, "state_pose_index", None)
        self.pose_cols = list(range(self.pose_dim)) if idx is None else [int(i) for i in idx]
        self.grip_cols = [c for c in range(self.state_dim) if c not in self.pose_cols]
        if len(self.pose_cols) != self.pose_dim or len(self.grip_cols) != self.n_grippers:
            raise ValueError(f"state_pose_index {self.pose_cols} does not leave {self.n_grippers} gripper columns "
                             f"in a {self.state_dim}-D state")
        self.camera_keys = tuple(camera_keys)
        self.replan_every = int(self.n_action_steps if replan_every is None else replan_every)
        if not 1 <= self.replan_every <= self.n_action_steps:
            raise ValueError(f"replan_every must be in [1, {self.n_action_steps}], got {self.replan_every}")
        G = self.n_grippers
        self.gripper_close_cmd = _per_gripper(gripper_close_cmd, G, "gripper_close_cmd")
        self.gripper_open_cmd = _per_gripper(gripper_open_cmd, G, "gripper_open_cmd")
        # The demos' gripper command is a teleop RAMP between the plateaus (~20 frames on the insert-flower
        # data); the head only says open/close. With gripper_ramp_ticks > 0 the command sent moves toward
        # the head's target by |open - close| / ticks per tick, from the measured width at the first tick,
        # which reproduces the demos' gripper timing. 0 = step straight to the target. All three gripper
        # settings take one value for every gripper or one per gripper (left first).
        self.gripper_ramp_ticks = _per_gripper(gripper_ramp_ticks, G, "gripper_ramp_ticks").astype(int)
        if np.any(self.gripper_ramp_ticks < 0):
            raise ValueError("gripper_ramp_ticks must be >= 0")
        self.carry_state = bool(carry_state)
        self.policy_label = type(policy).__name__
        self.reset_episode()
        log.info("ODE serving: replan_every=%d carry_state=%s gripper close=%s open=%s ramp_ticks=%s pose_dim=%d "
                 "n_grippers=%d joint cols=%s gripper cols=%s cameras=%s", self.replan_every, self.carry_state,
                 self.gripper_close_cmd.tolist(), self.gripper_open_cmd.tolist(), self.gripper_ramp_ticks.tolist(),
                 self.pose_dim, G, self.pose_cols, self.grip_cols, list(self.camera_keys))

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda:0", **kwargs) -> "ODEChunkedInference":
        policy, cfg = load_checkpoint_policy(checkpoint_path, device)
        if not (hasattr(policy, "integrate") and hasattr(policy, "pose_dim")):
            raise TypeError(f"{type(policy).__name__} is not an ODE policy; serve it with PolicyInference")
        if not getattr(policy, "joint_space", False):
            log.warning("ODE policy was trained with joint_space=False: its live pose is an end-effector "
                        "pose signal, not the R1 Lite joint state. Check the checkpoint.")
        state_dim, camera_keys = policy_io(cfg)
        inf = cls(policy, int(cfg.n_obs_steps), int(cfg.n_action_steps), int(cfg.horizon),
                  torch.device(device), camera_keys=camera_keys, **kwargs)
        if state_dim != inf.state_dim:
            raise ValueError(f"checkpoint agent_pos is {state_dim}-D, the ODE layout is {inf.state_dim}-D "
                             f"(pose_dim {inf.pose_dim} + {inf.n_grippers} grippers)")
        return inf

    # -- episode state --------------------------------------------------------------
    def reset_episode(self) -> None:
        """New connection = new episode: forget the chunk and the carried state."""
        self._chunk: Optional[np.ndarray] = None   # (n_action_steps, pose_dim + G), raw units, grippers +-1
        self._k = 0                                 # rows of the chunk already handed out
        self._a_state: Optional[np.ndarray] = None  # (pose_dim,) target to integrate from next
        self._grip_cmd: Optional[np.ndarray] = None  # (G,) last gripper commands sent (ramp state)
        self.n_replans = 0
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    # -- one request ------------------------------------------------------------------
    @torch.no_grad()
    def infer(self, obs_dict: dict[str, torch.Tensor]) -> dict:
        replanned = False
        if self._chunk is None or self._k >= self.replan_every:
            a0 = None
            if self.carry_state and self._a_state is not None:
                a0 = torch.as_tensor(self._a_state, dtype=torch.float32, device=self.device)[None]
            kw = {}
            if getattr(self.policy, "side_k", 0):
                kw["eps_center"] = True   # single-modal default: the mixture component's mean
            out = self.policy.predict_action(obs_dict, a0=a0, **kw)
            chunk = out["action"][0].detach().to("cpu", torch.float32).numpy()
            if chunk.shape != (self.n_action_steps, self.state_dim):
                raise RuntimeError(f"ODE chunk has shape {chunk.shape}, expected "
                                   f"{(self.n_action_steps, self.state_dim)}")
            if not np.all(np.isfinite(chunk)):
                raise RuntimeError("ODE produced a non-finite target -- refusing to send it to the arm")
            self._chunk, self._k, replanned = chunk, 0, True
            self.n_replans += 1

        k = self._k
        rows = self._chunk[k:]
        if rows.shape[0] < self.n_action_steps:   # hold the last target; only row 0 is executed anyway
            rows = np.concatenate([rows, np.repeat(rows[-1:], self.n_action_steps - rows.shape[0], axis=0)])
        if k == self.replan_every - 1:
            # the last row executed from this chunk: the next integration continues from it
            # (eval_ode_policy.py --carry_state: a_state = ref[i] of the last executed step)
            self._a_state = self._chunk[k, :self.pose_dim].copy()
        self._k += 1

        P = self.pose_dim
        action = np.empty((rows.shape[0], self.state_dim), dtype=np.float32)
        action[:, self.pose_cols] = rows[:, :P]
        targets = np.where(rows[:, P:] >= 0, self.gripper_close_cmd, self.gripper_open_cmd)   # (n, G)
        action[:, self.grip_cols] = targets
        ramp = self.gripper_ramp_ticks > 0
        if ramp.any():
            if self._grip_cmd is None:
                ap = obs_dict["agent_pos"]
                self._grip_cmd = (ap[0, -1, self.grip_cols].detach().to("cpu", torch.float64).numpy()
                                  if ap.shape[-1] == self.state_dim else targets[0].astype(np.float64))
            step = np.abs(self.gripper_open_cmd - self.gripper_close_cmd) / np.maximum(self.gripper_ramp_ticks, 1)
            moved = self._grip_cmd + np.clip(targets[0] - self._grip_cmd, -step, step)
            self._grip_cmd = np.where(ramp, moved, targets[0])
            # only row 0 is executed by the client
            action[0, self.grip_cols] = self._grip_cmd.astype(np.float32)
        return {"action": action, "router_weight": None, "steps_used": None,
                "replanned": replanned, "chunk_index": k}
