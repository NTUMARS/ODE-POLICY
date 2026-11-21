"""robomimic tasks (robosuite 1.4.1) behind the interface of the LIBERO env wrapper.

ode_policy/libero/eval_ode_policy.py drives a LIBERO env through reset / set_init_state / step /
check_success / _get_observations / close and reads the robosuite observation keys (agentview_image,
robot0_eye_in_hand_image, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos). RobomimicEnv offers the
same, so the LIBERO rollout (pose tracker, ODE carry state, gripper, metrics) runs unchanged. Controller,
control rate and the other env kwargs come from the dataset's env_args (robomimic convention).
"""
from __future__ import annotations

import json

import h5py
import numpy as np

from ode_policy.libero.libero_env import _configure_egl_device

CAMERAS = ("agentview", "robot0_eye_in_hand")
# task: (proficient-human low-dim hdf5, rollout horizon as robomimic / Diffusion Policy, instruction)
TASKS = {
    "tool_hang": ("data_policy/robomimic/tool_hang/ph/low_dim_v141.hdf5", 700, "hang the tool on the frame"),
    "square": ("data_policy/robomimic/square/ph/low_dim_v141.hdf5", 400, "put the square nut on the peg"),
    "can": ("data_policy/robomimic/can/ph/low_dim_v141.hdf5", 400, "put the can in its bin"),
    "lift": ("data_policy/robomimic/lift/ph/low_dim_v141.hdf5", 400, "lift the cube"),
}


def _env_args(path):
    with h5py.File(path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


class RobomimicEnv:
    def __init__(self, hdf5: str, image_size: int = 240, with_wrist: bool = True, control_freq: int = 20):
        import robosuite
        _configure_egl_device()
        args = _env_args(hdf5)
        kw = dict(args["env_kwargs"])
        kw.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                  use_object_obs=True, ignore_done=True, control_freq=control_freq,
                  camera_names=list(CAMERAS if with_wrist else CAMERAS[:1]),
                  camera_heights=image_size, camera_widths=image_size, camera_depths=False)
        robots = kw.pop("robots", "Panda")
        self.env = robosuite.make(args["env_name"], robots=robots, **kw)

    # --- LIBERO env interface ---
    def reset(self):
        return self.env.reset()

    def set_init_state(self, state):
        """Restore a flattened MuJoCo state (a dataset `states` row) and return the observation."""
        self.env.sim.set_state_from_flattened(np.asarray(state))
        self.env.sim.forward()
        return self._get_observations()

    def _get_observations(self):
        return self.env._get_observations(force_update=True)

    def step(self, action):
        return self.env.step(np.asarray(action, dtype=np.float64))

    def check_success(self) -> bool:
        return bool(self.env._check_success())

    def close(self):
        self.env.close()

    @property
    def robots(self):
        return self.env.robots
