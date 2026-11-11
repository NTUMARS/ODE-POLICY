"""LIBERO environment I/O and the benchmark's saved-state filename conventions."""
import math
import re
from pathlib import Path

import numpy as np


def make_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution,
                             camera_widths=resolution)
    env.seed(seed)
    return env


def initial_state_path(task, root, plus=False):
    name = task.init_states_file
    folder = Path(root) / task.problem_folder
    if plus:
        extension = name.rsplit(".", 1)[-1]
        if "_language_" in name:
            name = name.split("_language_", 1)[0] + "." + extension
        elif "_view_" in name:
            name = name.split("_view_", 1)[0] + "." + extension
        elif "_add_" in name or "_level" in name:
            return Path(root) / "libero_newobj" / task.problem_folder / name
        elif "_light_" in name:
            name = name.split("_light_", 1)[0] + "." + extension
        else:
            name = re.sub(r"_(?:table|tb)_\d+", "", name)
    return folder / name


def load_initial_states(suite, task_id):
    task = suite.get_task(task_id)
    if not hasattr(task, "init_states_file"):
        return suite.get_task_init_states(task_id)
    import torch
    from libero.libero import benchmark, get_libero_path

    plus = (Path(benchmark.__file__).parent / "task_classification.json").is_file()
    path = initial_state_path(task, get_libero_path("init_states"), plus)
    # The official saved-state assets contain NumPy arrays, not only tensors.
    states = torch.load(path, map_location="cpu", weights_only=False)
    if plus and ("_add_" in task.init_states_file or "_level" in task.init_states_file):
        states = states.reshape(1, -1)
    return states


def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(observation):
    return {"image": np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]),
            "wrist_image": np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])}


def quat2axisangle(quaternion):
    value = np.asarray(quaternion)
    w = float(np.clip(value[3], -1, 1))
    denominator = math.sqrt(1 - w * w)
    if math.isclose(denominator, 0):
        return np.zeros(3)
    return value[:3] * (2 * math.acos(w) / denominator)
