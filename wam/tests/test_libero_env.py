from types import SimpleNamespace

import numpy as np
import pytest

from ode_imagewam.libero_env import get_libero_image, initial_state_path, quat2axisangle


@pytest.mark.parametrize("filename,expected", [
    ("task_language_2.pruned_init", "suite/task.pruned_init"),
    ("task_view_0_initstate_3_noise_1.pruned_init", "suite/task.pruned_init"),
    ("task_table_2.pruned_init", "suite/task.pruned_init"),
    ("task_tb_3.pruned_init", "suite/task.pruned_init"),
    ("task_light_4.pruned_init", "suite/task.pruned_init"),
    ("task_add_2.pruned_init", "libero_newobj/suite/task_add_2.pruned_init"),
    ("task_level2_sample3.pruned_init", "libero_newobj/suite/task_level2_sample3.pruned_init"),
])
def test_plus_saved_state_mapping_uses_the_benchmark_conventions(tmp_path, filename, expected):
    task = SimpleNamespace(problem_folder="suite", init_states_file=filename)
    assert initial_state_path(task, tmp_path, plus=True) == tmp_path / expected
    assert initial_state_path(task, tmp_path, plus=False) == tmp_path / "suite" / filename


def test_camera_rotation_and_quaternion_conversion_do_not_mutate_feedback():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    result = get_libero_image({"agentview_image": image, "robot0_eye_in_hand_image": image + 20})
    np.testing.assert_array_equal(result["image"], image[::-1, ::-1])
    np.testing.assert_array_equal(result["wrist_image"], (image + 20)[::-1, ::-1])
    quaternion = np.array([0.0, 0.0, np.sin(0.3), np.cos(0.3)])
    before = quaternion.copy()
    np.testing.assert_allclose(quat2axisangle(quaternion), [0, 0, 0.6])
    np.testing.assert_array_equal(quaternion, before)
    np.testing.assert_array_equal(quat2axisangle([0, 0, 0, 1]), np.zeros(3))
