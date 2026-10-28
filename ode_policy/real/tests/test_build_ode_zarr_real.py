"""The row convention of the real-robot ODE zarr (the crux of the port) on a synthetic zarr."""

import os

import numpy as np
import zarr

from ode_policy.real.build_ode_zarr_real import build, data_report, gripper_labels, ode_actions

P = 6


def test_gripper_labels_ramp_start_flips_where_the_ramp_begins():
    # one episode: open plateau (100) x3, ramp down 80/60/40/20, closed (0) x3, ramp up 30/70, open x2
    cmd = np.array([100, 100, 100, 80, 60, 40, 20, 0, 0, 0, 30, 70, 100, 100], dtype=np.float64)
    ends = np.array([len(cmd)])
    thr = gripper_labels(cmd, ends, 50.0, True, mode="threshold")
    np.testing.assert_allclose(thr, [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1])
    rs = gripper_labels(cmd, ends, 50.0, True, mode="ramp_start")
    # close from the first ramp frame (80), open from the first frame of the up-ramp (30)
    np.testing.assert_allclose(rs, [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
    # a ramp cut by an episode end inherits nothing from the next episode
    two = gripper_labels(np.concatenate([cmd[:5], cmd]), np.array([5, 5 + len(cmd)]), 50.0, True)
    np.testing.assert_allclose(two[:5], [-1, -1, -1, -1, -1])   # ramp frames 80, 60 end the episode: threshold label


def _synthetic(tmp_path, lengths=(5, 4)):
    """command-layout zarr: state[t] = 100*ep + t (+0.1 per joint), action[t] = state[t] + 0.5 (the command
    leads the state by half a step), gripper command 100 (open) then 0 (close) half way."""
    src = tmp_path / "src.zarr"
    root = zarr.open(str(src), mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    states, actions, ends, n = [], [], [], 0
    for ep, L in enumerate(lengths):
        t = np.arange(L, dtype=np.float32)[:, None]
        joints = 100.0 * ep + t + 0.1 * np.arange(P, dtype=np.float32)[None, :]
        grip_state = np.full((L, 1), 80.0, dtype=np.float32)
        grip_cmd = np.where(t < L // 2, 100.0, 0.0).astype(np.float32)
        states.append(np.concatenate([joints, grip_state], axis=1))
        actions.append(np.concatenate([joints + 0.5, grip_cmd], axis=1))
        n += L
        ends.append(n)
    data.create_dataset("state", data=np.concatenate(states))
    data.create_dataset("action", data=np.concatenate(actions))
    data.create_dataset("head_camera", data=np.zeros((n, 3, 8, 8), dtype=np.uint8))
    data.create_dataset("wrist_camera", data=np.ones((n, 3, 8, 8), dtype=np.uint8))
    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    meta.attrs["task_name"] = "synthetic"
    return src, np.concatenate(states), np.concatenate(actions), np.asarray(ends)


def test_ode_actions_row_convention_and_gripper_sign(tmp_path):
    _, state, action, ends = _synthetic(tmp_path)
    acts = ode_actions(state, action, ends, P, gripper_threshold=50.0, close_is_low=True)
    assert acts.shape == (9, P + 1) and acts.dtype == np.float32
    # episode starts: the measured joints
    np.testing.assert_allclose(acts[0, :P], state[0, :P])
    np.testing.assert_allclose(acts[5, :P], state[5, :P])
    # every other row: the command issued one frame earlier (never crossing an episode cut)
    np.testing.assert_allclose(acts[1:5, :P], action[0:4, :P])
    np.testing.assert_allclose(acts[6:9, :P], action[5:8, :P])
    # gripper: NOT shifted; command < threshold means close -> +1
    expect = np.where(action[:, P] < 50.0, 1.0, -1.0)
    np.testing.assert_allclose(acts[:, P], expect)
    # the opposite sign convention
    acts_hi = ode_actions(state, action, ends, P, gripper_threshold=50.0, close_is_low=False)
    np.testing.assert_allclose(acts_hi[:, P], -expect)


def test_build_symlinks_images_and_rewrites_action(tmp_path):
    src, state, action, ends = _synthetic(tmp_path)
    out = tmp_path / "out_ode.zarr"
    rep = build(src, out, P, 50.0, True, force=False)
    for name in ("head_camera", "wrist_camera", "state"):
        assert os.path.islink(out / "data" / name)
    assert os.path.islink(out / "meta" / "episode_ends")
    assert not os.path.islink(out / "data" / "action")
    z = zarr.open(str(out), mode="r")
    np.testing.assert_array_equal(z["meta"]["episode_ends"][:], ends)
    np.testing.assert_allclose(z["data"]["action"][:], ode_actions(state, action, ends, P, 50.0, True))
    np.testing.assert_array_equal(z["data"]["state"][:], state)
    assert z["data"]["head_camera"].shape == (9, 3, 8, 8)
    attrs = z["meta"].attrs
    assert attrs["ode_joint_space"] is True and attrs["ode_pose_dim"] == P and attrs["task_name"] == "synthetic"
    assert attrs["gripper_close_cmd"] == 0.0 and attrs["gripper_open_cmd"] == 100.0
    assert attrs["gripper_label"] == "ramp_start" and attrs["recommended_gripper_ramp_ticks"] == 0
    assert len(attrs["recommended_perturb_pos"]) == P and all(v >= 0.005 for v in attrs["recommended_perturb_pos"])
    assert rep["episodes"] == 2 and rep["frames"] == 9
    # the command leads the measured state by half a step: lag 0 and lag 1 tie (0.5 each) -> lag 0 wins
    # the argmin; what matters is that the errors are small and that a lag >= 1 is not worse than lag 0
    err = rep["cmd_vs_state_mean_abs_err_by_lag"]
    assert err[0] == [0.5] * P and err[1] == [0.5] * P and err[2] == [1.5] * P
    assert rep["step_abs_p50"] == [1.0] * P
    assert rep["gripper_close_fraction"] > 0 and rep["gripper_cmd_binary_fraction"] == 1.0
    assert rep["gripper_toggles_per_episode_max"] == 1


def test_build_refuses_to_overwrite_without_force(tmp_path):
    src, *_ = _synthetic(tmp_path)
    out = tmp_path / "o.zarr"
    build(src, out, P, 50.0, True, force=False)
    import pytest
    with pytest.raises(FileExistsError):
        build(src, out, P, 50.0, True, force=False)
    build(src, out, P, 50.0, True, force=True)   # rebuild is fine


def test_data_report_gripper_histogram_keys(tmp_path):
    _, state, action, ends = _synthetic(tmp_path)
    rep = data_report(state, action, ends, P, 50.0, True)
    assert set(rep["gripper_cmd_hist"]) == {"counts", "edges"} and sum(rep["gripper_cmd_hist"]["counts"]) == 9
    assert len(rep["best_lag_per_joint"]) == P and len(rep["step_norm_p50"]) == P
    assert len(rep["recommended_perturb_pos"]) == P and min(rep["recommended_perturb_pos"]) >= 0.005
    assert rep["gripper_ramp_frames_mean"] == 0.0 and rep["recommended_gripper_ramp_ticks"] == 0


# ── two arms: [L1..L6, L_grip, R1..R6, R_grip], the layout lerobot2zarr.py --preset bimanual writes ──

BI_FIELDS = ["action.left_arm", "action.left_gripper", "action.right_arm", "action.right_gripper"]
BI_STATE_FIELDS = ["observation.state.left_arm", "observation.state.left_gripper",
                   "observation.state.right_arm", "observation.state.right_gripper"]
POSE_COLS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def test_zarr_layout_two_arms_and_single_arm_attrs():
    from ode_policy.real.build_ode_zarr_real import zarr_layout
    attrs = {"action_fields": BI_FIELDS, "action_field_dims": [6, 1, 6, 1],
             "state_fields": BI_STATE_FIELDS, "state_field_dims": [6, 1, 6, 1]}
    lay = zarr_layout(attrs, 14, 14)
    assert lay["pose_cols"] == POSE_COLS and lay["grip_cols"] == [6, 13]
    assert lay["joint_names"] == [f"L{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 7)]
    assert lay["gripper_names"] == ["L_grip", "R_grip"]
    # a single-arm zarr written before the widths were recorded (r1lite_insert_flower.zarr)
    old = zarr_layout({"action_fields": ["action.right_arm", "action.right_gripper"]}, 7, 7)
    assert old["pose_cols"] == list(range(6)) and old["grip_cols"] == [6] and old["gripper_names"] == ["R_grip"]
    # no field attrs at all: joints first, gripper last
    assert zarr_layout({}, 7, 7, pose_dim=6)["grip_cols"] == [6]
    import pytest
    with pytest.raises(ValueError, match="state field widths"):
        zarr_layout(dict(attrs, state_field_dims=[7, 7]), 14, 14)
    with pytest.raises(ValueError, match="pose-dim"):
        zarr_layout(attrs, 14, 14, pose_dim=6)


def _synthetic_bimanual(tmp_path, lengths=(6, 5)):
    """Two-arm command-layout zarr: left joints 10*ep + t + 0.1 j, right joints the same + 50; the command
    leads by half a step; the LEFT gripper closes at L // 2, the RIGHT one is closed from the start and
    opens at L - 2, so the two gripper columns can't be confused."""
    src = tmp_path / "bi.zarr"
    root = zarr.open(str(src), mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    states, actions, ends, n = [], [], [], 0
    for ep, L in enumerate(lengths):
        t = np.arange(L, dtype=np.float32)[:, None]
        left = 10.0 * ep + t + 0.1 * np.arange(6, dtype=np.float32)[None]
        right = left + 50.0
        lg = np.where(t < L // 2, 100.0, 0.0).astype(np.float32)
        rg = np.where(t < L - 2, 0.0, 100.0).astype(np.float32)
        width = np.full((L, 1), 70.0, dtype=np.float32)
        states.append(np.concatenate([left, width, right, width], axis=1))
        actions.append(np.concatenate([left + 0.5, lg, right + 0.5, rg], axis=1))
        n += L
        ends.append(n)
    for k, v in (("state", np.concatenate(states)), ("action", np.concatenate(actions))):
        data.create_dataset(k, data=v)
    for cam in ("head_camera", "wrist_camera", "left_wrist_camera"):
        data.create_dataset(cam, data=np.zeros((n, 3, 8, 8), dtype=np.uint8))
    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    meta.attrs.update({"task_name": "bi", "layout": "bimanual", "action_fields": BI_FIELDS,
                       "action_field_dims": [6, 1, 6, 1], "state_fields": BI_STATE_FIELDS,
                       "state_field_dims": [6, 1, 6, 1]})
    return src, np.concatenate(states), np.concatenate(actions), np.asarray(ends)


def test_ode_actions_two_arms(tmp_path):
    _, state, action, ends = _synthetic_bimanual(tmp_path)
    acts = ode_actions(state, action, ends, 12, 50.0, True, pose_cols=POSE_COLS, grip_cols=[6, 13])
    assert acts.shape == (11, 14) and acts.dtype == np.float32
    # joints: [left 6, right 6], each with the row convention (episode start = measured, else previous command)
    np.testing.assert_allclose(acts[0, :12], state[0, POSE_COLS])
    np.testing.assert_allclose(acts[6, :12], state[6, POSE_COLS])
    np.testing.assert_allclose(acts[1:6, :12], action[0:5][:, POSE_COLS])
    np.testing.assert_allclose(acts[7:11, :12], action[6:10][:, POSE_COLS])
    np.testing.assert_allclose(acts[:, 6:12] - acts[:, 0:6], 50.0, atol=1e-4)   # right arm block after the left one
    # grippers last, left then right, each from its own column (close = command < 50 -> +1)
    np.testing.assert_allclose(acts[:, 12], np.where(action[:, 6] < 50.0, 1.0, -1.0))
    np.testing.assert_allclose(acts[:, 13], np.where(action[:, 13] < 50.0, 1.0, -1.0))
    assert acts[0, 12] == -1.0 and acts[0, 13] == 1.0               # left open, right closed at the start


def test_build_two_arms_attrs_and_report(tmp_path):
    src, state, action, ends = _synthetic_bimanual(tmp_path)
    out = tmp_path / "bi_ode.zarr"
    rep = build(src, out, None, 50.0, True, force=False)
    for name in ("head_camera", "wrist_camera", "left_wrist_camera", "state"):
        assert os.path.islink(out / "data" / name)
    z = zarr.open(str(out), mode="r")
    np.testing.assert_allclose(z["data"]["action"][:], ode_actions(state, action, ends, 12, 50.0, True,
                                                                   pose_cols=POSE_COLS, grip_cols=[6, 13]))
    a = z["meta"].attrs
    assert a["ode_pose_dim"] == 12 and a["ode_n_grippers"] == 2
    assert a["ode_pose_cols"] == POSE_COLS and a["ode_grip_cols"] == [6, 13]
    assert a["ode_gripper_names"] == ["L_grip", "R_grip"] and a["layout"] == "bimanual"
    assert len(a["recommended_perturb_pos"]) == 12
    # per-gripper entries are lists, left first
    assert a["gripper_close_cmd"] == [0.0, 0.0] and a["gripper_open_cmd"] == [100.0, 100.0]
    assert a["recommended_gripper_ramp_ticks"] == [0, 0]
    assert rep["gripper_names"] == ["L_grip", "R_grip"] and rep["gripper_toggles_per_episode_max"] == [1, 1]
    assert len(rep["best_lag_per_joint"]) == 12 and rep["joint_names"][6] == "R1"
