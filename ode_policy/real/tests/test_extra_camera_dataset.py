"""RobotImageDataset.extra_camera_keys: a second camera flows from the zarr to the batch as
`wrist_cam` and gets an image-range normalizer; the single-camera default is unchanged."""

import numpy as np
import torch
import zarr

from ode_policy.datasets.robot_image_dataset import RobotImageDataset


def _zarr(tmp_path, n=12, ends=(6, 12)):
    path = tmp_path / "d.zarr"
    root = zarr.open(str(path), mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    data.create_dataset("head_camera", data=np.full((n, 3, 8, 8), 10, dtype=np.uint8))
    data.create_dataset("wrist_camera", data=np.full((n, 3, 8, 8), 200, dtype=np.uint8))
    data.create_dataset("state", data=np.random.default_rng(0).normal(size=(n, 7)).astype(np.float32))
    data.create_dataset("action", data=np.random.default_rng(1).normal(size=(n, 7)).astype(np.float32))
    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    return str(path)


def test_two_cameras_reach_the_batch(tmp_path):
    ds = RobotImageDataset(_zarr(tmp_path), horizon=4, pad_before=1, pad_after=1, val_ratio=0.0, batch_size=2,
                           extra_camera_keys=["wrist_camera"])
    idx = np.array([0, 1])
    batch = ds.postprocess(ds[idx], torch.device("cpu"))
    assert set(batch["obs"]) == {"head_cam", "wrist_cam", "agent_pos"}
    assert batch["obs"]["wrist_cam"].shape == (2, 4, 3, 8, 8)
    torch.testing.assert_close(batch["obs"]["wrist_cam"].mean(), torch.tensor(200 / 255))
    torch.testing.assert_close(batch["obs"]["head_cam"].mean(), torch.tensor(10 / 255))
    assert batch["action"].shape == (2, 4, 7)
    norm = ds.get_normalizer()
    assert "wrist_cam" in norm.params_dict and "head_cam" in norm.params_dict
    img = torch.full((1, 3, 8, 8), 0.5)
    torch.testing.assert_close(norm["wrist_cam"].normalize(img), torch.zeros_like(img))


def test_default_is_single_camera(tmp_path):
    ds = RobotImageDataset(_zarr(tmp_path), horizon=4, pad_before=1, pad_after=1, val_ratio=0.0, batch_size=2)
    batch = ds.postprocess(ds[np.array([0, 1])], torch.device("cpu"))
    assert set(batch["obs"]) == {"head_cam", "agent_pos"}
    assert "wrist_camera" not in ds.replay_buffer.keys()


def test_three_cameras_two_arms(tmp_path):
    path = _zarr(tmp_path)
    root = zarr.open(path, mode="a")
    root["data"].create_dataset("left_wrist_camera", data=np.full((12, 3, 8, 8), 100, dtype=np.uint8))
    ds = RobotImageDataset(path, horizon=4, pad_before=1, pad_after=1, val_ratio=0.0, batch_size=2,
                           extra_camera_keys=["wrist_camera", "left_wrist_camera"])
    batch = ds.postprocess(ds[np.array([0, 1])], torch.device("cpu"))
    assert set(batch["obs"]) == {"head_cam", "wrist_cam", "left_wrist_cam", "agent_pos"}
    torch.testing.assert_close(batch["obs"]["left_wrist_cam"].mean(), torch.tensor(100 / 255))
    torch.testing.assert_close(batch["obs"]["wrist_cam"].mean(), torch.tensor(200 / 255))
    assert "left_wrist_cam" in ds.get_normalizer().params_dict
