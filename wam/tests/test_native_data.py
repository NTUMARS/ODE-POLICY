"""Full-frame episode windows, camera endpoints and processor identities on CPU."""
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ode_imagewam.data.normalizer import LinearNormalizer
from ode_imagewam.data.processor import ImageWAMProcessor
from ode_imagewam.data.transforms import VideoAugmentation
from ode_imagewam.dataset import DEFAULT_PROMPT, ODERobotVideoDataset
from ode_imagewam.sidecar import prepare


def write_root(root, instruction, episode_sizes):
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_frames": sum(episode_sizes), "fps": 20,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": instruction}) + "\n")
    offset = 0
    for episode, size in enumerate(episode_sizes):
        state = np.zeros((size, 8), dtype=np.float32)
        state[:, 0] = np.arange(size) * 0.01 + offset * 0.01
        state[:, 6:] = 0.02
        action = np.zeros((size, 7), dtype=np.float32)
        action[:, 0] = np.arange(size) * 0.03 + 0.1
        action[:, 6] = 1
        table = pa.table({"index": np.arange(offset, offset + size), "frame_index": np.arange(size),
            "episode_index": np.full(size, episode), "timestamp": np.arange(size) / 20,
            "task_index": np.zeros(size, dtype=np.int64), "observation.state": state.tolist(),
            "action": action.tolist()})
        pq.write_table(table, root / f"data/chunk-000/episode_{episode:06d}.parquet")
        offset += size


def build_dataset(tmp_path):
    roots = [tmp_path / "spatial", tmp_path / "object"]
    write_root(roots[0], "move the mug", [3, 2])
    write_root(roots[1], "place the bowl", [4])
    labels, cache = tmp_path / "labels", tmp_path / "text"
    prepare(roots, None, labels)
    cache.mkdir()
    for task in ["move the mug", "place the bowl"]:
        prompt = DEFAULT_PROMPT.format(task=task)
        name = f"{hashlib.sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len128.pt"
        torch.save({"text_hidden_states": torch.ones(128, 8), "text_attention_mask": torch.ones(128, dtype=torch.bool)}, cache / name)
    cameras = [{"key": "image", "lerobot_key": "observation.images.agentview", "shape": [3, 4, 4]},
               {"key": "wrist_image", "shape": [3, 4, 4]}]
    calls = []
    def reader(path, timestamps, tolerance_s):
        calls.append((str(path), timestamps, tolerance_s))
        value = 64 if "agentview" in str(path) else 192
        return torch.full((len(timestamps), 3, 4, 4), value, dtype=torch.uint8)
    dataset = ODERobotVideoDataset(dataset_dirs=roots, shape_meta={"images": cameras},
        ode_sidecar_dir=labels, video_size=[4, 8], qwen_text_cache_dir=cache, video_reader=reader)
    return dataset, calls, roots


def test_native_windows_preserve_every_frame_and_endpoint_camera_order(tmp_path):
    dataset, calls, roots = build_dataset(tmp_path)
    assert len(dataset) == 9
    first = dataset[0]
    assert first["video"].shape == (3, 2, 4, 8)
    assert first["proprio"].shape == (12, 8)
    assert first["action"].shape == (12, 7)
    assert calls[0][0].startswith(str(roots[0]))
    assert calls[0][1] == [0.0, 0.1]
    torch.testing.assert_close(first["video"][..., :4], torch.full((3, 2, 4, 4), 64 * 2 / 255 - 1))
    torch.testing.assert_close(first["video"][..., 4:], torch.full((3, 2, 4, 4), 192 * 2 / 255 - 1))
    assert first["image_is_pad"].tolist() == [False, True]
    assert first["action_is_pad"].tolist() == [False] * 3 + [True] * 9
    assert torch.equal(first["ode_target"][3:], first["ode_target"][2:3].expand(9, -1))
    raw = dataset.processor.normalizer.normalizers["action"]["default"].backward(first["action"])
    torch.testing.assert_close(raw[3:, :6], torch.zeros(9, 6), atol=1e-7, rtol=0)
    assert raw[3:, 6].eq(1).all()
    final = dataset[8]
    assert final["dataset_index"] == 1 and final["episode_index"] == 0
    assert final["prompt"] == DEFAULT_PROMPT.format(task="place the bowl")
    assert calls[-1][1] == [0.15, 0.15]
    assert final["action_is_pad"].tolist() == [False] + [True] * 11
    with pytest.raises(IndexError):
        dataset[9]


def test_native_samples_do_not_cross_episode_boundaries(tmp_path):
    dataset, _, _ = build_dataset(tmp_path)
    last = dataset[2]
    next_episode = dataset[3]
    assert last["episode_index"] == 0 and next_episode["episode_index"] == 1
    assert last["action_is_pad"].sum().item() == 11
    torch.testing.assert_close(next_episode["ode_start"], torch.from_numpy(np.array(dataset.ode_store.starts[3])))
    restored_state = dataset.processor.normalizer.normalizers["state"]["default"].backward(last["proprio"])
    assert torch.equal(restored_state[1:], restored_state[:1].expand(11, -1))


def test_native_normalizer_constant_channel_and_roundtrip():
    stats = {field: {"default": {"global_min": [2.0] * width, "global_max": [2.0] * width}}
             for field, width in (("state", 8), ("action", 7))}
    normalizer = LinearNormalizer(stats)
    state = torch.full((13, 8), 2.0)
    torch.testing.assert_close(normalizer.normalizers["state"]["default"].forward(state), torch.zeros_like(state))
    value = torch.tensor([[2.5] * 7])
    encoded = normalizer.normalizers["action"]["default"].forward(value)
    torch.testing.assert_close(encoded, torch.full_like(value, 0.5))
    torch.testing.assert_close(normalizer.normalizers["action"]["default"].backward(encoded), value)


def test_clip_augmentation_keeps_shared_temporal_parameters():
    torch.manual_seed(17)
    frames = torch.full((2, 2, 3, 8, 8), 0.5)
    augment = VideoAugmentation(p=1, augment_types=["both"],
        color_jitter={"brightness": 0.3}, gaussian_noise={"std": 0.02},
        rotate={"degrees": 3.0, "fill": "mean"})
    actual = augment(frames)
    assert torch.equal(actual[:, 0], actual[:, 1])
    assert actual.shape == frames.shape and actual.min() >= 0 and actual.max() <= 1


def test_processor_rejects_non_libero_schema():
    with pytest.raises(ValueError, match="action7"):
        ImageWAMProcessor({"images": [{}, {}]}, action_output_dim=12)


def test_packaged_config_builds_native_dataset_without_external_checkout(tmp_path, monkeypatch):
    import ode_imagewam.dataset as module
    from ode_imagewam.config import load_config

    prepared, _, roots = build_dataset(tmp_path)
    monkeypatch.setattr(module, "decode_video_frames", prepared.video_reader)
    cfg = load_config()
    cfg.paths.dataset_dirs = [str(root) for root in roots]
    cfg.paths.sidecar_dir = str(tmp_path / "labels")
    cfg.paths.text_cache_dir = str(tmp_path / "text")
    cfg.data.train.video_augmentation = None
    train, validation = module.build_datasets(cfg.data)
    assert type(train) is ODERobotVideoDataset and validation is None
    assert len(train) == 9
    sample = train[5]
    assert sample["video"].shape == (3, 2, 224, 448)
    assert sample["dataset_index"] == 1
