"""Training and planning share camera transforms and normalize before BF16 casting."""
import numpy as np
import pytest
import torch

from hydra.utils import instantiate

from ode_imagewam.config import load_config
from ode_imagewam.inference import OfflinePlanner


@pytest.mark.parametrize("camera_shapes", [((32, 32), (32, 32)), ((19, 53), (47, 21))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eval_training_and_planner_use_identical_float32_images(camera_shapes, dtype):
    cfg = load_config()
    for meta in cfg.data.train.shape_meta.images:
        meta.shape = [3, 16, 16]
    for meta in cfg.data.train.processor.shape_meta.images:
        meta.shape = [3, 16, 16]
    for mode in ("train_transforms", "val_transforms"):
        cfg.data.train.processor[mode][1].size = [16, 16]
    cfg.data.train.video_size = [16, 32]
    processor = instantiate(cfg.data.train.processor).eval()
    stats = {field: {"default": {"global_min": [-1] * width, "global_max": [1] * width}}
             for field, width in (("state", 8), ("action", 7))}
    processor.set_normalizer_from_stats(stats)
    rng = np.random.default_rng(12)
    images = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
              for height, width in camera_shapes]
    keys = [str(meta.key) for meta in cfg.data.train.shape_meta.images]
    clips = {key: torch.from_numpy(image.copy()).permute(2, 0, 1)[None].expand(2, -1, -1, -1)
             for key, image in zip(keys, images)}
    state = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.3, 0.02, 0.02], dtype=np.float32)
    training = processor.preprocess({"images": clips,
        "state": {"default": torch.from_numpy(state.copy())[None].expand(13, -1)},
        "state_is_pad": torch.zeros(13, dtype=torch.bool), "image_is_pad": torch.zeros(2, dtype=torch.bool),
        "task": "move the mug", "idx": 0})
    expected = processor.pack_video(training["pixel_values"], cfg.data.train.video_size)[:, 0][None]
    assert expected.dtype == torch.float32
    captured = {}
    pack_video = processor.pack_video

    def capture_pack(pixels, video_size):
        result = pack_video(pixels, video_size)
        captured["before_cast"] = result[:, 0][None].clone()
        return result

    processor.pack_video = capture_pack

    class Model:
        def infer_action(self, **kwargs):
            captured.update(kwargs)
            horizon = kwargs["action_horizon"]
            return {"ode_targets": torch.zeros(1, horizon, 6),
                    "ode_gripper": torch.zeros(1, horizon), "ode_initial": torch.zeros(1, 6)}

    planner = OfflinePlanner.__new__(OfflinePlanner)
    planner.cfg, planner.processor, planner.model = cfg, processor, Model()
    planner.device, planner.dtype, planner.torch = "cpu", dtype, torch
    planner._text_context = lambda _: (torch.zeros(1, 128, 8), torch.ones(1, 128, dtype=torch.bool))
    planner.plan(*images, state, "move the mug")
    assert captured["before_cast"].dtype == torch.float32
    assert torch.equal(captured["before_cast"], expected)
    assert torch.equal(captured["input_image"], expected.to(dtype))
    assert torch.equal(captured["proprio"][0], training["proprio"][0])
