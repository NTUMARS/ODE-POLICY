from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ode_g1 import data
from ode_g1.features import FrozenFrameEncoder, combine_conditions
from ode_g1.runtime import ODEPolicy, RGBEncoder, sonic_token_fields, state44_from_body29, state44_from_source43
from test_g1_data_windows import fixture_episode, spec_for
from test_g1_model_contracts import SonicODEConfig, TinyNativeHead, install_sonic_ode
from test_g1_portable import source_fixture


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def prepare_input(self, inputs):
        assert inputs["input_ids"].dtype == torch.long
        return inputs

    def forward(self, inputs):
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        pixel = inputs["pixel_values"].reshape(-1)
        values = torch.stack([pixel + 0.123456, pixel + 0.654321, pixel + 9000], dim=1)
        batch = len(pixel)
        return {
            "backbone_features": values[:, :, None].expand(-1, -1, 8),
            "backbone_attention_mask": torch.tensor([[True, True, False]]).expand(batch, -1),
            "image_mask": torch.tensor([[False, True, True]]).expand(batch, -1),
        }


class FakeCollator:
    def __init__(self):
        self.texts = []
        self.processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")

    def __call__(self, examples):
        self.texts.extend(x["vlm_content"]["conversation"][0]["content"][-1]["text"] for x in examples)
        values = [float(np.asarray(x["vlm_content"]["images"][0])[0, 0, 0]) for x in examples]
        return {
            "inputs": {"pixel_values": torch.tensor(values), "input_ids": torch.ones(len(values), 2, dtype=torch.long)}
        }


def fake_model():
    return SimpleNamespace(
        backbone=FakeBackbone(),
        collator=FakeCollator(),
        config=SimpleNamespace(
            image_target_size=None,
            image_crop_size=None,
            shortest_image_edge=12,
            crop_fraction=1,
            formalize_language=True,
            use_albumentations_transforms=True,
        ),
    )


def identity_transform(*, image):
    return {"image": image}


def test_current_rgb_and_frozen_frame_api_use_same_order_masks_and_fp16(tmp_path):
    root = tmp_path / "source"
    cameras = source_fixture(root)
    images = {camera: np.full((12, 18, 3), i + 3, np.uint8) for i, camera in enumerate(cameras)}

    class Reader:
        def read(self, path, frame):
            assert frame == 1
            return images[next(key for key in cameras if key in str(path))]

    model = fake_model()
    dataset_encoder = FrozenFrameEncoder(
        model,
        root,
        "cpu",
        eval_transform=identity_transform,
        frame_reader=Reader(),
        camera_keys=cameras,
        feature_cache_dir=tmp_path / "features",
        feature_cache_identity={"toy": "v1"},
    )
    cached = combine_conditions(dataset_encoder.encode_frames([(0, 1)]))
    rgb = RGBEncoder(model, camera_keys=cameras, device="cpu", eval_transform=identity_transform)(
        images, "Move, the object!"
    )
    for key in cached:
        torch.testing.assert_close(cached[key], rgb[key], rtol=0, atol=0)
    assert rgb["backbone_features"].shape == (1, 2, 8)
    assert rgb["image_mask"].tolist() == [[False, True]]
    assert model.collator.texts == ["move the object", "move the object"]
    assert not model.backbone.anchor.requires_grad
    previous_calls = model.backbone.calls
    dataset_encoder.require_cache_hits = True
    dataset_encoder.encode_frames([(0, 1)])
    assert model.backbone.calls == previous_calls
    with pytest.raises(RuntimeError, match="cache hit is missing"):
        dataset_encoder.encode_frames([(0, 2)])


def test_runtime_feedback_layouts_are_explicit_and_hands_have_physical_units():
    source = np.arange(43, dtype=np.float32)
    body29 = np.r_[source[:22], source[29:36]]
    left, right, gravity = np.full(6, 0.2), np.full(6, 0.7), [0, 0, -1]
    expected = state44_from_body29(body29, left, right, gravity)
    np.testing.assert_array_equal(state44_from_source43(source, left, right, gravity), expected)
    np.testing.assert_array_equal(expected[:29], body29)
    with pytest.raises(ValueError):
        state44_from_source43(body29, left, right, gravity)
    with pytest.raises(ValueError):
        state44_from_body29(body29, left * 1000, right, gravity)
    packet = sonic_token_fields(np.arange(64), 7)
    assert packet["token_state"].shape == (1, 64)
    assert packet["frame_index"].dtype == np.int64


def test_prediction_never_commits_unsent_tail_and_reset_restores_bos():
    episode = fixture_episode(20)
    normalizer = data.fit_normalization([episode], spec_for([episode]))
    head = (
        install_sonic_ode(TinyNativeHead(), SonicODEConfig(horizon=4, feature_dim=8, cond_dim=8, width=16))
        .float()
        .eval()
    )
    policy = ODEPolicy(head, normalizer)
    condition = combine_conditions(
        [
            {
                "backbone_features": torch.zeros(1, 2, 8),
                "backbone_attention_mask": torch.ones(1, 2, dtype=torch.bool),
                "image_mask": torch.tensor([[True, False]]),
                "frame_ids": torch.zeros(1, 2, dtype=torch.long),
            }
        ]
    )
    state = episode.state[0]
    first = policy.predict_cached(condition, state)
    second = policy.predict_cached(condition, state)
    np.testing.assert_array_equal(first, second)
    assert policy.carry.executed_steps == 0
    acknowledged = first.copy()
    acknowledged[0, :64] = 37  # External caller may have actually sent a modified command.
    policy.commit_sent_prefix(acknowledged[:1])
    np.testing.assert_allclose(
        policy.carry.normalized_initial_body(normalizer),
        normalizer.normalize_actions_unclipped(acknowledged[:1])[0, :64],
    )
    assert policy.carry.executed_steps == 1
    policy.predict_cached(condition, state)
    assert policy.carry.executed_steps == 1
    with pytest.raises(ValueError):
        policy.initialize_last_sent_body(first[-1, :64])
    policy.reset_episode()
    np.testing.assert_array_equal(policy.carry.normalized_initial_body(normalizer), np.zeros(64))
    policy.initialize_last_sent_body(first[1, :64])
    assert policy.carry.executed_steps == 1
    assert np.all((first[:, 64:] >= 0) & (first[:, 64:] <= 1))
    with pytest.raises(ValueError):
        policy.predict_rgb({}, "test", state)


def test_runtime_rejects_non_rgb_images_and_missing_instruction():
    encoder = RGBEncoder(fake_model(), device="cpu", eval_transform=identity_transform)
    images = {key: np.ones((12, 18, 3), np.uint8) for key in encoder.camera_keys}
    with pytest.raises(ValueError):
        encoder(images, "")
    images[encoder.camera_keys[0]] = np.ones((12, 18, 3), np.float32)
    with pytest.raises(ValueError, match="uint8"):
        encoder(images, "test")
