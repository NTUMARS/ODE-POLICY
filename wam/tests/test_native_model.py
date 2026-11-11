"""Exercise the native model with small official FLUX.2 transformer blocks."""
import copy
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ode_imagewam.backbones.flux2_imports import ensure_flux2_importable
from ode_imagewam.checkpoint import save_weights, strict_load_checkpoint
from ode_imagewam.initialization import transfer_reader_weights
from ode_imagewam.model import create_imagewam_ode
from ode_imagewam.sidecar import DEFAULT_GEOMETRY


@pytest.fixture
def tiny_model(monkeypatch):
    source = os.environ.get("FLUX2_SOURCE")
    if source:
        assert (Path(source) / "src/flux2/model.py").is_file()
    ensure_flux2_importable(source)
    from flux2.model import Flux2, Flux2Params
    from ode_imagewam.backbones.action import ActionDiTFlux2
    from ode_imagewam.backbones.base import ImageWAM
    from ode_imagewam.backbones.mot import MoT
    from ode_imagewam.backbones.video import Flux2VideoExpert

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
        def encode(self, image):
            value = F.avg_pool2d(image.mean(1, keepdim=True), 16)
            return value.expand(-1, 8, -1, -1) * self.scale

    torch.manual_seed(8)
    params = Flux2Params(in_channels=8, context_in_dim=12, hidden_size=32,
        num_heads=4, depth=5, depth_single_blocks=20, axes_dim=[2, 2, 2, 2],
        mlp_ratio=3, use_guidance_embed=False)
    video = Flux2VideoExpert(Flux2(params), "test")
    action = ActionDiTFlux2(action_dim=7, hidden_dim=16, num_heads=4, attn_head_dim=8,
        num_layers_double=5, num_layers_single=20, max_action_horizon=12)
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=True)
    base = ImageWAM(video, action, mot, Autoencoder(), text_dim=12,
                    device="cpu", torch_dtype=torch.float32)
    metadata = {"reference_center": [0.0] * 6, "reference_half_range": [1.0] * 6,
                "geometry": DEFAULT_GEOMETRY, "action_min": [-1.0] * 7,
                "action_max": [1.0] * 7}
    monkeypatch.setattr("ode_imagewam.runtime.create_imagewam_flux2_klein", lambda **kwargs: base)
    monkeypatch.setattr("ode_imagewam.sidecar.load_metadata", lambda _: metadata)
    model = create_imagewam_ode({"sidecar_dir": ".", "horizon": 3, "cond_dim": 8,
                                "width": 16, "layers": 2, "perturb_prob": 0})
    with torch.no_grad():
        model.action_expert.ode_head.out.weight.normal_(std=0.01)
    return model


def sample(model):
    from ode_imagewam.backbones.video import Flux2VideoExpert
    return {"target_latent": torch.randn(2, 2, 8),
        "target_img_ids": Flux2VideoExpert.build_img_ids(2, 1, 2, time_value=0,
            device=torch.device("cpu"), dtype=torch.float32),
        "ref_image_latents": torch.randn(2, 2, 8),
        "ref_img_ids": Flux2VideoExpert.build_img_ids(2, 1, 2, time_value=10,
            device=torch.device("cpu"), dtype=torch.float32),
        "text_hidden_states": torch.randn(2, 4, 12),
        "text_attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
        "proprio": torch.randn(2, 3, 8), "action": torch.randn(2, 3, 7),
        "action_is_pad": torch.tensor([[False, False, False], [False, False, True]]),
        "ode_start": torch.randn(2, 6), "ode_target": torch.randn(2, 3, 6),
        "ode_grip_target": torch.randint(0, 2, (2, 3)).float()}


def test_real_full_depth_reader_two_objective_backward_and_strict_reload(tiny_model, tmp_path):
    model = tiny_model
    assert len(model.action_expert.double_blocks) == 5
    assert len(model.action_expert.single_blocks) == 20
    batch = sample(model)
    terms = []
    for loss, metrics in model.iter_training_losses(batch):
        assert torch.isfinite(loss) and metrics
        loss.backward()
        terms.append(float(loss.detach()))
    assert len(terms) == model.training_objective_count() == 2
    for component in (model.video_expert, model.action_expert, model.proprio_encoder):
        grads = [p.grad for p in component.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    assert all(p.grad is None and not p.requires_grad for p in model.vae.parameters())
    model.eval()
    prediction = model._ode_context(batch["ref_image_latents"], batch["ref_img_ids"],
                                    batch["text_hidden_states"], batch["text_attention_mask"])
    path = tmp_path / "model.pt"
    save_weights(model, path, step=4)
    replica = copy.deepcopy(model)
    with torch.no_grad():
        next(replica.mot.parameters()).add_(1)
    assert strict_load_checkpoint(replica, path) == 4
    repeated = replica._ode_context(batch["ref_image_latents"], batch["ref_img_ids"],
                                    batch["text_hidden_states"], batch["text_attention_mask"])
    torch.testing.assert_close(prediction, repeated, rtol=0, atol=0)


def test_cached_rgb_inference_does_not_load_a_text_model_and_uses_fp32_ode(tiny_model):
    model = tiny_model
    output = model.infer_action(prompt="pick up the object", input_image=torch.zeros(2, 3, 16, 32),
        action_horizon=3, proprio=torch.zeros(2, 8), raw_proprio=torch.zeros(2, 8),
        context=torch.randn(2, 4, 12), context_mask=torch.ones(2, 4, dtype=torch.bool))
    assert model.text_encoder is None
    assert output["ode_targets"].shape == (2, 3, 6)
    assert output["ode_targets"].dtype == torch.float32
    assert output["ode_gripper"].shape == (2, 3)
    assert all(torch.isfinite(output[key]).all() for key in ("ode_targets", "action", "ode_initial"))


def test_reader_initialization_transfers_every_retained_tensor(tiny_model):
    from ode_imagewam.backbones.action import ActionDiTFlux2
    action = ActionDiTFlux2(action_dim=7, hidden_dim=16, num_heads=4, attn_head_dim=8,
        num_layers_double=5, num_layers_single=20)
    source = tiny_model.video_expert.transformer.state_dict()
    values, counts = transfer_reader_weights(action, source)
    action.load_state_dict(values, strict=True)
    assert counts["copied"] > 0 and counts["resized"] > 0
    assert len(values) == len(action.state_dict())
    corrupted = dict(source)
    corrupted.pop("time_in.in_layer.weight")
    with pytest.raises(ValueError, match="missing reader tensor"):
        transfer_reader_weights(action, corrupted)


def test_real_reader_bfloat16_backward_keeps_ode_rollout_float32(tiny_model):
    model = tiny_model.to(dtype=torch.bfloat16)
    model.torch_dtype = torch.bfloat16
    for loss, _ in model.iter_training_losses(sample(model)):
        assert torch.isfinite(loss)
        loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    output = model.infer_action(prompt="move the object", input_image=torch.zeros(2, 3, 16, 32),
        action_horizon=3, proprio=torch.zeros(2, 8), raw_proprio=torch.zeros(2, 8),
        context=torch.randn(2, 4, 12), context_mask=torch.ones(2, 4, dtype=torch.bool))
    assert output["ode_targets"].dtype == output["ode_initial"].dtype == torch.float32
    assert torch.isfinite(output["ode_targets"]).all()
