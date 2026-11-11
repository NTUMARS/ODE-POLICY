from types import SimpleNamespace

import pytest
import torch

from ode_imagewam.config import load_config, validate_config
from ode_imagewam.training import training_scope


def test_full_finetuning_checks_optimizer_membership_and_frozen_encoders():
    groups = {name: torch.nn.Linear(2, 2) for name in ("video_expert", "action_expert", "proprio_encoder")}
    model = SimpleNamespace(**groups, vae=torch.nn.Linear(2, 2).requires_grad_(False), text_encoder=None)
    params = [p for module in groups.values() for p in module.parameters()]
    optimizer = torch.optim.AdamW(params)
    scope = training_scope(model, optimizer)
    assert scope["lora"] is False and sum(scope["trainable_parameters"].values()) == 18
    model.video_expert.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="visual_dit"):
        training_scope(model, optimizer)
    model.video_expert.weight.requires_grad_(True)
    missing = torch.optim.AdamW(list(model.action_expert.parameters()))
    with pytest.raises(ValueError, match="visual_dit"):
        training_scope(model, missing)
    model.vae.requires_grad_(True)
    with pytest.raises(ValueError, match="vae"):
        training_scope(model, optimizer)


def test_recipe_disables_lora_and_keeps_all_frames():
    cfg = load_config()
    validate_config(cfg)
    assert cfg.model.flux2_lora_config.enabled is False
    assert cfg.data.train.global_sample_stride == 1
    assert cfg.data.train.val_set_proportion == 0
    cfg.model.flux2_lora_config.enabled = True
    with pytest.raises(ValueError, match="LoRA"):
        validate_config(cfg)
