import os
from types import SimpleNamespace

import pytest
import torch

from ode_imagewam.config import load_config
from ode_imagewam.devices import device_and_dtype, training_accelerator


def test_explicit_cuda_index_reaches_accelerate_without_gpu_calls(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cpu")
    captured = {}
    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(device=torch.device(os.environ["ACCELERATE_TORCH_DEVICE"]))
    cfg = load_config(overrides=["device=cuda:3", "gradient_accumulation_steps=2"])
    accelerator = training_accelerator(cfg, factory)
    assert accelerator.device == torch.device("cuda:3")
    assert captured["cpu"] is False and captured["gradient_accumulation_steps"] == 2
    assert os.environ["ACCELERATE_TORCH_DEVICE"] == "cpu"
    with pytest.raises(RuntimeError, match="different device"):
        training_accelerator(cfg, lambda **kwargs: SimpleNamespace(device=torch.device("cpu")))
    assert os.environ["ACCELERATE_TORCH_DEVICE"] == "cpu"


def test_rank_selects_device_and_ambiguous_explicit_multi_device_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("LOCAL_RANK", "2")
    cfg = load_config(overrides=["num_processes=4", "backend=deepspeed", "device=cuda"])
    factory = lambda **kwargs: SimpleNamespace(device=torch.device(os.environ["ACCELERATE_TORCH_DEVICE"]))
    assert training_accelerator(cfg, factory).device == torch.device("cuda:2")
    cfg.device = "cuda:0"
    with pytest.raises(ValueError, match="ranks select"):
        training_accelerator(cfg, factory)
    cfg.device = "cuda:5"
    with pytest.raises(ValueError, match="not visible"):
        device_and_dtype(cfg)


def test_cpu_path_uses_real_accelerator_and_restores_environment(monkeypatch):
    monkeypatch.delenv("ACCELERATE_TORCH_DEVICE", raising=False)
    cfg = load_config(overrides=["device=cpu", "mixed_precision=no"])
    assert training_accelerator(cfg).device == torch.device("cpu")
    assert "ACCELERATE_TORCH_DEVICE" not in os.environ


def test_deepspeed_configuration_uses_requested_clipping_and_accumulation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("ACCELERATE_DEEPSPEED_CONFIG_FILE", "/unused/config.json")
    monkeypatch.setenv("ACCELERATE_CONFIG_DS_FIELDS", "zero_stage,gradient_clipping")
    captured = {}
    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(device=torch.device(os.environ["ACCELERATE_TORCH_DEVICE"]))
    cfg = load_config(overrides=["num_processes=2", "backend=deepspeed", "device=cuda",
                                 "max_grad_norm=0.25", "gradient_accumulation_steps=2"])
    training_accelerator(cfg, factory)
    plugin = captured["deepspeed_plugin"]
    plugin.set_mixed_precision("bf16")
    assert os.environ["ACCELERATE_CONFIG_DS_FIELDS"] == "zero_stage,gradient_clipping"
    assert os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] == "/unused/config.json"
    assert plugin.zero_stage == 1
    assert plugin.deepspeed_config["gradient_clipping"] == 0.25
    assert plugin.deepspeed_config["gradient_accumulation_steps"] == 2
    assert plugin.deepspeed_config["bf16"]["check_grad_overflow"] is True
    cfg.max_grad_norm = -1
    with pytest.raises(ValueError, match="max_grad_norm"):
        training_accelerator(cfg, factory)
