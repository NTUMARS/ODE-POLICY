"""Small, portable data/method bindings stored alongside model tensors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def make_binding(metadata, model_config):
    runtime_keys = {"_target_", "device", "model_dtype", "load_text_encoder", "flux2_src_path",
                    "flux2_model_path", "ae_model_path", "qwen3_model_spec", "action_dit_pretrained_path"}
    config = {key: model_config[key] for key in model_config.keys() if key not in runtime_keys}
    ode = config["ode"]
    config["ode"] = {key: ode[key] for key in ode.keys() if key not in ("sidecar_dir", "metadata_path")}
    config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    return {"schema": "ode_imagewam_binding_v1", "sidecar_sha256": _digest(metadata),
            "model_config_sha256": _digest(config)}


def require_binding(actual, expected):
    if expected is None or actual != expected:
        raise ValueError("Checkpoint data/method binding is missing or differs from this sidecar/config")


def read_weights(path, expected_binding):
    import torch
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Our weights use torch's ZIP format; mmap checks the tiny binding before
    # allocating a backbone and avoids a second full checkpoint read.
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("mot"), dict):
        raise ValueError("Expected a complete ImageWAM weights checkpoint with a mot state_dict")
    require_binding(payload.get("ode_binding"), expected_binding)
    return payload


def strict_load_checkpoint(model, path, *, payload=None, optimizer=None):
    expected = getattr(model, "ode_binding", None)
    payload = read_weights(path, expected) if payload is None else payload
    require_binding(payload.get("ode_binding"), expected)
    model.mot.load_state_dict(payload["mot"], strict=True)
    encoder = getattr(model, "proprio_encoder", None)
    if encoder is not None:
        if "proprio_encoder" not in payload:
            raise ValueError("Checkpoint is missing its trained proprio encoder")
        encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    elif payload.get("proprio_encoder"):
        raise ValueError("Checkpoint/config proprio encoder mismatch")
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("step")


def save_weights(model, path, step=None, optimizer=None):
    import torch
    binding = getattr(model, "ode_binding", None)
    if binding is None:
        raise ValueError("Cannot save ODE weights without a sidecar/config binding")
    payload = {"mot": model.mot.state_dict(), "step": step, "ode_binding": binding,
               "torch_dtype": str(getattr(model, "torch_dtype", torch.float32))}
    if getattr(model, "proprio_encoder", None) is not None:
        payload["proprio_encoder"] = model.proprio_encoder.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
