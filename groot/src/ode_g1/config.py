"""YAML/CLI configuration with paths relative to the YAML file, never a lab checkout."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import os
import re

DEFAULTS = {
    "device": "cuda",
    "paths": {
        "dataset_root": "data/task",
        "prepared": "runs/task/prepared",
        "feature_cache": "runs/task/features",
        "output": "runs/task/train",
    },
    "dataset": {
        "name": "custom-task",
        "instruction": None,
        "repo_id": "local",
        "revision": None,
        "expected_episodes": None,
        "expected_frames": None,
        "camera_keys": ["observation.images.ego_view", "observation.images.chest"],
        "metadata_sha256": None,
    },
    "base_model": {
        "path": None,
        "revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
        "cosmos_revision": "9ce19a195e423419c349abfc86fd07178b230561",
        "cosmos_path": None,
        "allow_download": False,
        "channels_last_3d": True,
    },
    "head": {
        "horizon": 16,
        "body_dim": 64,
        "hand_dim": 12,
        "state_dim": 44,
        "feature_dim": 1024,
        "cond_dim": 512,
        "width": 512,
        "layers": 4,
        "field_scale": 1.0,
    },
    "features": {"batch_size": 16, "max_ram_frames": 32, "disk_reserve_gb": 0, "require_cache_hits": True},
    "training": {
        "epochs": 90,
        "batch_size": 112,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "warmup_ratio": 0.02,
        "min_lr_ratio": 0.1,
        "max_grad_norm": 1.0,
        "seed": 17,
        "log_every_steps": 10,
        "checkpoint_every_epochs": 5,
        "checkpoint_milestone_epochs": [60, 70, 80, 90],
        "autocast_bfloat16": True,
        "max_train_seconds": None,
        "resume": None,
    },
    "inference": {"checkpoint_dir": None, "verify_head_sha256": False},
    "deployment": {
        "command_endpoint": None,
        "state_endpoint": None,
        "camera_host": None,
        "camera_port": 60000,
        "camera_sources": ["head", "chest"],
        "network_interface": None,
        "dds_domain": 0,
        "head_source": "full",
        "chest_rotate180": True,
        "max_sensor_age_s": 0.25,
        "max_camera_skew_s": 0.1,
        "hand_speed": 2.0,
        "initialization_s": 3.0,
        "max_inference_s": 0.5,
    },
}


def validate_deployment_config(cfg):
    settings = cfg["deployment"]
    for key in ("command_endpoint", "state_endpoint", "camera_host", "network_interface"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"Set deployment.{key} for this robot")
    for key in ("command_endpoint", "state_endpoint"):
        if not settings[key].startswith("tcp://"):
            raise ValueError(f"deployment.{key} must be a tcp:// endpoint")
    if settings["command_endpoint"] == settings["state_endpoint"]:
        raise ValueError("Command and state channels must use different endpoints")
    for key in ("camera_port", "dds_domain"):
        if type(settings[key]) is not int or not (0 <= settings[key] <= 65535):
            raise ValueError(f"Invalid deployment.{key}")
    if settings["camera_port"] == 0 or settings["head_source"] not in ("full", "left", "right"):
        raise ValueError("Invalid camera port/head source")
    if not isinstance(settings["camera_sources"], list) or sorted(settings["camera_sources"]) != ["chest", "head"]:
        raise ValueError("camera_sources must contain head and chest in the dataset camera-key order")
    if type(settings["chest_rotate180"]) is not bool:
        raise ValueError("chest_rotate180 must be a YAML boolean")
    for key in ("max_sensor_age_s", "max_camera_skew_s", "hand_speed", "initialization_s", "max_inference_s"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"deployment.{key} must be finite and positive")
    if settings["max_inference_s"] >= cfg["head"]["horizon"] / 30:
        raise ValueError("Inference timeout must be shorter than the saved control horizon")
    return settings


def _merge(target, updates, prefix=""):
    if not isinstance(updates, dict):
        raise ValueError(f"Configuration {prefix or 'root'} must be a mapping")
    for key, value in updates.items():
        if key not in target:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(target[key], dict) and target[key] is not None:
            _merge(target[key], value, prefix + key + ".")
        else:
            target[key] = value


def _expand(value):
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if re.search(r"\$\{[^}]+\}", expanded):
            raise ValueError(f"Unresolved environment variable in configuration: {value}")
        return expanded
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path, overrides=()):
    import yaml

    path = Path(path).expanduser().resolve()
    cfg = deepcopy(DEFAULTS)
    raw = yaml.safe_load(path.read_text()) or {}
    _merge(cfg, raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError("Overrides must be dotted.key=YAML_VALUE")
        key, value = item.split("=", 1)
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown override: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"Unknown override: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    cfg = _expand(cfg)
    for key, value in cfg["paths"].items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"paths.{key} must be a nonempty path")
        p = Path(value)
        cfg["paths"][key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    for section, key in [("training", "resume"), ("inference", "checkpoint_dir"), ("base_model", "cosmos_path")]:
        value = cfg[section][key]
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{section}.{key} must be null or a nonempty local path")
            p = Path(value)
            cfg[section][key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    base = cfg["base_model"]["path"]
    if base is not None:
        if not isinstance(base, str) or not base:
            raise ValueError("base_model.path must be a local path or model ID")
        # Relative local paths are unambiguous with ./ or ../; owner/model IDs remain IDs.
        if base.startswith((".", os.sep)) or (path.parent / base).is_dir():
            cfg["base_model"]["path"] = str((path.parent / base).resolve())
    if not isinstance(cfg["device"], str) or not cfg["device"]:
        raise ValueError("Nonempty device required")
    head = cfg["head"]
    if type(head["horizon"]) is not int or not 1 <= head["horizon"] <= 16:
        raise ValueError("H16 schema requires horizon in 1..16")
    if (head["body_dim"], head["hand_dim"], head["state_dim"], head["layers"], head["field_scale"]) != (
        64,
        12,
        44,
        4,
        1.0,
    ):
        raise ValueError("Checkpoint schema is body64/hands12/state44/four FiLM layers/dt scale1")
    cameras = cfg["dataset"]["camera_keys"]
    if (
        not isinstance(cameras, list)
        or len(cameras) != 2
        or any(not isinstance(k, str) or not k for k in cameras)
        or len(set(cameras)) != 2
    ):
        raise ValueError("Two distinct ordered camera keys required")
    for section, key in [
        ("base_model", "allow_download"),
        ("base_model", "channels_last_3d"),
        ("features", "require_cache_hits"),
        ("training", "autocast_bfloat16"),
        ("inference", "verify_head_sha256"),
    ]:
        if type(cfg[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be a YAML boolean")
    for section, keys in [
        ("features", ["batch_size", "max_ram_frames"]),
        ("training", ["epochs", "batch_size", "log_every_steps", "checkpoint_every_epochs"]),
        ("head", ["feature_dim", "cond_dim", "width"]),
    ]:
        for key in keys:
            if type(cfg[section][key]) is not int or cfg[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for key in ["learning_rate", "max_grad_norm"]:
        value = cfg["training"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    for section, key in [("features", "disk_reserve_gb"), ("training", "weight_decay")]:
        value = cfg[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{section}.{key} must be finite and nonnegative")
    for key, upper, inclusive in [("warmup_ratio", 1, False), ("min_lr_ratio", 1, True)]:
        value = cfg["training"][key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (value > upper if inclusive else value >= upper)
        ):
            raise ValueError(f"Invalid training.{key}")
    milestones = cfg["training"]["checkpoint_milestone_epochs"]
    if (
        not isinstance(milestones, list)
        or any(type(x) is not int or x < 1 for x in milestones)
        or milestones != sorted(set(milestones))
    ):
        raise ValueError("Milestones must be sorted unique positive epochs")
    seconds = cfg["training"]["max_train_seconds"]
    if seconds is not None and (
        isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0
    ):
        raise ValueError("max_train_seconds must be null or finite and positive")
    if type(cfg["training"]["seed"]) is not int or cfg["training"]["seed"] < 0:
        raise ValueError("training.seed must be a nonnegative integer")
    return cfg
