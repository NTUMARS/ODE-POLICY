# Adapted for ODE-POLICY public integration; see THIRD_PARTY_NOTICES.md.
"""Persistent independently encoded frozen observations with strict provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


class FrozenFeatureStore:
    """Atomic FP16 frame files; a mismatch is an error, never a silent miss."""

    def __init__(self, directory, identity, *, reserve_bytes=0):
        self.directory = Path(directory).resolve()
        self.reserve_bytes = int(reserve_bytes)
        if not identity or self.reserve_bytes < 0:
            raise ValueError("A nonempty feature identity and nonnegative reserve are required")
        serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        self.identity = json.loads(serialized)
        self.identity_sha256 = hashlib.sha256(serialized.encode()).hexdigest()
        metadata = {
            "schema_version": 1,
            "identity_sha256": self.identity_sha256,
            "identity": identity,
        }
        manifest = self.directory / "cache_identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != metadata:
                raise ValueError("Frozen feature cache identity does not match this dataset/model/transform")
        else:
            if self.directory.exists() and any(self.directory.iterdir()):
                raise ValueError("Nonempty feature directory has no cache identity; inspect it first")
            self.directory.mkdir(parents=True, exist_ok=True)
            self._atomic_json(manifest, metadata)

    @staticmethod
    def _atomic_json(path, value):
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        temporary.replace(path)

    def path(self, key):
        episode, frame = map(int, key)
        if episode < 0 or frame < 0:
            raise ValueError("Feature episode and frame must be nonnegative")
        return self.directory / f"episode_{episode:06d}" / f"frame_{frame:06d}.safetensors"

    def contains(self, key):
        return self.path(key).is_file()

    def load(self, key):
        path = self.path(key)
        if not path.exists():
            return None
        expected = {
            "schema_version": "1",
            "identity_sha256": self.identity_sha256,
            "episode_id": str(int(key[0])),
            "frame_index": str(int(key[1])),
        }
        with safe_open(path, framework="pt", device="cpu") as stream:
            if stream.metadata() != expected:
                raise ValueError(f"Frozen frame cache metadata mismatch: {path}")
        condition = load_file(path, device="cpu")
        self._validate(condition)
        return condition

    @staticmethod
    def _validate(condition):
        if set(condition) != {"backbone_features", "backbone_attention_mask", "image_mask"}:
            raise ValueError("Frozen frame cache has unexpected tensor fields")
        features = condition["backbone_features"]
        if features.ndim != 3 or features.shape[0] != 1 or features.dtype != torch.float16:
            raise ValueError("Frozen cached features must be single-frame FP16 tokens")
        if not torch.isfinite(features).all():
            raise ValueError("Frozen cached features contain nonfinite values")
        for name in ("backbone_attention_mask", "image_mask"):
            if condition[name].dtype != torch.bool or condition[name].shape != features.shape[:2]:
                raise ValueError("Frozen frame masks must be Boolean and align with tokens")
        if not (condition["backbone_attention_mask"] & condition["image_mask"]).any():
            raise ValueError("Frozen frame cache contains no valid image tokens")

    def save(self, key, condition):
        self._validate(condition)
        path = self.path(key)
        if self.contains(key):
            self.load(key)  # Verify an existing frame instead of replacing its identity.
            return
        required = sum(value.numel() * value.element_size() for value in condition.values()) + 4096
        if shutil.disk_usage(self.directory).free < self.reserve_bytes + required:
            raise OSError("Insufficient disk space for frozen features and reserved checkpoints")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        metadata = {
            "schema_version": "1",
            "identity_sha256": self.identity_sha256,
            "episode_id": str(int(key[0])),
            "frame_index": str(int(key[1])),
        }
        try:
            save_file(
                {name: value.detach().cpu().contiguous() for name, value in condition.items()},
                temporary,
                metadata=metadata,
            )
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
