# Adapted for ODE-POLICY; see THIRD_PARTY_NOTICES.md.
"""Frozen, independently encoded camera observations for the G1 ODE integration.

This module receives the actual loaded GR00T N1.7 model. It never creates a
substitute backbone or downloads weights. Each time step is encoded separately
so the current observation's representation cannot attend to future frames.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import torch


class FeatureBatch(dict):
    """Small tensor mapping matching the cached head API without importing Transformers."""

    def to(self, device):
        return FeatureBatch({key: value.to(device) for key, value in self.items()})


def read_video_frame(path: str | Path, frame_index: int) -> np.ndarray:
    from .video import VideoFrameReader

    return VideoFrameReader().read(path, frame_index)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_condition(condition: dict):
    return FeatureBatch(
        {
            key: (value.detach().cpu().float() if key == "backbone_features" else value.detach().cpu()).clone()
            for key, value in condition.items()
        }
    )


class FrozenFrameEncoder:
    """Encode both cameras with the supplied N1.7 backbone and a CPU LRU.

    Image preprocessing uses the installed GR00T evaluation transform and the
    supplied model's configuration. Task language follows GR00T's configured
    lowercasing and punctuation removal. No state or action enters the VLM.
    """

    def __init__(
        self,
        model,
        dataset_root: str | Path,
        device,
        max_cache: int = 32,
        *,
        eval_transform=None,
        frame_reader=None,
        video_path_resolver=None,
        video_fingerprints=None,
        feature_batch_size: int = 16,
        feature_cache_dir=None,
        feature_cache_identity=None,
        disk_reserve_bytes: int = 0,
        camera_keys=("observation.images.ego_view", "observation.images.chest"),
        require_cache_hits=False,
    ):
        if max_cache < 1:
            raise ValueError("max_cache must be positive")
        self.camera_keys = tuple(camera_keys)
        self.require_cache_hits = bool(require_cache_hits)
        self.model = model
        self.dataset_root = Path(dataset_root).resolve()
        self.device = torch.device(device)
        self.max_cache = max_cache
        if feature_batch_size < 1:
            raise ValueError("feature_batch_size must be positive")
        self.frame_reader = frame_reader
        self.video_path_resolver = video_path_resolver
        self.video_fingerprints = video_fingerprints or {}
        self.feature_batch_size = feature_batch_size
        config = model.config
        self.formalize_language = bool(config.formalize_language)
        self._transform_parameters = {
            name: getattr(config, name)
            for name in (
                "image_target_size",
                "image_crop_size",
                "shortest_image_edge",
                "crop_fraction",
            )
        }
        self._transform_metadata = {
            "parameters": self._transform_parameters,
            "deterministic": True,
            "formalize_language": self.formalize_language,
        }
        if eval_transform is None:
            from gr00t.model.gr00t_n1d7.image_augmentations import (
                build_image_transformations_albumentations,
            )

            if not getattr(config, "use_albumentations_transforms", True):
                raise ValueError("This encoder requires the configured GR00T Albumentations transform")
            _, eval_transform = build_image_transformations_albumentations(
                **self._transform_parameters,
                random_rotation_angle=getattr(config, "random_rotation_angle", None),
                color_jitter_params=getattr(config, "color_jitter_params", None),
                extra_augmentation_config=getattr(config, "extra_augmentation_config", None),
            )
            source = Path(inspect.getsourcefile(build_image_transformations_albumentations))
            self._transform_metadata.update(
                {
                    "api": "gr00t.model.gr00t_n1d7.image_augmentations.build_image_transformations_albumentations",
                    "upstream_source_sha256": _sha256(source),
                    "operations": [
                        "LetterBoxPad",
                        "SmallestMaxSize",
                        "FractionalCenterCrop",
                        "SmallestMaxSize",
                    ],
                    "interpolation": "cv2.INTER_AREA",
                }
            )
        else:
            self._transform_metadata["api"] = "explicitly_injected_eval_transform"
        self.eval_transform = eval_transform
        self._cache: OrderedDict[tuple[int, int], dict] = OrderedDict()
        self._task_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._source_sha256: dict[str, str] = {}
        info_path = self.dataset_root / "meta/info.json"
        tasks_path = self.dataset_root / "meta/tasks.jsonl"
        self.info = json.loads(info_path.read_text())
        self.tasks = {
            item["task_index"]: item["task"]
            for line in tasks_path.read_text().splitlines()
            if line.strip()
            for item in [json.loads(line)]
        }
        self._remember_source(info_path)
        self._remember_source(tasks_path)
        self.disk_store = None
        self.cache_counts = {"ram_hits": 0, "disk_hits": 0, "encoded_frames": 0}
        if feature_cache_dir is not None:
            from .cache import FrozenFeatureStore

            if not feature_cache_identity:
                raise ValueError("Persistent frozen features require an explicit dataset/model identity")
            directory = Path(feature_cache_dir).resolve()
            if directory.is_relative_to(self.dataset_root):
                raise ValueError("Feature cache must be outside the source numeric dataset")
            self.disk_store = FrozenFeatureStore(
                directory,
                {"inputs": feature_cache_identity, "transform": self._transform_metadata},
                reserve_bytes=disk_reserve_bytes,
            )
        self.model.backbone.requires_grad_(False)
        self.model.backbone.eval()

    @property
    def provenance(self) -> dict:
        return {
            "encoder": "supplied_gr00t_n1d7_backbone",
            "cameras": list(self.camera_keys),
            "transform": dict(self._transform_metadata),
            "independent_frame_encoding": True,
            "source_sha256": dict(self._source_sha256),
            "persistent_feature_cache": None
            if self.disk_store is None
            else {
                "identity_sha256": self.disk_store.identity_sha256,
                "cache_directory": str(self.disk_store.directory),
                "require_existing_cache_hits": self.require_cache_hits,
                "counts": dict(self.cache_counts),
            },
        }

    def _remember_source(self, path: Path) -> None:
        key = str(path.relative_to(self.dataset_root)) if path.is_relative_to(self.dataset_root) else str(path)
        if key not in self._source_sha256:
            fingerprint = self.video_fingerprints.get(str(path))
            self._source_sha256[key] = fingerprint["sha256"] if fingerprint else _sha256(path)

    def _task_text(self, episode_id: int, frame_index: int) -> str:
        if episode_id not in self._task_cache:
            import pyarrow.parquet as pq

            path = self.dataset_root / self.info["data_path"].format(
                episode_chunk=episode_id // self.info["chunks_size"],
                episode_index=episode_id,
            )
            task_ids = np.asarray(pq.read_table(path, columns=["task_index"], use_threads=False)["task_index"])
            self._remember_source(path)
            self._task_cache[episode_id] = task_ids
            if len(self._task_cache) > self.max_cache:
                self._task_cache.popitem(last=False)
        self._task_cache.move_to_end(episode_id)
        task_ids = self._task_cache[episode_id]
        if not 0 <= frame_index < len(task_ids):
            raise IndexError("Frame index is outside the selected episode")
        task = self.tasks[int(task_ids[frame_index])]
        return re.sub(r"[^\w\s]", "", task.lower()) if self.formalize_language else task

    def _image(self, episode_id: int, frame_index: int, camera: str):
        from PIL import Image

        path = (
            self.video_path_resolver(episode_id, camera)
            if self.video_path_resolver
            else self.dataset_root
            / self.info["video_path"].format(
                episode_chunk=episode_id // self.info["chunks_size"],
                episode_index=episode_id,
                video_key=camera,
            )
        )
        self._remember_source(path)
        raw = (
            self.frame_reader.read(path, frame_index)
            if self.frame_reader is not None
            else read_video_frame(path, frame_index)
        )
        image = self.eval_transform(image=raw)["image"]
        # Match upstream apply_with_replay's evaluation conversion to uint8.
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise ValueError(f"Unexpected transformed image dtype: {image.dtype}")
        return Image.fromarray(image)

    def _example(self, episode_id, frame_index):
        task = self._task_text(episode_id, frame_index)
        images = [self._image(episode_id, frame_index, camera) for camera in self.camera_keys]
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": task},
                ],
            }
        ]
        text = self.model.collator.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        return {"vlm_content": {"text": text, "images": images, "conversation": conversation}}

    def encode_frames(self, requests):
        """Batch independent VLM observations and retain a bounded CPU cache.

        Padded tokens are removed per sample before caching, preserving the
        single-frame head contract. Cache entries use FP16; returned features
        use FP32 as expected by the trainable action head.
        """
        requests = [(int(ep), int(frame)) for ep, frame in requests]
        if any(ep < 0 or frame < 0 for ep, frame in requests):
            raise ValueError("Episode and frame indices must be nonnegative")
        unique = list(dict.fromkeys(requests))
        available = {key: self._cache[key] for key in unique if key in self._cache}
        self.cache_counts["ram_hits"] += len(available)
        if self.disk_store is not None:
            for key in unique:
                if key not in available:
                    condition = self.disk_store.load(key)
                    if condition is not None:
                        available[key] = condition
                        self._cache[key] = condition
                        self._cache.move_to_end(key)
                        while len(self._cache) > self.max_cache:
                            self._cache.popitem(last=False)
                        self.cache_counts["disk_hits"] += 1
        missing = [key for key in unique if key not in available]
        if missing and self.require_cache_hits:
            raise RuntimeError(
                "Required frozen feature-cache hit is missing; encoding and cache writes are disabled "
                f"({len(missing)} missing keys; first={missing[0]})"
            )
        parameter = next(self.model.backbone.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.float32
        self.model.backbone.eval()
        for start in range(0, len(missing), self.feature_batch_size):
            keys = missing[start : start + self.feature_batch_size]
            inputs = self.model.collator([self._example(*key) for key in keys])["inputs"]
            prepared = {
                k: value.to(self.device, dtype=dtype if value.is_floating_point() else value.dtype)
                for k, value in inputs.items()
            }
            with torch.no_grad():
                encoded = self.model.backbone(self.model.backbone.prepare_input(prepared))
            features = encoded["backbone_features"]
            if features.ndim != 3 or features.shape[0] != len(keys) or not torch.isfinite(features).all():
                raise ValueError("Invalid batched backbone features")
            for b, key in enumerate(keys):
                valid = encoded["backbone_attention_mask"][b].bool()
                image = encoded["image_mask"][b].bool()
                if not (valid & image).any():
                    raise ValueError("Frame has no valid image tokens")
                condition = FeatureBatch(
                    {
                        "backbone_features": features[b : b + 1, valid].detach().cpu().to(torch.float16).clone(),
                        "backbone_attention_mask": torch.ones((1, int(valid.sum())), dtype=torch.bool),
                        "image_mask": image[None, valid].detach().cpu().clone(),
                    }
                )
                if self.disk_store is not None:
                    self.disk_store.save(key, condition)
                self.cache_counts["encoded_frames"] += 1
                available[key] = condition
                self._cache[key] = condition
                self._cache.move_to_end(key)
                while len(self._cache) > self.max_cache:
                    self._cache.popitem(last=False)
        for key in unique:
            if key in self._cache:
                self._cache.move_to_end(key)
        return [_copy_condition(available[key]) for key in requests]

    def encode_frame(self, episode_id: int, frame_index: int):
        """Return one independently encoded observation without future inputs."""
        return self.encode_frames([(episode_id, frame_index)])[0]


def combine_conditions(conditions: Sequence[dict]):
    """Pad single-sample token conditions to one batch, masking padded slots."""
    if not conditions:
        raise ValueError("Cannot combine an empty condition batch")
    first = conditions[0]["backbone_features"]
    if first.ndim != 3 or first.shape[0] != 1:
        raise ValueError("Each condition must contain one sample")
    max_tokens = max(condition["backbone_features"].shape[1] for condition in conditions)
    batch_size, feature_dim = len(conditions), first.shape[-1]
    result = {
        "backbone_features": first.new_zeros(batch_size, max_tokens, feature_dim),
        "backbone_attention_mask": torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=first.device),
        "image_mask": torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=first.device),
        "frame_ids": torch.zeros(batch_size, max_tokens, dtype=torch.long, device=first.device),
    }
    for row, condition in enumerate(conditions):
        features = condition["backbone_features"]
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[-1] != feature_dim:
            raise ValueError("Conditions must have matching feature dimensions and batch size one")
        count = features.shape[1]
        result["backbone_features"][row, :count] = features[0]
        for name in ("backbone_attention_mask", "image_mask", "frame_ids"):
            values = condition.get(name, torch.zeros((1, count), dtype=torch.long, device=features.device))
            if values.shape != (1, count):
                raise ValueError(f"{name} does not match its condition token dimensions")
            result[name][row, :count] = values[0]
    return FeatureBatch(result)
