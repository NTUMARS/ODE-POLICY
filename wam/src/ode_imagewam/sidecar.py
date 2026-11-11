"""Content-bound, relocatable LIBERO virtual-target sidecars.

The ordered identities bind metadata and numeric parquet content, not absolute
filesystem locations. Images/videos are not hashed by this numeric-label audit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = "ode_imagewam_portable_sidecar_v1"
DEFAULT_GEOMETRY = {
    "canonical_quat": [0.99959104, -0.00150985, -0.02851977, 0.00144439],
    "position_scale": 0.05,
    "rotation_scale": 0.5,
    "control_hz": 20.0,
    "quaternion_order": "xyzw",
    "controller_frame": "world",
}
META_FILES = (
    "info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl",
    "modality.json", "stats.json",
)
ARRAY_NAMES = ("states", "commands", "references", "starts", "episode_ends")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_geometry(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError("Geometry must be a JSON object")
    unknown = set(value or {}) - set(DEFAULT_GEOMETRY)
    if unknown:
        raise ValueError(f"Unknown geometry keys: {sorted(unknown)}")
    geometry = {**DEFAULT_GEOMETRY, **(value or {})}
    q = np.asarray(geometry["canonical_quat"], dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) <= 0:
        raise ValueError("canonical_quat must contain four finite xyzw values with nonzero norm")
    for key in ("position_scale", "rotation_scale", "control_hz"):
        if not math.isfinite(float(geometry[key])) or float(geometry[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if geometry["quaternion_order"] != "xyzw" or geometry["controller_frame"] != "world":
        raise ValueError("This LIBERO adapter requires xyzw quaternions and world-frame commands")
    return geometry


def dataset_identity(root):
    root = Path(root).expanduser().resolve(strict=True)
    info = json.loads((root / "meta/info.json").read_text())
    paths = [root / "meta" / name for name in META_FILES if (root / "meta" / name).is_file()]
    parquet = sorted((root / "data").rglob("*.parquet"))
    if not parquet:
        raise ValueError(f"No numeric parquet episodes under {root / 'data'}")
    records = []
    for path in sorted(paths + parquet):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Dataset file escapes its selected root: {path}")
        records.append({"path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"files": records, "numeric_and_metadata_sha256": hashlib.sha256(encoded).hexdigest(),
            "frames": int(info["total_frames"]), "fps": float(info["fps"])}


def load_metadata(path):
    path = Path(path).expanduser()
    metadata = json.loads(path.read_text())
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Expected a portable ODE sidecar; rebuild legacy absolute-path sidecars with prepare")
    metadata["geometry"] = validate_geometry(metadata.get("geometry"))
    for key, width in (("reference_center", 6), ("reference_half_range", 6),
                       ("action_min", 7), ("action_max", 7)):
        value = np.asarray(metadata.get(key), dtype=np.float64)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid metadata field {key}: expected {width} finite values")
    if np.any(np.asarray(metadata["reference_half_range"]) <= 0):
        raise ValueError("Reference half ranges must be positive")
    if np.any(np.asarray(metadata["action_max"]) <= np.asarray(metadata["action_min"])):
        raise ValueError("Action normalization bounds must have positive ranges")
    if int(metadata.get("num_frames", 0)) <= 0 or not metadata.get("datasets"):
        raise ValueError("Empty sidecar")
    return metadata


class SidecarStore:
    def __init__(self, directory, dataset_dirs=None):
        self.root = Path(directory).expanduser().resolve()
        self.metadata = load_metadata(self.root / "metadata.json")
        if sha256_file(self.root / "dataset_stats.json") != self.metadata["normalization_stats_sha256"]:
            raise ValueError("Sidecar normalization statistics hash mismatch")
        if dataset_dirs is not None:
            expected = self.metadata["datasets"]
            if len(dataset_dirs) != len(expected):
                raise ValueError("Dataset count/order does not match sidecar manifest")
            for slot, (root, record) in enumerate(zip(dataset_dirs, expected)):
                current = dataset_identity(root)
                if current != record["identity"]:
                    raise ValueError(f"Dataset content or order mismatch at slot {slot}")
        for name in ARRAY_NAMES:
            entry = self.metadata["arrays"][name]
            path = self.root / f"{name}.npy"
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Sidecar content hash mismatch: {name}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
                raise ValueError(f"Sidecar shape/dtype mismatch: {name}")
            setattr(self, name, array)
        n = int(self.metadata["num_frames"])
        shapes = {"states": (n, 8), "commands": (n, 7), "references": (n, 6),
                  "starts": (n, 6), "episode_ends": (n,)}
        for name, expected_shape in shapes.items():
            if getattr(self, name).shape != expected_shape:
                raise ValueError(f"Invalid {name} dimensions")
        if not np.issubdtype(self.episode_ends.dtype, np.integer):
            raise ValueError("Episode ends must be integer indices")
        previous = 0
        for record in self.metadata["episodes"]:
            start, end = int(record["start"]), int(record["end"])
            if start != previous or end <= start or end > n:
                raise ValueError("Noncontiguous or invalid episode manifest")
            if not np.all(self.episode_ends[start:end] == end):
                raise ValueError("Episode-end array disagrees with episode manifest")
            previous = end
        if previous != n:
            raise ValueError("Episode manifest does not cover the sidecar")

    def window(self, index, horizon):
        index, horizon = int(index), int(horizon)
        if not 0 <= index < len(self.states) or horizon < 1:
            raise ValueError("Invalid sidecar index/horizon")
        positions = index + np.arange(horizon)
        end = int(self.episode_ends[index])
        valid = positions < end
        indices = np.minimum(positions, end - 1)
        return {"indices": indices, "valid": valid,
                "start": np.array(self.starts[index], copy=True),
                "targets": np.array(self.references[indices], copy=True),
                "raw_state": np.array(self.states[index], copy=True),
                "commands": np.array(self.commands[indices], copy=True)}


def _action_bounds(stats):
    record = stats["action"]["default"]
    low = record.get("global_min", record.get("min"))
    high = record.get("global_max", record.get("max"))
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    if low.shape != (7,) or high.shape != (7,) or not np.isfinite([low, high]).all():
        raise ValueError("Normalization statistics must contain seven finite global action bounds")
    if np.any(high < low):
        raise ValueError("Action maximum cannot be smaller than minimum")
    # The upstream constant-channel min/max normalizer substitutes range=2.
    constant = high - low < 1e-4
    high = np.where(constant, low + 1.0, high)
    low = np.where(constant, low - 1.0, low)
    return low, high


def prepare(dataset_dirs, normalization_stats, output_dir, geometry=None, threads=4):
    import pyarrow.parquet as pq
    import torch
    from .head import command_to_reference, reference_to_command, state_to_reference

    geometry = validate_geometry(geometry)
    torch.set_num_threads(int(threads))
    roots = [Path(root).expanduser().resolve(strict=True) for root in dataset_dirs]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Provide a nonempty ordered list of distinct dataset directories")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite sidecar output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ode-sidecar-", dir=output.parent))
    try:
        arrays = {key: [] for key in ARRAY_NAMES}
        datasets, episodes = [], []
        offset, worst = 0, 0.0
        for slot, root in enumerate(roots):
            identity = dataset_identity(root)
            if not np.isclose(identity["fps"], geometry["control_hz"]):
                raise ValueError(f"Dataset slot {slot} fps differs from configured control_hz")
            dataset_frames = 0
            for path in sorted((root / "data").rglob("*.parquet")):
                table = pq.read_table(path, columns=["index", "frame_index", "episode_index", "observation.state", "action"])
                size = len(table)
                index = np.asarray(table["index"].to_numpy())
                frame = np.asarray(table["frame_index"].to_numpy())
                episode = np.asarray(table["episode_index"].to_numpy())
                if size == 0 or not np.array_equal(index, np.arange(dataset_frames, dataset_frames + size)):
                    raise ValueError(f"Expected contiguous dataset-local indices: {path}")
                if not np.array_equal(frame, np.arange(size)) or len(np.unique(episode)) != 1:
                    raise ValueError(f"Expected one complete ordered episode per parquet: {path}")
                state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
                action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
                if state.shape != (size, 8) or action.shape != (size, 7):
                    raise ValueError(f"Expected state8/action7 numeric schema: {path}")
                if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isin(action[:, 6], [0, 1]).all():
                    raise ValueError(f"Invalid numeric data or gripper encoding: {path}")
                ts, ta = torch.from_numpy(state), torch.from_numpy(action)
                reference = command_to_reference(ts, ta, geometry)
                reconstructed = reference_to_command(reference.float().double(), ts, geometry)
                worst = max(worst, float((reconstructed - ta[:, :6]).abs().max()))
                starts = torch.cat((state_to_reference(ts[:1], geometry), reference[:-1]), 0)
                for key, value in (("states", state), ("commands", action),
                                   ("references", reference.numpy()), ("starts", starts.numpy())):
                    arrays[key].append(value.astype(np.float32))
                arrays["episode_ends"].append(np.full(size, offset + size, dtype=np.int64))
                episodes.append({"dataset_slot": slot, "episode_index": int(episode[0]),
                                 "parquet": path.relative_to(root).as_posix(), "start": offset, "end": offset + size})
                offset += size
                dataset_frames += size
            if dataset_frames != identity["frames"]:
                raise ValueError(f"Dataset frame count mismatch at slot {slot}")
            datasets.append({"slot": slot, "identity": identity})
        if worst > 1e-5:
            raise ValueError(f"Reference-to-command roundtrip failed: {worst}")
        arrays = {key: np.concatenate(value) for key, value in arrays.items()}
        records = {}
        for key, value in arrays.items():
            file = stage / f"{key}.npy"
            np.save(file, value, allow_pickle=False)
            records[key] = {"bytes": file.stat().st_size, "sha256": sha256_file(file),
                            "shape": list(value.shape), "dtype": str(value.dtype)}
        if normalization_stats:
            stats_path = Path(normalization_stats).expanduser().resolve(strict=True)
            stats = json.loads(stats_path.read_text())
            shutil.copyfile(stats_path, stage / "dataset_stats.json")
        else:
            stats = {name: {"default": {"global_min": arrays[source].min(0).tolist(),
                                       "global_max": arrays[source].max(0).tolist()}}
                     for name, source in (("state", "states"), ("action", "commands"))}
            (stage / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        low, high = _action_bounds(stats)
        references = np.concatenate((arrays["references"], arrays["starts"]))
        rmin, rmax = references.min(0), references.max(0)
        metadata = {"schema": SCHEMA, "num_frames": offset, "num_episodes": len(episodes),
                    "datasets": datasets, "episodes": episodes, "arrays": records,
                    "geometry": geometry, "reference_center": ((rmin + rmax) / 2).tolist(),
                    "reference_half_range": np.maximum((rmax - rmin) / 2, 1e-4).tolist(),
                    "action_min": low.tolist(), "action_max": high.tolist(),
                    "normalization_stats_sha256": sha256_file(stage / "dataset_stats.json"),
                    "normalization_stats_origin": "user_provided" if normalization_stats else "computed_minmax_over_selected_numeric_training_frames",
                    "roundtrip_max_abs_error": worst,
                    "initial_reference": "previous_command_target_or_measured_pose_at_episode_start",
                    "dataset_identity_scope": "ordered_metadata_and_numeric_parquet_content_not_video_bytes"}
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        load_metadata(stage / "metadata.json")
        # Recheck only source identities here: arrays were just generated,
        # hashed and shape-checked above. Detect mutation during preparation.
        if any(dataset_identity(root) != record["identity"] for root, record in zip(roots, datasets)):
            raise ValueError("Dataset changed while preparing its ODE sidecar")
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        os.rename(stage, output)
        return metadata
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
