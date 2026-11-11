"""Portable LeRobot v2.1 single-task data for the G1 SONIC64 + measured DFX6+6 schema.

Only NumPy is imported until Parquet is read. Source data is never modified.
Architecture/channel semantics are explicit; dataset locations, counts, language,
camera names and storage templates come from user configuration and metadata.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

SCHEMA = "ode-policy-g1-task-v1"
NORMALIZATION_SCHEMA = "g1-task-ode-q01q99-fixed-hands-v1"
CAMERAS = ("observation.images.ego_view", "observation.images.chest")
HAND_ORDER = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation")
NUMERIC_COLUMNS = (
    "observation.state",
    "observation.inspire_left_hand_state",
    "observation.inspire_right_hand_state",
    "observation.projected_gravity",
    "action.motion_token",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    repo_id: str
    revision: str
    episodes: int
    frames: int
    instruction: str
    metadata_sha256: Mapping[str, str]
    fps: int = 30
    camera_keys: tuple[str, str] = CAMERAS

    def __post_init__(self):
        object.__setattr__(self, "camera_keys", tuple(self.camera_keys))
        if not self.name or not self.instruction or not self.revision:
            raise ValueError("Task name, instruction and revision/content identity are required")
        if any(type(getattr(self, k)) is not int or getattr(self, k) <= 0 for k in ("episodes", "frames", "fps")):
            raise ValueError("Task counts and FPS must be positive integers")
        if len(self.camera_keys) != 2 or len(set(self.camera_keys)) != 2:
            raise ValueError("This G1 schema requires two distinct ordered camera keys")


def get_task(task):
    if not isinstance(task, TaskSpec):
        raise TypeError("Pass an explicit TaskSpec; no machine-specific task registry is installed")
    return task


def _sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            d.update(block)
    return d.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _document(spec):
    return json.loads(json.dumps(asdict(spec)))


def _write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def _inside(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    path = (root / rel).resolve()
    if rel.is_absolute() or ".." in rel.parts or not path.is_relative_to(root):
        raise ValueError("Dataset reference escapes its root")
    return path


def file_fingerprint(path, *, full_hash=False):
    """Content identity plus race-detection provenance; portable across mount/inode changes."""
    p = Path(path)
    a = p.stat()
    d = hashlib.sha256()
    block = 128 * 1024
    with p.open("rb") as f:
        if full_hash:
            for chunk in iter(lambda: f.read(4 * 1024**2), b""):
                d.update(chunk)
        else:
            d.update(f.read(block))
            if a.st_size > block:
                f.seek(max(block, a.st_size - block))
                d.update(f.read(block))
    z = p.stat()
    if (a.st_ino, a.st_size, a.st_mtime_ns, a.st_ctime_ns) != (z.st_ino, z.st_size, z.st_mtime_ns, z.st_ctime_ns):
        raise RuntimeError(f"Source changed while fingerprinting: {p}")
    return {
        "size_bytes": a.st_size,
        "sha256": d.hexdigest(),
        "hash_mode": "full_sha256" if full_hash else "head_tail_sha256",
        "mtime_ns": a.st_mtime_ns,
        "ctime_ns": a.st_ctime_ns,
        "inode": a.st_ino,
        "device": a.st_dev,
    }


def _verify_source(path, expected):
    actual = file_fingerprint(path, full_hash=expected["hash_mode"] == "full_sha256")
    if any(actual[k] != expected[k] for k in ("size_bytes", "sha256", "hash_mode")):
        raise RuntimeError(f"Task source content changed: {path}")
    return actual


def _jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def task_spec_from_root(
    root,
    *,
    name,
    instruction=None,
    repo_id="local",
    revision=None,
    expected_episodes=None,
    expected_frames=None,
    camera_keys=CAMERAS,
    metadata_sha256=None,
):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    tasks = _jsonl(root / "meta/tasks.jsonl")
    if len(tasks) != 1 or tasks[0].get("task_index") != 0:
        raise ValueError("Use one task per G1 training run, with task_index=0")
    language = tasks[0]["task"] if instruction is None else instruction
    for expected, actual in [(expected_episodes, info["total_episodes"]), (expected_frames, info["total_frames"])]:
        if expected is not None and expected != actual:
            raise ValueError("Configured dataset budget differs from metadata")
    paths = ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl", "meta/episodes_stats.jsonl"]
    if (root / "meta/modality.json").is_file():
        paths.append("meta/modality.json")
    pins = {p: _sha(_inside(root, p)) for p in paths}
    if metadata_sha256 is not None and any(pins.get(k) != v for k, v in metadata_sha256.items()):
        raise ValueError("Configured metadata content pins differ")
    return TaskSpec(
        name,
        repo_id,
        revision or _digest(pins),
        info["total_episodes"],
        info["total_frames"],
        language,
        pins,
        info["fps"],
        tuple(camera_keys),
    )


def _relative(info, kind, episode_id, camera=None):
    template = info["data_path" if kind == "parquet" else "video_path"]
    return template.format(
        episode_chunk=episode_id // int(info["chunks_size"]), episode_index=episode_id, video_key=camera
    )


def load_metadata(dataset_root, task):
    root = Path(dataset_root).resolve()
    spec = get_task(task)
    for rel, sha in spec.metadata_sha256.items():
        if _sha(_inside(root, rel)) != sha:
            raise ValueError(f"Pinned metadata differs: {rel}")
    info = json.loads((root / "meta/info.json").read_text())
    episodes = _jsonl(root / "meta/episodes.jsonl")
    tasks = _jsonl(root / "meta/tasks.jsonl")
    stats = _jsonl(root / "meta/episodes_stats.jsonl")
    if (
        info.get("codebase_version") != "v2.1"
        or info.get("fps") != spec.fps
        or info.get("total_episodes") != spec.episodes
        or info.get("total_frames") != spec.frames
        or type(info.get("chunks_size")) is not int
        or info["chunks_size"] < 1
        or info.get("splits") != {"train": f"0:{spec.episodes}"}
        or [int(x["episode_index"]) for x in episodes] != list(range(spec.episodes))
        or sum(int(x["length"]) for x in episodes) != spec.frames
        or any(int(x["length"]) < 1 for x in episodes)
        or [int(x["episode_index"]) for x in stats] != list(range(spec.episodes))
    ):
        raise ValueError("Single-task metadata/count/full-training split differs")
    if tasks != [{"task_index": 0, "task": spec.instruction}]:
        raise ValueError("Task instruction differs")
    for key, dim in [
        ("observation.state", 43),
        ("action.motion_token", 64),
        ("observation.inspire_left_hand_state", 6),
        ("observation.inspire_right_hand_state", 6),
        ("observation.projected_gravity", 3),
    ]:
        if info["features"][key]["shape"] != [dim]:
            raise ValueError(f"Wrong source dimension: {key}")
    for camera in spec.camera_keys:
        f = info["features"][camera]
        shape = f.get("shape", [])
        if (
            f.get("dtype") != "video"
            or len(shape) != 3
            or shape[-1] != 3
            or any(type(n) is not int or n < 1 for n in shape)
        ):
            raise ValueError(f"Expected positive HWC RGB video dimensions: {camera}")
    for side in ["left", "right"]:
        names = info["features"][f"observation.inspire_{side}_hand_state"].get("names")
        if names is not None and list(names) not in [list(HAND_ORDER), [f"{side}_{n}" for n in HAND_ORDER]]:
            raise ValueError("Measured hand field side/order differs")
    if info.get("total_videos") != spec.episodes * 2:
        raise ValueError("Wrong camera video count")
    for ep, row in zip(episodes, stats):
        for v in row["stats"].values():
            if "count" in v and np.any(np.asarray(v["count"]) != int(ep["length"])):
                raise ValueError("Statistics frame count differs")
    return {"root": root, "task": spec, "info": info, "episodes": episodes}


@dataclass(frozen=True)
class NumericEpisode:
    episode_id: int
    state: np.ndarray
    actions: np.ndarray
    frame_indices: np.ndarray
    timestamps: np.ndarray

    def __len__(self) -> int:
        return len(self.state)


def numeric_episode(
    columns: Mapping, episode_id: int, expected_length: int, global_start: int = 0, *, fps: int = 30
) -> NumericEpisode:
    """Validate same-row numeric inputs; no command labels or unit rescaling."""
    n = expected_length
    arrays = {}
    for key, dim in {
        "observation.state": 43,
        "action.motion_token": 64,
        "observation.inspire_left_hand_state": 6,
        "observation.inspire_right_hand_state": 6,
        "observation.projected_gravity": 3,
    }.items():
        value = np.asarray(columns[key], dtype=np.float32)
        if value.shape != (n, dim) or not np.isfinite(value).all():
            raise ValueError(f"Invalid or non-finite numeric feature: {key}")
        if "inspire_" in key and np.any((value < 0) | (value > 1)):
            raise ValueError("Measured hand values must already be physical [0,1]")
        arrays[key] = value
    frames = np.asarray(columns["frame_index"])
    times = np.asarray(columns["timestamp"], dtype=np.float64)
    if (
        not np.array_equal(frames, np.arange(n))
        or times.shape != (n,)
        or not np.isfinite(times).all()
        or not np.allclose(times, np.arange(n) / fps, rtol=0, atol=3e-6)
        or not np.array_equal(np.asarray(columns["index"]), np.arange(global_start, global_start + n))
        or not np.array_equal(np.asarray(columns["episode_index"]), np.full(n, episode_id))
        or not np.array_equal(np.asarray(columns["task_index"]), np.zeros(n))
    ):
        raise ValueError("Nonsequential frame/global indices, task ID, or timestamps")
    body = arrays["observation.state"]
    left, right = (arrays[f"observation.inspire_{side}_hand_state"] for side in ("left", "right"))
    state = np.concatenate(
        (body[:, :22], body[:, 29:36], left, right, arrays["observation.projected_gravity"]),
        axis=-1,
    )
    actions = np.concatenate((arrays["action.motion_token"], left, right), axis=-1)
    for value in (state, actions, frames, times):
        value.setflags(write=False)
    return NumericEpisode(episode_id, state, actions, frames, times)


class TaskNormalizer:
    """Fresh task q01/q99; constants and clipping match the new task adapters."""

    def __init__(self, statistics: dict):
        if (
            statistics.get("schema") != NORMALIZATION_SCHEMA
            or statistics.get("hand_target_time_offset") != 0
            or statistics.get("train_split_mode") != "all_episodes"
            or statistics.get("validation_episode_ids") != []
        ):
            raise ValueError("Incompatible task normalization schema")
        self.statistics = statistics
        self.transforms = {}
        for field, width, hands in (("state", 44, slice(29, 41)), ("action", 76, slice(64, 76))):
            bounds = statistics[f"{field}_bounds"]
            lower, upper = (np.asarray(bounds[key], dtype=np.float32) for key in ("q01", "q99"))
            if (
                lower.shape != (width,)
                or upper.shape != (width,)
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or np.any(upper < lower)
                or not np.array_equal(lower[hands], np.zeros(12))
                or not np.array_equal(upper[hands], np.ones(12))
            ):
                raise ValueError("Invalid task-specific bounds or physical hand bounds")
            span = upper - lower
            constant = span < 1e-4
            span[constant] = 2.0
            scale = 2.0 / span
            offset = -1.0 - scale * lower
            offset[constant] = -lower[constant]
            self.transforms[field] = (scale, offset)

    def forward(self, values: np.ndarray, field: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        scale, offset = self.transforms[field]
        if array.shape[-1] != len(scale) or not np.isfinite(array).all():
            raise ValueError("Invalid normalization input")
        return np.clip(array * scale + offset, -5.0, 5.0).astype(np.float32)

    def backward(self, values: np.ndarray, field: str = "action") -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        scale, offset = self.transforms[field]
        if array.shape[-1] != len(scale) or not np.isfinite(array).all():
            raise ValueError("Invalid denormalization input")
        return ((array - offset) / scale).astype(np.float32)

    def normalize_state(self, value: np.ndarray) -> np.ndarray:
        return self.forward(value, "state")

    def normalize_actions(self, value: np.ndarray) -> np.ndarray:
        return self.forward(value, "action")

    def normalize_actions_unclipped(self, value: np.ndarray) -> np.ndarray:
        """Affine coordinates of an actually issued token, without carry projection.

        Recorded training windows intentionally keep [-5,5] clipping. An issued
        ODE body endpoint may leave that interval; clipping it again would reset
        the next chunk's reference rather than carry the token actually issued.
        This method does not authorize actuator commands or alter hand bounds.
        """
        array = np.asarray(value, dtype=np.float32)
        scale, offset = self.transforms["action"]
        if array.shape[-1] != len(scale) or not np.isfinite(array).all():
            raise ValueError("Invalid unclipped action normalization input")
        result = (array * scale + offset).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite unclipped action normalization result")
        return result

    def denormalize_actions(self, value: np.ndarray) -> np.ndarray:
        return self.backward(value, "action")

    def save(self, path: str | Path) -> None:
        _write_new(Path(path), self.statistics)

    @classmethod
    def load(cls, path: str | Path) -> TaskNormalizer:
        return cls(json.loads(Path(path).read_text()))


def fit_normalization(episodes: Sequence[NumericEpisode], task: str | TaskSpec) -> TaskNormalizer:
    spec = get_task(task)
    if [ep.episode_id for ep in episodes] != list(range(spec.episodes)) or sum(
        len(ep) for ep in episodes
    ) != spec.frames:
        raise ValueError("Fresh normalization must cover every task episode exactly once")
    result = {
        "schema": NORMALIZATION_SCHEMA,
        "task": spec.name,
        "task_revision": spec.revision,
        "train_split_mode": "all_episodes",
        "train_episode_ids": list(range(spec.episodes)),
        "validation_episode_ids": [],
        "frames": spec.frames,
        "hand_target_time_offset": 0,
        "clip": [-5.0, 5.0],
        "state_dim": 44,
        "action_dim": 76,
        "hand_order": list(HAND_ORDER),
        "fitted_from_this_task_only": True,
    }
    for field, width, hands in (("state", 44, slice(29, 41)), ("actions", 76, slice(64, 76))):
        values = np.concatenate([getattr(ep, field) for ep in episodes])
        if values.shape != (spec.frames, width) or not np.isfinite(values).all():
            raise ValueError("Invalid normalization source arrays")
        low, high = np.quantile(values.astype(np.float64), [0.01, 0.99], axis=0).astype(np.float32)
        low[hands], high[hands] = 0.0, 1.0
        result[f"{'action' if field == 'actions' else field}_bounds"] = {
            "q01": low.tolist(),
            "q99": high.tolist(),
        }
    return TaskNormalizer(result)


def make_window(episode: NumericEpisode, anchor: int, normalizer: TaskNormalizer, *, horizon: int = 16) -> dict:
    """Current state/RGB, strictly previous body r0, same-row continuous targets."""
    if not 0 <= anchor < len(episode) or type(horizon) is not int or not 1 <= horizon <= 16:
        raise ValueError("Invalid anchor or horizon")
    count = min(horizon, len(episode) - anchor)
    actions = np.zeros((horizon, 76), dtype=np.float32)
    mask = np.zeros_like(actions)
    actions[:count] = normalizer.normalize_actions(episode.actions[anchor : anchor + count])
    mask[:count] = 1.0
    # A SONIC token cannot be reconstructed from joint state. At t0 there is
    # no causal recorded token: use an explicit normalized zero BOS, not GT[0].
    r0 = np.zeros(64, dtype=np.float32)
    if anchor:
        r0 = normalizer.normalize_actions(episode.actions[anchor - 1 : anchor])[0, :64].copy()
    return {
        "state": normalizer.normalize_state(episode.state[anchor : anchor + 1]),
        "r0": r0,
        "r0_is_bos": anchor == 0,
        "actions": actions,
        "action_mask": mask,
        "episode_id": episode.episode_id,
        "frame_index": int(episode.frame_indices[anchor]),
        "timestamp": float(episode.timestamps[anchor]),
        "sample_id": f"{episode.episode_id}:{int(episode.frame_indices[anchor])}",
        "camera_frames": {camera: int(episode.frame_indices[anchor]) for camera in CAMERAS},
    }


class TaskDataset:
    """Immutable numeric snapshot plus portable content-bound video references."""

    def __init__(self, root, prepared_dir, spec, manifest, episodes, normalizer):
        self.source_root = self.dataset_root = Path(root).resolve()
        self.prepared_dir = Path(prepared_dir)
        self.manifest_path = self.prepared_dir / "dataset_manifest.json"
        self.spec = spec
        self.manifest = manifest
        self.episodes = dict(episodes)
        self.normalizer = self.statistics = normalizer
        self.task = _document(spec)
        self.instruction = spec.instruction
        self.camera_keys = spec.camera_keys
        self.episode_ids = self.train_episode_ids = tuple(range(spec.episodes))
        self.validation_episode_ids = self.val_episode_ids = ()
        self.examples = tuple(
            (ep, t) for ep in self.episode_ids for t in range(0, len(self.episodes[ep]), manifest["anchor_stride"])
        )
        self.episode_lengths = {ep: len(v) for ep, v in self.episodes.items()}
        self.video_fingerprints = {
            str(self.source_root / k): v["fingerprint"] for k, v in manifest["sources"].items() if v["kind"] == "video"
        }
        # Mount names and filesystem IDs are provenance, never model/sample identity.
        content = {k: v for k, v in manifest.items() if k not in ("source_root", "identity_sha256")}
        content["sources"] = {
            rel: {
                "kind": r["kind"],
                "fingerprint": {k: r["fingerprint"][k] for k in ("size_bytes", "sha256", "hash_mode")},
            }
            for rel, r in manifest["sources"].items()
        }
        self.content_identity_sha256 = _digest(content)

    def __len__(self):
        return len(self.examples)

    def load_episode(self, episode_id):
        return self.episodes[episode_id]

    def sample(self, episode_id, anchor, *, horizon=16):
        out = make_window(self.episodes[episode_id], anchor, self.normalizer, horizon=horizon)
        out["camera_frames"] = {k: int(anchor) for k in self.camera_keys}
        return out

    def __getitem__(self, index):
        return self.sample(*self.examples[index])

    def video_path(self, episode_id, camera_key):
        if episode_id not in self.episodes or camera_key not in self.camera_keys:
            raise KeyError("Unknown episode/camera")
        return _inside(self.source_root, self.manifest["episode_sources"][str(episode_id)]["videos"][camera_key])

    def verify_sources(self):
        load_metadata(self.source_root, self.spec)
        for rel, record in self.manifest["sources"].items():
            _verify_source(_inside(self.source_root, rel), record["fingerprint"])

    @classmethod
    def load(cls, prepared_dir, *, source_root=None, verify_sources=True):
        directory = Path(prepared_dir).resolve()
        m = json.loads((directory / "dataset_manifest.json").read_text())
        spec = TaskSpec(**m["task"])
        identity = dict(m)
        expected = identity.pop("identity_sha256")
        if (
            _digest(identity) != expected
            or m.get("schema") != SCHEMA
            or m.get("hand_target_time_offset") != 0
            or m.get("horizon") != 16
            or m.get("camera_condition") != "current_anchor_only"
            or m.get("r0") != "previous_normalized_motion64_or_zero_BOS"
            or m.get("normalization_sha256") != _sha(directory / "normalization.json")
            or m.get("train_episode_ids") != list(range(spec.episodes))
            or m.get("validation_episode_ids") != []
            or type(m.get("anchor_stride")) is not int
            or m["anchor_stride"] < 1
        ):
            raise ValueError("Prepared task identity/schema differs")
        normalizer = TaskNormalizer.load(directory / "normalization.json")
        if (
            normalizer.statistics["task"] != spec.name
            or normalizer.statistics["task_revision"] != spec.revision
            or normalizer.statistics["frames"] != spec.frames
        ):
            raise ValueError("Normalizer task identity differs")
        episodes = {}
        for key, r in m["numeric_snapshots"].items():
            p = _inside(directory, r["path"])
            if _sha(p) != r["sha256"]:
                raise ValueError("Prepared numeric snapshot content differs")
            with np.load(p, allow_pickle=False) as x:
                arrays = {k: x[k].copy() for k in ["state", "actions", "frame_indices", "timestamps"]}
            for a in arrays.values():
                a.setflags(write=False)
            episodes[int(key)] = NumericEpisode(int(key), **arrays)
        if sorted(episodes) != list(range(spec.episodes)) or sum(len(e) for e in episodes.values()) != spec.frames:
            raise ValueError("Prepared episode coverage differs")
        obj = cls(source_root or m["source_root"], directory, spec, m, episodes, normalizer)
        if len(obj) != m["anchor_count"]:
            raise ValueError("Prepared anchor count differs")
        expected_paths = set(spec.metadata_sha256)
        for ep in range(spec.episodes):
            record = m["episode_sources"][str(ep)]
            expected_paths.add(record["parquet"])
            expected_paths.update(record["videos"].values())
        if set(m["sources"]) != expected_paths:
            raise ValueError("Prepared source coverage differs")
        for rel, pin in spec.metadata_sha256.items():
            if m["sources"][rel]["fingerprint"]["sha256"] != pin:
                raise ValueError("Metadata pin differs")
        if verify_sources:
            obj.verify_sources()
        return obj


def prepare_task_dataset(dataset_root, task, output_dir, *, anchor_stride=1):
    root = Path(dataset_root).resolve()
    directory = Path(output_dir).resolve()
    spec = get_task(task)
    if type(anchor_stride) is not int or anchor_stride < 1:
        raise ValueError("Positive integer anchor stride required")
    if directory == root or directory.is_relative_to(root):
        raise ValueError("Prepared output must be outside source data")
    if directory.exists():
        raise FileExistsError("Prepare into a new directory")
    meta = load_metadata(root, spec)
    info = meta["info"]
    sources = {
        k: {"kind": "metadata", "fingerprint": file_fingerprint(_inside(root, k), full_hash=True)}
        for k in spec.metadata_sha256
    }
    import pyarrow.parquet as pq

    episodes = {}
    paths = {}
    start = 0
    for row in meta["episodes"]:
        ep = int(row["episode_index"])
        count = int(row["length"])
        rel = _relative(info, "parquet", ep)
        p = _inside(root, rel)
        fp = file_fingerprint(p, full_hash=True)
        cols = pq.read_table(p, columns=list(NUMERIC_COLUMNS), use_threads=False).to_pydict()
        _verify_source(p, fp)
        episodes[ep] = numeric_episode(cols, ep, count, start, fps=spec.fps)
        start += count
        sources[rel] = {"kind": "parquet", "fingerprint": fp}
        paths[str(ep)] = {"parquet": rel, "videos": {}}
        for camera in spec.camera_keys:
            rel = _relative(info, "video", ep, camera)
            paths[str(ep)]["videos"][camera] = rel
            sources[rel] = {"kind": "video", "fingerprint": file_fingerprint(_inside(root, rel))}
    normalizer = fit_normalization(list(episodes.values()), spec)
    for rel, r in sources.items():
        _verify_source(_inside(root, rel), r["fingerprint"])
    directory.mkdir(parents=True)
    (directory / "numeric").mkdir()
    normalizer.save(directory / "normalization.json")
    snapshots = {}
    for ep, value in episodes.items():
        rel = f"numeric/episode_{ep:06d}.npz"
        p = directory / rel
        with p.open("xb") as f:
            np.savez(
                f,
                state=value.state,
                actions=value.actions,
                frame_indices=value.frame_indices,
                timestamps=value.timestamps,
            )
        snapshots[str(ep)] = {"path": rel, "sha256": _sha(p), "frames": len(value)}
    m = {
        "schema": SCHEMA,
        "task": _document(spec),
        "source_root": str(root),
        "sources": sources,
        "episode_sources": paths,
        "numeric_snapshots": snapshots,
        "normalization_sha256": _sha(directory / "normalization.json"),
        "anchor_stride": anchor_stride,
        "anchor_count": sum(len(range(0, len(e), anchor_stride)) for e in episodes.values()),
        "train_episode_ids": list(range(spec.episodes)),
        "validation_episode_ids": [],
        "hand_target_time_offset": 0,
        "r0": "previous_normalized_motion64_or_zero_BOS",
        "camera_condition": "current_anchor_only",
        "horizon": 16,
        "source_data_written": False,
        "normalization_inherited": False,
        "video_hash_scope": "head/tail content fingerprint; not a full video SHA",
        "source_relocation_rule": "resolve references against source_root override; verify content; filesystem IDs are provenance",
    }
    m["identity_sha256"] = _digest(m)
    _write_new(directory / "dataset_manifest.json", m)
    return TaskDataset(root, directory, spec, m, episodes, normalizer)
