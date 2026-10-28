"""Convert a LeRobot v2.1 dataset (R1 Lite) to the zarr layout RobotImageDataset reads.

Two things are deliberately identical to the deployment path, so a policy trained
here sees the same pixels the deployed server feeds it:
  * images are PAD-TO-SQUARE + INTER_AREA resized (never center-cropped): the robot
    client sends 224x224 pad-to-square PNGs and the server resizes them to --image-size;
  * the per-frame vectors are concatenated in the given field order: right_arm 6 +
    right_gripper 1 = 7 dims (--preset right_arm), or left_arm 6 + left_gripper 1 +
    right_arm 6 + right_gripper 1 = 14 dims (--preset bimanual), which is exactly the
    robot client's 14-D wire order [L1..L6, L_grip, R1..R6, R_grip].

LeRobot layout (input):
    <dataset>/meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl
    <dataset>/data/chunk-XXX/episode_YYYYYY.parquet
    <dataset>/videos/chunk-XXX/<video_key>/episode_YYYYYY.mp4

Zarr layout (output):
    <out>.zarr/data/head_camera   (N, 3, H, W) uint8 NCHW
               data/wrist_camera  (N, 3, H, W) uint8 NCHW   the RIGHT wrist (historical name)
               data/left_wrist_camera (N, 3, H, W)          --preset bimanual only
               data/state         (N, state_dim) float32   measured joints (rad) + gripper (mm)
               data/action        (N, action_dim) float32  commanded joint targets + gripper command
               meta/episode_ends  (E,) int64

This is the "command layout" zarr: fm_dit / a2a_dit / mars train on it directly. The ODE
policy trains on the derived zarr from build_ode_zarr_real.py.

    python ode_policy/real/lerobot2zarr.py \\
        --src data_raw/lerobot/r1lite_insert_flower \\
        --out data_policy/r1lite_insert_flower.zarr --overwrite
    python ode_policy/real/lerobot2zarr.py --preset bimanual \\
        --src data_raw/lerobot/r1lite_microwave --out data_policy/r1lite_microwave.zarr --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as iio
import numpy as np
import pyarrow.parquet as pq
import zarr
from tqdm import tqdm


@dataclass(frozen=True)
class CameraSpec:
    out_key: str   # zarr array name: "head_camera" / "wrist_camera" / "left_wrist_camera"
    src_key: str   # LeRobot video_key, e.g. "observation.images.head_rgb"


# short name -> zarr array. "wrist" is the RIGHT wrist: the name predates the bimanual tasks and the
# deploy server maps wrist_cam <- the client's cam_right_wrist, left_wrist_cam <- cam_left_wrist.
CAMERA_SHORT_NAMES = ("head", "wrist", "left_wrist")

# what to read for each robot layout; --cameras / --state-fields / --action-fields override a preset
PRESETS = {
    "right_arm": {
        "cameras": ["head:observation.images.head_rgb", "wrist:observation.images.right_wrist_rgb"],
        "state_fields": ["observation.state.right_arm", "observation.state.right_gripper"],
        "action_fields": ["action.right_arm", "action.right_gripper"],
    },
    "bimanual": {
        "cameras": ["head:observation.images.head_rgb", "wrist:observation.images.right_wrist_rgb",
                    "left_wrist:observation.images.left_wrist_rgb"],
        "state_fields": ["observation.state.left_arm", "observation.state.left_gripper",
                         "observation.state.right_arm", "observation.state.right_gripper"],
        "action_fields": ["action.left_arm", "action.left_gripper", "action.right_arm", "action.right_gripper"],
    },
}


def parse_camera_spec(s: str) -> CameraSpec:
    """Parse "head:observation.images.head_rgb" -> CameraSpec."""
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"--cameras entry {s!r} must be 'short_name:lerobot_video_key' (short_name in {CAMERA_SHORT_NAMES})")
    short, src = (x.strip() for x in s.split(":", 1))
    if short not in CAMERA_SHORT_NAMES:
        raise argparse.ArgumentTypeError(f"camera short name must be one of {CAMERA_SHORT_NAMES}, got {short!r}")
    return CameraSpec(out_key=f"{short}_camera", src_key=src)


def field_dims(table_dict: dict, fields: Iterable[str]) -> list[int]:
    """Per-frame width of each field (6 for an arm, 1 for a gripper), from the first frame."""
    return [int(np.asarray(table_dict[f][0], dtype=np.float32).size) for f in fields]


def pad_to_square_and_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Pad the shorter side with zeros to a square, then resize to (size, size).

    Matches the robot client's `prepare_image` geometry (aspect-preserving fit into a
    square with black bars). Center-cropping instead would show the policy ~280 px of
    side content that the deployed image replaces with black -- a silent FOV shift.
    """
    h, w = img.shape[:2]
    side = max(h, w)
    if h != w:
        padded = np.zeros((side, side, img.shape[2]), dtype=img.dtype)
        pad_h, pad_w = (side - h) // 2, (side - w) // 2
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = img
        img = padded
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def read_video_frames(path: Path, image_size: int) -> np.ndarray:
    """Decode an mp4 (AV1 / h264 / ...) into (T, 3, H, W) uint8 RGB."""
    reader = iio.get_reader(str(path))
    frames = []
    for frame in reader:   # RGB HWC uint8
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frames.append(pad_to_square_and_resize(frame, image_size))
    reader.close()
    arr = np.moveaxis(np.stack(frames, axis=0), -1, 1)   # (T, 3, H, W)
    return arr.astype(np.uint8, copy=False)


def extract_vector_columns(table_dict: dict, fields: Iterable[str], expected_len: int) -> np.ndarray:
    """Concatenate per-frame list[float] columns into (T, total_dim) float32."""
    parts = []
    for f in fields:
        if f not in table_dict:
            raise KeyError(f"field {f!r} not in parquet; available: {sorted(table_dict)}")
        arr = np.asarray([np.asarray(v, dtype=np.float32).ravel() for v in table_dict[f]])
        if arr.shape[0] != expected_len:
            raise ValueError(f"field {f!r} has {arr.shape[0]} rows, episode has {expected_len}")
        parts.append(arr)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def discover_episodes(src: Path) -> list[tuple[int, Path, dict[str, Path]]]:
    """(episode_index, parquet_path, {video_key: mp4_path}) for every episode, sorted."""
    info = json.loads((src / "meta" / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    episodes = []
    with open(src / "meta" / "episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            ep_idx = int(ep["episode_index"])
            chunk = ep_idx // chunks_size
            parquet = src / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
            videos = {vk: src / info["video_path"].format(episode_chunk=chunk, video_key=vk, episode_index=ep_idx)
                      for vk in video_keys}
            episodes.append((ep_idx, parquet, videos))
    episodes.sort(key=lambda x: x[0])
    return episodes


def convert(src: Path, out: Path, cameras: list[CameraSpec], state_fields: list[str],
            action_fields: list[str], image_size: int, max_episodes: int | None, overwrite: bool,
            layout: str = "custom") -> None:
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    info = json.loads((src / "meta" / "info.json").read_text())
    tasks = [json.loads(l)["task"] for l in (src / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()] \
        if (src / "meta" / "tasks.jsonl").is_file() else []
    episodes = discover_episodes(src)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise RuntimeError(f"no episodes under {src}")

    print(f"[convert] {len(episodes)} episodes, fps={info.get('fps')}, tasks={tasks}")
    print(f"[convert] cameras={[(c.out_key, c.src_key) for c in cameras]}")
    print(f"[convert] state_fields={state_fields}\n[convert] action_fields={action_fields}\n[convert] -> {out}")

    root = zarr.group(str(out))
    data, meta = root.create_group("data"), root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    state_dim = action_dim = state_dims = action_dims = None
    cam_initialized = False
    total, episode_ends, ep_indices = 0, [], []

    for ep_idx, parquet_path, video_paths in tqdm(episodes, desc="episodes"):
        table = pq.read_table(str(parquet_path))
        ep_len = table.num_rows
        table_dict = {name: table.column(name).to_pylist() for name in table.column_names}
        state = extract_vector_columns(table_dict, state_fields, ep_len)
        action = extract_vector_columns(table_dict, action_fields, ep_len)

        cam_arrays: dict[str, np.ndarray] = {}
        for spec in cameras:
            mp4 = video_paths.get(spec.src_key)
            if mp4 is None or not mp4.is_file():
                raise FileNotFoundError(f"episode {ep_idx}: missing video {spec.src_key!r} at {mp4}")
            frames = read_video_frames(mp4, image_size)
            if frames.shape[0] != ep_len:
                n = min(frames.shape[0], ep_len)
                print(f"  [warn] episode {ep_idx} {spec.src_key}: video={frames.shape[0]} parquet={ep_len} -> trim to {n}")
                frames, state, action, ep_len = frames[:n], state[:n], action[:n], n
            cam_arrays[spec.out_key] = frames
        # a camera trimmed after another was decoded: cut every array to the shortest
        for k in cam_arrays:
            cam_arrays[k] = cam_arrays[k][:ep_len]

        if state_dim is None:
            state_dim, action_dim = state.shape[1], action.shape[1]
            state_dims, action_dims = field_dims(table_dict, state_fields), field_dims(table_dict, action_fields)
            data.create_dataset("state", shape=(0, state_dim), chunks=(1024, state_dim), dtype="float32",
                                compressor=compressor, overwrite=True)
            data.create_dataset("action", shape=(0, action_dim), chunks=(1024, action_dim), dtype="float32",
                                compressor=compressor, overwrite=True)
            meta.create_dataset("episode_ends", shape=(0,), chunks=(1024,), dtype="int64",
                                compressor=compressor, overwrite=True)
        if not cam_initialized:
            for out_key, frames in cam_arrays.items():
                data.create_dataset(out_key, shape=(0, *frames.shape[1:]), chunks=(64, *frames.shape[1:]),
                                    dtype="uint8", compressor=compressor, overwrite=True)
            cam_initialized = True

        data["state"].append(state)
        data["action"].append(action)
        for out_key, frames in cam_arrays.items():
            data[out_key].append(frames)
        total += ep_len
        episode_ends.append(total)
        ep_indices.append(ep_idx)

    meta["episode_ends"].append(np.asarray(episode_ends, dtype=np.int64))
    meta.attrs.update({
        "task_name": src.name,
        "tasks": tasks,
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "num_episodes": len(episodes),
        "episode_indices": ep_indices,
        "layout": layout,
        "state_fields": state_fields,
        "action_fields": action_fields,
        "state_field_dims": state_dims,
        "action_field_dims": action_dims,
        "image_size": image_size,
        "image_geometry": "pad_to_square+INTER_AREA",
        "cameras": [{"out_key": c.out_key, "src_key": c.src_key} for c in cameras],
    })
    print(f"[convert] done: {total} frames / {len(episodes)} episodes -> {out} "
          f"(state_dim={state_dim}, action_dim={action_dim})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="LeRobot v2.1 dataset root")
    p.add_argument("--out", type=Path, required=True, help="output zarr path")
    p.add_argument("--preset", choices=sorted(PRESETS), default="right_arm",
                   help="right_arm: head + right wrist, 7-D (single-arm tasks); bimanual: head + right wrist + "
                        "left wrist, 14-D in the client's wire order")
    p.add_argument("--cameras", nargs="+", type=parse_camera_spec, default=None,
                   help=f"'short:lerobot_key' pairs, short in {CAMERA_SHORT_NAMES} (default: the preset's)")
    p.add_argument("--state-fields", nargs="+", default=None, help="default: the preset's")
    p.add_argument("--action-fields", nargs="+", default=None, help="default: the preset's")
    p.add_argument("--image-size", type=int, default=256, help="pad-to-square + resize target")
    p.add_argument("--max-episodes", type=int, default=None, help="cap for smoke tests")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    preset = PRESETS[args.preset]
    custom = any(v is not None for v in (args.cameras, args.state_fields, args.action_fields))
    convert(args.src, args.out,
            args.cameras or [parse_camera_spec(c) for c in preset["cameras"]],
            args.state_fields or preset["state_fields"], args.action_fields or preset["action_fields"],
            args.image_size, args.max_episodes, args.overwrite, layout="custom" if custom else args.preset)


if __name__ == "__main__":
    main()
