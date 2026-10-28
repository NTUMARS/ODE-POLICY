"""ODE supervision for openpi's LIBERO dataset (physical-intelligence/libero, LeRobot format).

The ODE head is trained on virtual-target poses, not on LIBERO's delta commands. For every
frame t this writes, frame-aligned with the LeRobot dataset (no images are copied):

    start[t]    the reference the integration starts from: the virtual target of the command
                issued at t-1 (the measured pose at an episode's first frame)
    actions[t]  [virtual target of the command at t (6), gripper command (1)]
    proprio[t]  [eef position (3), sign-continuous quaternion (4), gripper opening (1)]

plus the episode bookkeeping that openpi.shared.ode_data.ODETargetDataset checks row by row,
and the limits normalisation statistics (poses fit on starts and targets, gripper identity).
The tracker (openpi.shared.ode_pose.track_targets, gain 1) must reproduce every demonstrated
command from its virtual target; the script refuses to write the cache otherwise.

    cd vla/pi05_ode/openpi && uv run ../prepare_ode_data.py
    # -> vla/pi05_ode/assets/ode_targets.npz, vla/pi05_ode/assets/physical-intelligence/libero/norm_stats.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tqdm

from openpi.shared import normalize
from openpi.shared.ode_data import limits_stats, proprioception
from openpi.shared.ode_pose import episode_targets, track_targets

HERE = Path(__file__).resolve().parent
REPO_ID = "physical-intelligence/libero"


def main():
    default_root = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot")) / REPO_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=Path, default=default_root,
                    help="local copy of the LeRobot dataset (download it once, e.g. by running openpi's training)")
    ap.add_argument("--out_dir", type=Path, default=HERE / "assets")
    args = ap.parse_args()

    cache_path = args.out_dir / "ode_targets.npz"
    asset_dir = args.out_dir / REPO_ID
    if cache_path.exists() or (asset_dir / "norm_stats.json").exists():
        raise SystemExit(f"{cache_path} or {asset_dir}/norm_stats.json exists; remove them to rebuild")
    files = sorted((args.dataset_root / "data").glob("chunk-*/episode_*.parquet"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise SystemExit(f"no episodes under {args.dataset_root}/data")

    parts = {k: [] for k in ("start", "actions", "proprio", "episode_index", "frame_index", "episode_end")}
    rows, max_error = 0, 0.0
    for path in tqdm.tqdm(files, desc="episodes"):
        tab = pq.read_table(path, columns=["state", "actions", "episode_index", "frame_index"])
        state = np.asarray(tab["state"].to_pylist(), np.float32)
        command = np.asarray(tab["actions"].to_pylist(), np.float32)
        ep, frame = (np.asarray(tab[k].to_pylist()) for k in ("episode_index", "frame_index"))
        if len(np.unique(ep)) != 1 or not np.array_equal(frame, np.arange(len(frame))):
            raise SystemExit(f"{path}: expected one episode with contiguous frames")
        start, target, grip = episode_targets(state, command)
        max_error = max(max_error, float(np.max(np.abs(track_targets(target, state, grip) - np.clip(command, -1, 1)))))
        rows += len(state)
        values = (start, np.concatenate([target, grip[:, None]], -1), proprioception(state), ep, frame,
                  np.full(len(state), rows))
        for key, value in zip(parts, values, strict=True):
            parts[key].append(value)
    if max_error > 1e-5:
        raise SystemExit(f"tracker does not reproduce the demonstrated commands (max error {max_error:.2e})")
    arrays = {key: np.concatenate(value) for key, value in parts.items()}

    pose = np.concatenate([arrays["start"], arrays["actions"][:, :6]])
    pose_stat = limits_stats(pose)
    stats = {
        "ode_start": pose_stat,
        "state": limits_stats(arrays["proprio"]),
        "actions": normalize.NormStats(mean=np.r_[pose_stat.mean, 0.0].astype(np.float32),
                                       std=np.r_[pose_stat.std, 1.0].astype(np.float32)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **arrays)
    normalize.save(asset_dir, stats)
    print(json.dumps({"episodes": len(files), "frames": rows, "max_roundtrip_error": max_error,
                      "cache": str(cache_path), "norm_stats": str(asset_dir / "norm_stats.json")}, indent=2))


if __name__ == "__main__":
    main()
