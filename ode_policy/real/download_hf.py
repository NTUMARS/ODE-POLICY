"""Download a LeRobot v2.1 dataset from the HuggingFace Hub, only the cameras we train on.

    python ode_policy/real/download_hf.py --repo <hf_user>/r1lite_insert_flower \\
        --dest data_raw/lerobot/r1lite_insert_flower \\
        --cameras observation.images.head_rgb observation.images.right_wrist_rgb

Downloads meta/*, data/** and videos/*/<camera>/* (the other cameras are skipped), then
checks the file counts against meta/info.json. Re-running is incremental.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--cameras", nargs="+",
                   default=["observation.images.head_rgb", "observation.images.right_wrist_rgb"])
    p.add_argument("--max-workers", type=int, default=8)
    args = p.parse_args()

    patterns = ["meta/*", "data/**"] + [f"videos/*/{c}/*" for c in args.cameras]
    print(f"[download] {args.repo} -> {args.dest}\n[download] patterns={patterns}")
    snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(args.dest),
                      allow_patterns=patterns, max_workers=args.max_workers)

    info = json.loads((args.dest / "meta" / "info.json").read_text())
    n_ep = int(info["total_episodes"])
    n_parquet = len(list(args.dest.glob("data/chunk-*/episode_*.parquet")))
    ok = n_parquet == n_ep
    print(f"[download] episodes={n_ep} frames={info['total_frames']} fps={info['fps']} "
          f"codebase={info['codebase_version']} parquet_files={n_parquet}")
    for c in args.cameras:
        n_mp4 = len(list(args.dest.glob(f"videos/chunk-*/{c}/episode_*.mp4")))
        print(f"[download] {c}: {n_mp4} mp4")
        ok &= n_mp4 == n_ep
    if not ok:
        raise SystemExit("[download] FILE COUNT MISMATCH -- incomplete download")
    print("[download] OK")


if __name__ == "__main__":
    main()
