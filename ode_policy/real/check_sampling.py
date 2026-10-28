"""Offline check of a chunk policy's serving-time sampling settings on real demonstration windows.

For each (sampler, steps) setting: the action-chunk error against the demonstrations (joints in rad,
gripper in mm) on windows from the held-out episode and from training episodes, and the latency of
one batch-1 prediction. Used to pick a real-time setting (e.g. DDIM steps for the Diffusion Policy)
that does not change what the policy does.

    python ode_policy/real/check_sampling.py --ckpt il_outputs/ddpm_unet/r1lite_insert_flower/checkpoints/100.ckpt \\
        --zarr data_policy/r1lite_insert_flower.zarr --settings ddpm:100 ddim:16 ddim:10 ddim:5
    python ode_policy/real/check_sampling.py --ckpt il_outputs/fm_dit/.../100.ckpt --settings keep:10 keep:5
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ode_policy.datasets.robot_image_dataset import RobotImageDataset  # noqa: E402
from ode_policy.deploy.policy_inference import (  # noqa: E402
    load_checkpoint_policy, set_num_inference_steps, set_sampler)


def windows(ds: RobotImageDataset, episode_mask: np.ndarray, n: int, rng) -> np.ndarray:
    """n sample indices of the dataset's sampler whose window starts inside the masked episodes."""
    ends = np.asarray(ds.replay_buffer.episode_ends)
    starts = np.concatenate([[0], ends[:-1]])
    idx = []
    for i, (b0, b1, s0, s1) in enumerate(ds.sampler.indices):
        ep = int(np.searchsorted(ends, b0, side="right"))
        if episode_mask[ep] and s0 == 0 and s1 == ds.horizon:   # full windows only (no padding)
            idx.append(i)
    idx = np.asarray(idx)
    return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--zarr", required=True, help="the command-layout zarr the policy was trained on")
    ap.add_argument("--settings", nargs="+", default=["keep:0"],
                    help="sampler:steps pairs; sampler 'keep' keeps the trained one, steps 0 keeps the trained count")
    ap.add_argument("--n", type=int, default=64, help="windows per split")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=2, help="noise draws per window")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base, cfg = load_checkpoint_policy(args.ckpt, str(dev))
    To, Ta = int(cfg.n_obs_steps), int(cfg.n_action_steps)
    cams = [k.replace("_cam", "_camera") for k, v in cfg.shape_meta.obs.items()
            if v.get("type") == "rgb" and k != "head_cam"]
    # the training-time split: same val_ratio / seed as the run (val = the last episode)
    ds = RobotImageDataset(args.zarr, horizon=int(cfg.horizon), pad_before=To - 1, pad_after=Ta - 1,
                           val_ratio=float(cfg.dataset_config.val_ratio), seed=int(cfg.dataset_config.seed),
                           batch_size=args.batch, extra_camera_keys=cams)
    ds_all = RobotImageDataset(args.zarr, horizon=int(cfg.horizon), pad_before=To - 1, pad_after=Ta - 1,
                               val_ratio=0.0, batch_size=args.batch, extra_camera_keys=cams)
    rng = np.random.default_rng(0)
    split = {"held-out": windows(ds_all, ~ds.train_mask, args.n, rng),
             "train": windows(ds_all, ds.train_mask, args.n, rng)}
    print(f"[check] {type(base).__name__} {args.ckpt} on {dev}; windows: "
          + ", ".join(f"{k} {len(v)}" for k, v in split.items()))

    for setting in args.settings:
        sampler, steps = setting.split(":")
        policy = copy.deepcopy(base)
        if sampler != "keep":
            set_sampler(policy, sampler)
        if int(steps) > 0:
            set_num_inference_steps(policy, int(steps))
        n_steps = getattr(policy, "num_inference_steps", getattr(policy, "num_sampling_steps", "?"))
        res = {}
        for name, idx in split.items():
            err_j, err_g = [], []
            for s in range(args.seeds):
                torch.manual_seed(s)
                for b in range(0, len(idx), args.batch):
                    sel = idx[b:b + args.batch]
                    if len(sel) < args.batch:
                        sel = np.concatenate([sel, np.repeat(sel[-1:], args.batch - len(sel))])
                    batch = ds_all.postprocess(ds_all[sel], dev)
                    with torch.no_grad():
                        pred = policy.predict_action(batch["obs"])["action"]          # (B, Ta, D)
                    gt = batch["action"][:, To - 1:To - 1 + Ta]
                    err_j.append((pred[..., :-1] - gt[..., :-1]).abs().mean().item())
                    err_g.append((pred[..., -1] - gt[..., -1]).abs().mean().item())
            res[name] = (np.mean(err_j), np.mean(err_g))
        # latency: one batch-1 prediction, warm
        one = ds_all.postprocess(ds_all[np.repeat(split["held-out"][:1], args.batch)], dev)
        obs1 = {k: v[:1] for k, v in one["obs"].items()}
        with torch.no_grad():
            policy.predict_action(obs1)
            t = []
            for _ in range(15):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                policy.predict_action(obs1)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t.append((time.perf_counter() - t0) * 1e3)
        print(f"[check] {setting:10s} steps={n_steps:>3}  "
              + "  ".join(f"{k}: |joint err| {v[0]:.4f} rad, |gripper err| {v[1]:5.2f} mm" for k, v in res.items())
              + f"  latency p50 {np.percentile(t, 50):6.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
