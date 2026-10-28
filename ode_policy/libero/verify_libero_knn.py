"""Verify the KNN neighbourhood invariants on a LIBERO zarr.

Checks the two properties MARS's diversity loss depends on, both of which are
silent when broken (training runs fine, the router just learns the wrong thing):

1. **Same task.** Every neighbour shares the anchor's ``task_id``. Without this a
   sample's "disagreement" is dominated by the difference between two unrelated
   tasks, so ``target_spread`` is high everywhere and the router fires everywhere.
2. **Different episode.** Every neighbour comes from a different demo. Same-demo
   neighbours are time-shifted copies of one trajectory, which turns
   ``target_spread`` into "how fast does the action change" and collapses it
   toward zero exactly at forks.

It also reports the ``target_spread`` distribution, since a spread that is flat
across the dataset means the router has nothing to key on.

Usage:
    python ode_policy/libero/verify_libero_knn.py \
        --zarr data_policy/libero_spatial_50demo.zarr
"""
from __future__ import annotations

import argparse

import numpy as np
import zarr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
    p.add_argument("--knn_k", type=int, default=20)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n_obs_steps", type=int, default=8)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--group_key", default="task_id",
                   help="Set to 'none' to check the task-mixed (ablation) behaviour.")
    args = p.parse_args()

    from ode_policy.libero.libero_dataset import LiberoImageDatasetKNN

    group_key = None if args.group_key.lower() == "none" else args.group_key
    ds = LiberoImageDatasetKNN(
        zarr_path=args.zarr,
        horizon=args.horizon,
        pad_before=args.n_obs_steps - 1,
        pad_after=args.n_action_steps - 1,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        knn_k=args.knn_k,
        knn_group_key=group_key,
        # Matches what train_libero.sh sets: the flow source is the measured
        # motion, not the recorded command.
        history_source="measured_delta",
        knn_key_source="resnet",
        knn_proprio_weight=0,
        val_ratio=0.0,
        batch_size=8,
    )

    neighbor_ids = ds.per_sample_neighbor_ids           # (N, k)
    sample_ep = ds.per_sample_ep                        # (N,)
    spread = ds.per_sample_target_spread                # (N, D)

    root = zarr.open(args.zarr, "r")
    task_of_frame = np.asarray(root["data"]["task_id"]).reshape(-1)
    sample_task = task_of_frame[ds._current_frame_indices(ds.sampler.indices)]

    n, k = neighbor_ids.shape
    print(f"\nsamples={n}  k={k}  tasks={len(np.unique(sample_task))}  "
          f"episodes={len(np.unique(sample_ep))}")

    ok = True

    same_task = sample_task[neighbor_ids] == sample_task[:, None]
    frac_same_task = same_task.mean()
    if group_key is None:
        print(f"[ablation] group_key=None -> same-task fraction {frac_same_task:.3f} "
              f"(no constraint expected)")
    elif same_task.all():
        print(f"[PASS] every neighbour shares the anchor's task ({frac_same_task:.3f})")
    else:
        bad = int((~same_task).sum())
        print(f"[FAIL] {bad}/{n * k} neighbour pairs cross task boundaries")
        ok = False

    same_ep = sample_ep[neighbor_ids] == sample_ep[:, None]
    if not same_ep.any():
        print("[PASS] no neighbour comes from the anchor's own demo")
    else:
        bad = int(same_ep.sum())
        print(f"[FAIL] {bad}/{n * k} neighbour pairs come from the same demo")
        ok = False

    if (neighbor_ids == np.arange(n)[:, None]).any():
        print("[FAIL] some sample is its own neighbour")
        ok = False
    else:
        print("[PASS] no sample is its own neighbour")

    per_sample = spread.mean(axis=1)
    print(f"\ntarget_spread: mean={per_sample.mean():.4f} std={per_sample.std():.4f} "
          f"p10={np.percentile(per_sample, 10):.4f} p50={np.percentile(per_sample, 50):.4f} "
          f"p90={np.percentile(per_sample, 90):.4f}")
    if per_sample.std() < 1e-6:
        print("[WARN] target_spread is constant — the router has no signal to key on.")
    print(f"per-dim mean spread: {np.round(spread.mean(axis=0), 4)}")

    print("\nRESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
