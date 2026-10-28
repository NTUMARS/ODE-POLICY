"""LIBERO dataset variants: language conditioning + task-aware KNN.

Two things separate a LIBERO zarr from a single-task one:

1. **It is multi-task.** One zarr merges every task in a suite, and each step
   carries a ``task_id``. The policy therefore has to be told *which* task it is
   doing, or it can only learn the average of ten behaviours that start from
   near-identical robot poses.

2. **Its multi-task-ness breaks the diversity loss.** MARS measures "how much
   do different demos disagree at this moment" over KNN neighbours. Across tasks
   that quantity is dominated by the task difference, not by ambiguity, so the
   router would see spread everywhere and fire everywhere. Neighbours must stay
   inside one task.

Both are handled here without touching any policy code:

* ``task_emb`` enters as an ordinary ``low_dim`` observation key.
  ``MultiImageObsEncoder`` concatenates every ``low_dim`` key in ``shape_meta``
  into the global condition, so declaring the key in ``shape_meta`` is the whole
  integration. Embeddings are precomputed (see ``text_encoder.py``) and looked up
  per step by ``task_id``.
* ``knn_group_key="task_id"`` makes the base KNN dataset build one neighbour tree
  per task (see ``robot_image_dataset_knn.py``).

The embedding is consumed through an **identity** normalizer. It is already
L2-normalized at conversion time, and fitting a LinearNormalizer to it would be
actively harmful: with only ~10 distinct vectors in the dataset, per-channel
min/max normalization would rescale each of the 512 channels by whatever tiny
spread those 10 points happen to have, amplifying noise directions into the same
range as real ones.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import zarr

from ode_policy.libero.libero_conventions import (
    measured_delta_from_poses,
    measured_delta_from_poses_torch,
)
from ode_policy.datasets.robot_image_dataset import RobotImageDataset
from ode_policy.utils.history import to_action_space_history_np
from ode_policy.datasets.robot_image_dataset_knn import RobotImageDatasetKNN
from ode_policy.utils.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)


class _LiberoObsMixin:
    """Adds the LIBERO-specific observation keys: ``task_emb`` and ``wrist_cam``.

    ``wrist_cam`` (the eye-in-hand view) is opt-in via ``use_wrist_cam``. Published
    LIBERO baselines condition on both views and the wrist camera carries most of
    the fine alignment signal near grasp, so enabling it is the comparable setting;
    it is off by default only so the agentview-only ablation stays the cheaper
    default. Turning it on also requires adding ``wrist_cam`` to ``shape_meta.obs``
    so the encoder builds a backbone for it.
    """

    use_wrist_cam = False

    def _init_task_meta(self, zarr_path: str) -> None:
        root = zarr.open(zarr_path, "r")
        if "task_emb" not in root["meta"]:
            raise KeyError(
                f"{zarr_path} has no meta/task_emb. Regenerate it with "
                "ode_policy/libero/hdf5_to_zarr.py, which encodes the "
                "instructions into meta/task_emb."
            )
        self.task_emb = np.asarray(root["meta"]["task_emb"], dtype=np.float32)  # (K, D)
        self.task_names = list(root.attrs.get("task_names", []))
        self.task_instructions = list(root.attrs.get("task_instructions", []))
        self.text_model = root.attrs.get("text_model", "unknown")
        self._task_emb_t = torch.from_numpy(self.task_emb)
        self._task_emb_device = None  # lazily-cached device copy

        n_tasks, dim = self.task_emb.shape
        print(f"[libero-dataset] {n_tasks} tasks, task_emb dim={dim}, "
              f"text_model={self.text_model}, wrist_cam={self.use_wrist_cam}")

        self._init_measured_delta_stats()

        if self.use_wrist_cam and "wrist_camera" not in root["data"]:
            raise KeyError(
                f"use_wrist_cam=True but {zarr_path} has no data/wrist_camera. "
                "Re-run hdf5_to_zarr.py without --no_wrist."
            )

    def _measured_delta_all(self) -> np.ndarray:
        """(N, 7) measured delta over the whole buffer, computed per episode so no
        delta spans an episode cut. Cached: both the normalizer fit and the KNN
        history hook need it, and they run at different points in construction.
        """
        cached = getattr(self, "_md_all", None)
        if cached is not None:
            return cached
        states = np.asarray(self.replay_buffer["state"])
        ends = np.asarray(self.replay_buffer.episode_ends)
        starts = np.concatenate([[0], ends[:-1]])
        md = np.concatenate(
            [measured_delta_from_poses(states[a:b, :3], states[a:b, 3:7], states[a:b, 7])
             for a, b in zip(starts, ends)],
            axis=0,
        )
        self._md_all = md
        return md

    def _init_measured_delta_stats(self) -> None:
        """Fit the measured-delta normalizer, per episode so no delta spans a cut.

        `measured_delta` needs its OWN normalizer rather than sharing the action
        one. The OSC controller only partially reaches its target within a control
        step, so the motion the arm actually achieves is ~4x smaller than the
        command that produced it (measured |dpos| 0.081 vs commanded 0.344 on
        libero_spatial). Pushed through the action normalizer, the history would
        collapse toward zero and stop being a useful flow source — A2A's premise
        is that source and target are close, and a 4x scale gap breaks it. Fitting
        the delta's own range restores that.
        """
        md = self._measured_delta_all()
        self._measured_delta_stats = array_to_stats(md)
        print(f"[libero-dataset] measured_delta range "
              f"pos {np.round(md[:, :3].min(0), 3)}..{np.round(md[:, :3].max(0), 3)}  "
              f"rot {np.round(md[:, 3:6].min(0), 3)}..{np.round(md[:, 3:6].max(0), 3)}")

    @property
    def task_emb_dim(self) -> int:
        return int(self.task_emb.shape[1])

    def _task_emb_on(self, device) -> torch.Tensor:
        """Device-resident copy of the (K, D) embedding table, cached."""
        if self._task_emb_device is None or self._task_emb_device.device != torch.device(device):
            self._task_emb_device = self._task_emb_t.to(device)
        return self._task_emb_device

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = super().get_normalizer(mode=mode, **kwargs)
        # Identity: the embedding is already unit-norm and only ~10 distinct
        # vectors exist, so a fitted per-channel range would be meaningless.
        normalizer["task_emb"] = get_identity_normalizer_from_stat(
            array_to_stats(self.task_emb)
        )
        if self.use_wrist_cam:
            normalizer["wrist_cam"] = get_image_range_normalizer()
        normalizer["measured_delta"] = get_range_normalizer_from_stat(
            self._measured_delta_stats
        )
        return normalizer

    def postprocess(self, samples, device) -> Dict[str, torch.Tensor]:
        result = super().postprocess(samples, device)
        if "task_id" not in samples:
            raise KeyError(
                "LIBERO dataset batch is missing 'task_id'. The dataset must be "
                "constructed with extra_keys=('task_id',) so the sampler carries it."
            )
        task_id = samples["task_id"].to(device, non_blocking=True).long()  # (B, T)
        result["obs"]["task_emb"] = self._task_emb_on(device)[task_id]     # (B, T, D)
        if self.use_wrist_cam:
            result["obs"]["wrist_cam"] = (
                samples["wrist_camera"].to(device, non_blocking=True) / 255.0
            )
        # Measured motion, derived from the pose history rather than from the
        # recorded commands. Sits at the batch top level (not under "obs") so the
        # obs encoder and the obs normalizer never see it — it is a flow source,
        # not a condition input.
        result["measured_delta"] = measured_delta_from_poses_torch(result["obs"]["agent_pos"])
        return result


def _obs_extra_keys(use_wrist_cam: bool):
    return ("task_id", "wrist_camera") if use_wrist_cam else ("task_id",)


class LiberoImageDataset(_LiberoObsMixin, RobotImageDataset):
    """Plain LIBERO dataset (fm_dit, a2a_dit, ddpm_dit, ... ) with language."""

    def __init__(self, zarr_path, use_wrist_cam=False, **kwargs):
        self.use_wrist_cam = bool(use_wrist_cam)
        kwargs.setdefault("extra_keys", _obs_extra_keys(self.use_wrist_cam))
        super().__init__(zarr_path=zarr_path, **kwargs)
        self._init_task_meta(zarr_path)


class LiberoImageDatasetKNN(_LiberoObsMixin, RobotImageDatasetKNN):
    """LIBERO dataset for MARS: language + per-task KNN neighbourhoods.

    ``knn_group_key`` defaults to ``"task_id"`` here. Setting it to ``None``
    reproduces the (incorrect for multi-task) global neighbourhood, which is
    occasionally useful as an ablation to show what task-mixing does to the
    router.
    """

    def __init__(self, zarr_path, use_wrist_cam=False, **kwargs):
        self.use_wrist_cam = bool(use_wrist_cam)
        kwargs.setdefault("extra_keys", _obs_extra_keys(self.use_wrist_cam))
        kwargs.setdefault("knn_group_key", "task_id")
        super().__init__(zarr_path=zarr_path, **kwargs)
        self._init_task_meta(zarr_path)

    def _history_signal_np(self, states, actions_norm, act_min, act_max, st_min, st_max):
        """Add the ``measured_delta`` source, normalized with its own range so it
        matches ``normalizer["measured_delta"]`` on the policy side.

        Runs during the parent's __init__ (before _init_task_meta), so it derives
        the deltas itself rather than relying on the cached stats.
        """
        if self.history_source == "measured_delta":
            md = self._measured_delta_all()
            return to_action_space_history_np(md, md.min(axis=0), md.max(axis=0))
        return super()._history_signal_np(
            states, actions_norm, act_min, act_max, st_min, st_max
        )
