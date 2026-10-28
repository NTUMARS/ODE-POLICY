"""
RobotImageDatasetKNN: extends RobotImageDataset with pre-computed global KNN.

Pre-computes per-sample:
  - target_spread: future diversity among global KNN neighbors (normalized)
  - nn_histories: neighbors' history actions (action-normalizer space), for start_spread

All stored history tensors live in ACTION-normalizer space (see
``ode_policy.utils.history`` for the contract). This guarantees the
diversity loss compares spreads in a single consistent numerical space.
"""

import copy
import numpy as np
import torch

from ode_policy.datasets.robot_image_dataset import (
    RobotImageDataset,
    batch_sample_sequence,
)
from ode_policy.utils.sampler import SequenceSampler
from ode_policy.utils.history import (
    assert_history_compatible,
    to_action_space_history_np,
    warn_if_ranges_diverge,
)


class RobotImageDatasetKNN(RobotImageDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=64,
        max_train_episodes=None,
        # KNN params
        knn_k=20,
        # Where to draw the history signal from (same contract as the policy;
        # see ``ode_policy.utils.history``).
        #   * "agent_pos"   (default) — use state as a history-action surrogate;
        #                   REQUIRES state_dim == action_dim (metasim joint_pos).
        #                   State is normalized with the ACTION normalizer so
        #                   dataset buffers align with the policy's flow start.
        #   * "past_action" — use real past actions; always valid. Pick this
        #                   whenever state and action have different semantics
        #                   (e.g. LIBERO: 8-dim pose vs 7-dim delta).
        history_source=None,
        # Deprecated alias. True → "past_action", False → "agent_pos".
        history_from_actions=None,
        # Future window for target_spread (must match the policy's future_actions
        # slicing). Defaults preserve legacy behavior (off=1, len=horizon).
        n_obs_steps=None,
        n_action_steps=None,
        # KNN neighbor key source:
        #   "state"  — n_obs-frame proprioceptive history (default/legacy).
        #   "image"  — downsampled CURRENT-frame image (hand-tuned pixel heuristic,
        #              see knn_image_mode). Position-aware but brittle / per-task.
        #   "resnet" — frozen pretrained ResNet-18 features of the CURRENT frame,
        #              L2-normalized, concatenated with the current-frame proprio
        #              (state). Task-agnostic semantic key; same recipe works for
        #              2D and 3D, so it replaces the redness/rgb split. RECOMMENDED.
        knn_key_source="state",
        # How to turn the image into the key (only used when knn_key_source="image"):
        #   "redness" — isolate a small red agent (R-(G+B)/2) + max-pool. For the 2D
        #               synthetic env where the agent is a ~2px red dot.
        #   "gray"    — plain grayscale mean-pool downsample. For real scenes where
        #               the subject (e.g. a robot arm) is large and survives averaging.
        #   "rgb"     — downsampled color, keeps color cues.
        #   "motion"  — downsampled color, but KEEP ONLY moving cells (drop near-static
        #               background) → raw-pixel key focused on task-relevant regions,
        #               texture-free (no mujoco collapse), color-preserving, unified.
        knn_image_mode="redness",
        # Downsample grid for the image key (cells per side). Larger = finer / less
        # info loss. Used by knn_image_mode rgb/gray/redness/motion.
        knn_image_grid=16,
        # Fraction of cells to KEEP (highest temporal saliency) for knn_image_mode=motion.
        knn_motion_keep=0.5,
        # Frozen image encoder for knn_key_source="resnet":
        #   "resnet18" (default; uses knn_resnet_weights), "dinov2_vits14" (384-d),
        #   "dinov2_vitb14" (768-d). DINOv2 = stronger, more render-robust features.
        knn_img_backbone="resnet18",
        # ResNet feature tap (only backbone=resnet18):
        #   "avgpool" (default, final 512-d semantic vector) or "layer2"/"layer3"
        #   (MID-LEVEL spatial feature map — lower-level structure/geometry, more
        #   domain-general → discriminative even on flat/synthetic renders like
        #   mujoco where the final ImageNet-semantic vector degrades).
        knn_resnet_layer="avgpool",
        # Spatial grid the mid-level feature map is adaptive-avg-pooled to (keeps
        # layout). Key dim = channels * pool^2 (layer3=256ch, layer2=128ch).
        knn_resnet_pool=3,
        # Grayscale the image before the encoder (replicate to 3ch). Drops color /
        # white-balance — a big chunk of the cross-sim render gap — keying on
        # structure. Composable with knn_resnet_layer.
        knn_img_gray=False,
        # Pretrained weights for the frozen ResNet-18 key (only when backbone=resnet18):
        #   "IMAGENET1K_V1" (default, universal) or "r3m" (manipulation-pretrained).
        knn_resnet_weights="IMAGENET1K_V1",
        # Relative weight of the proprio block vs the (unit-norm) image block when
        # building the resnet key. 1.0 → image and proprio contribute comparably.
        knn_proprio_weight=1.0,
        # Which zarr array supplies the proprio block of the resnet key. Must be a
        # POSITION-level signal (joint angles / agent xy), NOT velocity.
        #   "state"     — use `state` (default). For 3D joint_pos this IS the joint
        #                 angles, so it's already position-level. For the 2D nav env
        #                 `state` is VELOCITY, so 2D must point this elsewhere.
        #   "agent_pos" — read the dedicated true-position array (2D nav, regenerated
        #                 data). Read directly from the zarr; not a batch key.
        knn_proprio_key="state",
        # For knn_key_source="state": if True, the key is only the CURRENT frame's
        # state (1 frame) instead of the n_obs-frame history. Clean 1-frame
        # counterpart to the image/resnet keys.
        knn_state_current=False,
        # Which zarr array supplies the state KNN key (only for knn_key_source=state).
        # Must be POSITION-level: "state" (3D joint_pos = joint angles) or
        # "agent_pos" (2D nav, where `state` is velocity).
        knn_state_key="state",
        # Distance kernel for per-neighbor contribution. None → uniform (every
        # neighbor weighed equally, legacy). A float = softmax temperature over
        # -distance/sqrt(key_dim): closer neighbors contribute more (continuous,
        # subtle-variance-sensitive). Applies to target_spread + start_spread.
        knn_weight_temp=None,
        # Restrict neighbors to OTHER demos. The BallTree is built over every
        # sample window in the split, so without this the nearest neighbors of a
        # sample are usually its own demo's adjacent frames (t±1, t±2 have almost
        # identical state / image keys). Their future chunks are just time-shifted
        # copies of the same trajectory, so target_spread then measures how fast
        # the action changes over time instead of how much DIFFERENT demos disagree
        # at the same observation -- and it collapses toward 0 exactly at forks,
        # which is where the router is supposed to fire. True = drop every
        # same-episode candidate (this also drops self, so no positional
        # self-exclusion is needed). False = legacy global KNN.
        knn_cross_episode_only=True,
        # Name of a zarr `data/` array holding a per-frame GROUP label; when set,
        # neighbors are additionally restricted to the sample's own group. This
        # exists for multi-task datasets (LIBERO merges 10 tasks into one zarr,
        # labelled by `task_id`): across tasks the "disagreement" between future
        # chunks is just the difference between two unrelated tasks, which would
        # drive target_spread — and hence the router — high everywhere instead of
        # only at genuine within-task forks. None = single-task behavior, exactly
        # as before.
        knn_group_key=None,
        # Extra zarr arrays forwarded to the base dataset's replay buffer.
        extra_keys=(),
    ):
        super().__init__(
            zarr_path=zarr_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
            batch_size=batch_size,
            max_train_episodes=max_train_episodes,
            extra_keys=extra_keys,
        )
        self.knn_group_key = knn_group_key

        self.knn_k = knn_k
        self.zarr_path = zarr_path
        # Resolve history_source (new name) from either the new arg or the
        # legacy alias. If both are given they must agree.
        if history_from_actions is not None:
            import warnings
            warnings.warn(
                "RobotImageDatasetKNN: 'history_from_actions' is deprecated, "
                "use history_source='past_action'|'agent_pos' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            legacy = "past_action" if history_from_actions else "agent_pos"
            if history_source is not None and history_source != legacy:
                raise ValueError(
                    f"history_source={history_source!r} conflicts with "
                    f"history_from_actions={history_from_actions} (legacy → {legacy!r})."
                )
            history_source = legacy
        if history_source is None:
            history_source = "agent_pos"  # preserve legacy default
        self.history_source = history_source
        # n_obs_steps / n_action_steps are now REQUIRED — they pin the dataset's
        # history-window and future-window to exactly what the policy slices
        # at compute_loss time. The old fallback (n_obs = max(pad_before+1, 8),
        # fut_window = horizon) silently misaligned shapes whenever the user
        # set n_obs_steps to anything other than 8: the policy's
        # `start.shape[1]` would be n_obs_steps but the dataset's nn_histories
        # would be shape (·, ·, max(pad_before+1, 8), D), and the diversity
        # loss broadcast (`start.unsqueeze(1) - nn_start`) would crash. Fail
        # loud here instead of leaving the trap in place.
        assert n_obs_steps is not None and n_action_steps is not None, (
            "RobotImageDatasetKNN requires explicit n_obs_steps and "
            "n_action_steps (must match the policy's). Pass them via the "
            "dataset_config yaml: "
            "`n_obs_steps: ${n_obs_steps}` and `n_action_steps: ${n_action_steps}`."
        )
        self.n_obs = int(n_obs_steps)
        # Future window aligned with policy's
        # future_actions = nactions[:, n_obs-1 : n_obs-1 + n_action_steps].
        self.fut_window_offset = self.n_obs - 1
        self.fut_window_len = int(n_action_steps)
        assert knn_key_source in ("state", "image", "resnet"), knn_key_source
        assert knn_image_mode in ("redness", "gray", "rgb", "motion"), knn_image_mode
        self.knn_key_source = knn_key_source
        self.knn_image_mode = knn_image_mode
        self.knn_image_grid = int(knn_image_grid)
        self.knn_motion_keep = float(knn_motion_keep)
        self.knn_img_backbone = knn_img_backbone
        self.knn_resnet_layer = knn_resnet_layer
        self.knn_resnet_pool = int(knn_resnet_pool)
        self.knn_img_gray = bool(knn_img_gray)
        self.knn_resnet_weights = knn_resnet_weights
        self.knn_proprio_weight = float(knn_proprio_weight)
        self.knn_proprio_key = knn_proprio_key
        self.knn_state_current = bool(knn_state_current)
        self.knn_state_key = knn_state_key
        self.knn_weight_temp = knn_weight_temp
        self.knn_cross_episode_only = bool(knn_cross_episode_only)
        self._precompute_knn()

        # Pre-allocate buffers for batch data.
        # Per-dim target_spread: (batch_size, action_dim) — one spread number
        # per action dimension so the diversity loss can fire independently per
        # dim (matches per-dim router weight in the policy).
        action_dim_for_buf = self.replay_buffer['action'].shape[1]
        self.target_spread_buffer = np.zeros((batch_size, action_dim_for_buf), dtype=np.float32)
        self.target_spread_buffer_torch = torch.from_numpy(self.target_spread_buffer)

        # NOT min(knn_k, N-1): cross-episode filtering can lower the neighbor
        # count, and this buffer is indexed with per_sample_nn_histories.
        k = self._knn_k_effective
        n_obs = self.n_obs
        action_dim = self.replay_buffer['action'].shape[1]
        self.nn_hist_buffer = np.zeros(
            (batch_size, k, n_obs, action_dim), dtype=np.float32
        )
        self.nn_hist_buffer_torch = torch.from_numpy(self.nn_hist_buffer)

        # Per-neighbor contribution weights (distance kernel), (batch_size, k).
        self.nn_weights_buffer = np.zeros((batch_size, k), dtype=np.float32)
        self.nn_weights_buffer_torch = torch.from_numpy(self.nn_weights_buffer)

    def _history_signal_np(self, states, actions_norm, act_min, act_max, st_min, st_max):
        """The history signal, in the SAME normalized space the policy's flow
        source lives in — the diversity loss compares the two directly, so a
        mismatch here silently corrupts it.

        Subclasses add sources that need dataset-specific knowledge; the LIBERO
        dataset overrides this for ``measured_delta``, which has to be derived
        from consecutive poses.
        """
        if self.history_source == "past_action":
            return actions_norm
        if self.history_source == "agent_pos":
            warn_if_ranges_diverge(
                act_min, act_max, st_min, st_max,
                owner=type(self).__name__,
            )
            return to_action_space_history_np(states, act_min, act_max)
        raise ValueError(
            f"{type(self).__name__}: history_source={self.history_source!r} has no "
            f"dataset-side implementation. Override _history_signal_np in a subclass."
        )

    def _precompute_knn(self):
        """Pre-compute global KNN: target_spread + neighbor history actions."""
        from sklearn.neighbors import BallTree
        from ode_policy.utils.normalizer import LinearNormalizer

        indices = self.sampler.indices  # (N, 4)
        states = self.replay_buffer['state']
        actions = self.replay_buffer['action']
        episode_ends = self.replay_buffer.episode_ends[:]
        state_dim = states.shape[1]
        action_dim = actions.shape[1]

        # Validate history_source against task shapes — fails loud for
        # state_dim != action_dim tasks misconfigured to 'agent_pos'.
        assert_history_compatible(
            self.history_source, state_dim, action_dim,
            owner="RobotImageDatasetKNN",
        )

        # Compute normalizer (same as policy will use)
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": actions, "agent_pos": states},
            last_n_dims=1, mode="limits",
        )
        act_params = normalizer["action"].params_dict
        act_min = act_params["input_stats"]["min"].numpy()
        act_max = act_params["input_stats"]["max"].numpy()
        actions_norm = to_action_space_history_np(actions, act_min, act_max)

        # KNN key: normalized state via STATE stats. Used only by BallTree to
        # find similar obs-moments — we want every state dim to contribute
        # meaningfully to the L2 distance, so we rescale each dim into [-1, 1]
        # using its own range. (Using the action normalizer here would dwarf
        # state dims whose action counterparts are constant-range, e.g.
        # push_cube dims 0/1 → KNN would ignore those dims entirely.)
        st_params = normalizer["agent_pos"].params_dict
        st_min = st_params["input_stats"]["min"].numpy()
        st_max = st_params["input_stats"]["max"].numpy()
        states_norm_for_key = to_action_space_history_np(states, st_min, st_max)

        # History signal → always ACTION-normalizer space, regardless of source.
        # This is what the diversity loss compares (start_self vs nn_start);
        # both must live in the same numerical space as the policy's flow start.
        # * past_action: real past actions, trivially in action space.
        # * agent_pos:   state-as-action-surrogate; normalize with ACTION
        #                min/max so dataset buffers match the policy's runtime
        #                (policy's _get_history_actions calls
        #                 self.normalizer["action"].normalize). Fixes the
        #                state/action normalizer drift bug.
        history_norm = self._history_signal_np(
            states, actions_norm, act_min, act_max, st_min, st_max
        )

        n_samples = len(indices)
        n_obs = self.n_obs
        # Future window aligned with the policy's future_actions slice
        fut_chunk_len = max(1, self.fut_window_len)
        fut_offset = self.fut_window_offset

        ep_starts_arr = np.concatenate([[0], episode_ends[:-1]])

        # Build per-sample: history features (for KNN key),
        # history actions (normalized, for start_spread),
        # future action chunks (normalized, for target_spread)
        history_features = np.zeros((n_samples, n_obs * state_dim), dtype=np.float32)
        history_actions_norm = np.zeros((n_samples, n_obs, action_dim), dtype=np.float32)
        future_chunks = np.zeros((n_samples, fut_chunk_len, action_dim), dtype=np.float32)
        future_valid_len = np.zeros(n_samples, dtype=np.int64)
        sample_ep = np.zeros(n_samples, dtype=np.int64)   # episode (demo) of each sample

        for i in range(n_samples):
            buf_start = int(indices[i, 0])
            sample_start_idx = int(indices[i, 2])  # head pad length (>0 → pad_before sample)
            sample_end_idx = int(indices[i, 3])    # last real position (exclusive)
            ep_idx = np.searchsorted(episode_ends, buf_start, side='right')
            ep_s = int(ep_starts_arr[ep_idx]) if ep_idx < len(ep_starts_arr) else 0
            ep_e = int(episode_ends[ep_idx])
            sample_ep[i] = ep_idx

            # History window: first n_obs positions of the SEQUENCE (= what the
            # policy slices as history_actions[:, :n_obs]). Mirror the sampler's
            # replicate-pad behavior so this matches what the policy actually
            # sees at training time:
            #   sample[sample_start_idx : sample_end_idx] = real buffer data
            #   sample[: sample_start_idx]   = replica of the first real value (head pad)
            #   sample[sample_end_idx :]     = replica of the last real value  (tail pad)
            # Diversity loss compares start_self (built from policy's history) to
            # nn_start (built from this stored history); they must follow the
            # same pad semantics or padded-start samples decorrelate.
            real_lo = sample_start_idx
            real_hi = min(sample_end_idx, n_obs)
            # KNN key: normalized state (STATE normalizer) so each dim
            # contributes proportionally to L2 distance.
            hist = np.empty((n_obs, state_dim), dtype=np.float32)
            # nn_histories: history signal in ACTION-normalizer space.
            hist_act = np.empty((n_obs, action_dim), dtype=np.float32)
            key_src = states_norm_for_key  # state normalizer
            act_src = history_norm         # action normalizer
            if real_lo >= n_obs:
                # Sub-window is entirely head-pad (only with extreme pad_before).
                hist[:] = key_src[buf_start]
                hist_act[:] = act_src[buf_start]
            else:
                real_len = max(0, real_hi - real_lo)
                real_seg_key = key_src[buf_start : buf_start + real_len]
                real_seg_act = act_src[buf_start : buf_start + real_len]
                hist[real_lo:real_hi] = real_seg_key
                hist_act[real_lo:real_hi] = real_seg_act
                if real_lo > 0:  # head pad → first real value
                    hist[:real_lo] = real_seg_key[0]
                    hist_act[:real_lo] = real_seg_act[0]
                if real_hi < n_obs:  # tail pad → last real value (short-ep edge case)
                    hist[real_hi:] = real_seg_key[-1]
                    hist_act[real_hi:] = real_seg_act[-1]
            history_features[i] = hist.flatten()
            history_actions_norm[i] = hist_act

            # Future action chunks (normalized), pad-aware to mirror the
            # sampler. Policy sees
            #   future_actions = nactions[:, n_obs-1 : n_obs-1+n_action_steps]
            # where nactions applies replicate-padding: sample position p maps
            # to buffer buf_start + (p - sample_start_idx) inside the real
            # range, and replicates the first/last real value outside it. The
            # history branch above handles this; earlier versions of this
            # branch indexed actions_norm at buf_start + fut_offset directly,
            # which silently shifted the future window by sample_start_idx on
            # head-padded samples (the first n_obs-1 of each episode),
            # feeding the diversity loss a target_spread computed over a
            # completely different action window than the policy predicts.
            real_lo_f = max(fut_offset, sample_start_idx)
            real_hi_f = min(fut_offset + fut_chunk_len, sample_end_idx)
            out_lo = real_lo_f - fut_offset
            real_len_f = max(0, real_hi_f - real_lo_f)
            out_hi = out_lo + real_len_f
            if real_len_f > 0:
                buf_lo = buf_start + real_lo_f - sample_start_idx
                future_chunks[i, out_lo:out_hi] = actions_norm[buf_lo : buf_lo + real_len_f]
            if out_lo > 0:  # head pad → first real value (buf_start)
                future_chunks[i, :out_lo] = actions_norm[buf_start]
            if out_hi < fut_chunk_len:  # tail pad → last real value in sample window
                buf_end_real = buf_start + (sample_end_idx - sample_start_idx)
                future_chunks[i, out_hi:] = actions_norm[buf_end_real - 1]
            # All positions now match what the policy sees (including replicate
            # padding), so the full chunk is valid for the spread comparison.
            future_valid_len[i] = fut_chunk_len

        # Optionally replace the KNN key with an image-derived key (target_spread /
        # nn_histories stay action-based — only neighbor SELECTION changes).
        if getattr(self, "knn_key_source", "state") == "image":
            history_features = self._build_image_keys(indices)
            print(f"[KNN] key source = IMAGE (downsampled current frame), "
                  f"dim={history_features.shape[1]}")
        elif getattr(self, "knn_key_source", "state") == "resnet":
            # Proprio block of the resnet key must be POSITION-level. Default
            # "state" reuses the state normalizer key (= joint angles for 3D
            # joint_pos). 2D nav points knn_proprio_key at a dedicated position
            # array (state there is velocity), read straight from the zarr.
            if self.knn_proprio_key == "state":
                proprio_norm_for_key = states_norm_for_key
            else:
                import zarr
                zg = zarr.open(self.zarr_path, "r")
                if self.knn_proprio_key not in zg["data"]:
                    raise KeyError(
                        f"knn_proprio_key={self.knn_proprio_key!r} not in zarr "
                        f"{self.zarr_path} (data arrays: {list(zg['data'].array_keys())}). "
                        f"Regenerate the dataset with this position array."
                    )
                pos = np.asarray(zg["data"][self.knn_proprio_key]).astype(np.float32)
                assert pos.shape[0] == states.shape[0], (
                    f"knn_proprio_key {self.knn_proprio_key!r} has {pos.shape[0]} rows "
                    f"!= state {states.shape[0]}; must align frame-for-frame."
                )
                p_min = pos.min(axis=0)
                p_max = pos.max(axis=0)
                proprio_norm_for_key = to_action_space_history_np(pos, p_min, p_max)
            history_features = self._build_resnet_keys(indices, proprio_norm_for_key)
            _prop = (f"⊕ proprio[{self.knn_proprio_key}](w={self.knn_proprio_weight})"
                     if self.knn_proprio_weight > 0 else "(no proprio)")
            print(f"[KNN] key source = RESNET (frozen {self.knn_resnet_weights} "
                  f"ResNet-18 current frame {_prop}), dim={history_features.shape[1]}")
        elif (getattr(self, "knn_key_source", "state") == "state"
              and getattr(self, "knn_state_current", False)):
            # Current-frame state only (1 frame), not the n_obs history. Clean
            # 1-frame counterpart to the resnet/image keys — isolates "image vs
            # state" from "1 frame vs n_obs frames".
            cur = self._current_frame_indices(indices)
            # knn_state_key selects the POSITION-level array for the key: "state"
            # (default; 3D joint_pos = joint angles) or e.g. "agent_pos" (2D nav,
            # where `state` is velocity). Read directly from the zarr, per-dim
            # normalized with its own range.
            ks_key = getattr(self, "knn_state_key", "state")
            if ks_key == "state":
                key_arr = states_norm_for_key
            else:
                import zarr
                zg = zarr.open(self.zarr_path, "r")
                if ks_key not in zg["data"]:
                    raise KeyError(
                        f"knn_state_key={ks_key!r} not in zarr {self.zarr_path} "
                        f"(data arrays: {list(zg['data'].array_keys())}). "
                        f"Regenerate the dataset with this position array."
                    )
                arr = np.asarray(zg["data"][ks_key]).astype(np.float32)
                assert arr.shape[0] == states.shape[0], (
                    f"knn_state_key {ks_key!r} has {arr.shape[0]} rows != state "
                    f"{states.shape[0]}; must align frame-for-frame."
                )
                key_arr = to_action_space_history_np(arr, arr.min(axis=0), arr.max(axis=0))
            history_features = key_arr[cur].astype(np.float32)
            print(f"[KNN] key source = STATE (current frame only, key={ks_key!r}), "
                  f"dim={history_features.shape[1]}")

        # ---- neighbor groups (multi-task datasets) -------------------------- #
        # A neighbor drawn from a DIFFERENT TASK is meaningless for the diversity
        # loss: its future chunk differs because the task differs, not because
        # this moment is genuinely ambiguous. On a merged multi-task zarr
        # (LIBERO packs 10 tasks into one file, labelled by `task_id`) that would
        # push target_spread — and hence the router weight — high everywhere,
        # destroying the very signal the router is trained on.
        #
        # Restricting via one BallTree PER GROUP is both correct and cheaper than
        # filtering a single global tree: once ~90% of the candidates belong to
        # other tasks, a global tree would have to return ~n_samples candidates
        # per query for the filter to still yield k survivors.
        #
        # knn_group_key=None -> one group holding every sample, which reduces to
        # the previous single-tree behavior exactly.
        sample_group = None
        gkey = getattr(self, "knn_group_key", None)
        if gkey is not None:
            # Local import: the module's other zarr uses are inside conditional
            # branches that may not have run, so there is no module-level `zarr`.
            import zarr

            zg = zarr.open(self.zarr_path, "r")
            if gkey not in zg["data"]:
                raise KeyError(
                    f"knn_group_key={gkey!r} not in zarr {self.zarr_path} "
                    f"(data arrays: {list(zg['data'].array_keys())}). "
                    f"Regenerate the dataset with this label array."
                )
            group_arr = np.asarray(zg["data"][gkey]).reshape(-1)
            n_frames = self.replay_buffer["state"].shape[0]
            assert group_arr.shape[0] == n_frames, (
                f"knn_group_key {gkey!r} has {group_arr.shape[0]} rows != state "
                f"{n_frames}; must align frame-for-frame."
            )
            sample_group = group_arr[self._current_frame_indices(indices)].astype(np.int64)

        if sample_group is None:
            group_members = [np.arange(n_samples, dtype=np.int64)]
        else:
            group_members = [
                np.flatnonzero(sample_group == g).astype(np.int64)
                for g in np.unique(sample_group)
            ]
            sizes = [len(m) for m in group_members]
            print(f"[KNN] group key {gkey!r}: {len(group_members)} groups in split "
                  f"(sizes {min(sizes)}..{max(sizes)}) — neighbors stay inside a group.")

        cross_only = self.knn_cross_episode_only

        # Resolve one k for every group so neighbor_ids stays rectangular.
        caps = []
        for members in group_members:
            n_g = len(members)
            cap = n_g - 1  # self is always excluded
            if cross_only:
                ep_counts_g = np.bincount(sample_ep[members])
                if int((ep_counts_g > 0).sum()) >= 2:
                    cap = min(cap, n_g - int(ep_counts_g.max()))
                else:
                    print(f"[KNN] WARNING: knn_cross_episode_only=True but a group holds "
                          f"{int((ep_counts_g > 0).sum())} episode(s) — no cross-demo "
                          f"neighbor exists there. Falling back to the global "
                          f"(self-excluded) KNN for that group.")
            caps.append(max(cap, 0))
        k = min(self.knn_k, min(caps))
        if k <= 0:
            raise ValueError(
                f"[KNN] no valid neighbor exists: per-group caps={caps} "
                f"(groups={len(group_members)}, cross_episode_only={cross_only}). "
                f"Use more demos per task, or set knn_cross_episode_only=false."
            )
        if k < self.knn_k:
            print(f"[KNN] k reduced {self.knn_k} -> {k} (smallest group/episode budget).")

        print(f"[KNN] Building {len(group_members)} BallTree(s) over {n_samples} samples, k={k} ...")
        nn_dists = np.empty((n_samples, k), dtype=np.float64)
        neighbor_ids = np.empty((n_samples, k), dtype=np.int64)

        for members in group_members:
            feats = history_features[members]
            n_g = len(members)
            eps_g = sample_ep[members]
            ep_counts_g = np.bincount(eps_g)
            g_cross = cross_only and int((ep_counts_g > 0).sum()) >= 2
            tree = BallTree(feats)

            if g_cross:
                # Keep the k nearest candidates from a DIFFERENT episode. Querying
                # k + max_ep guarantees k survivors even if the sample's whole own
                # demo sorts ahead of them.
                max_ep = int(ep_counts_g.max())
                k_query = min(n_g, k + max_ep)
                for lo in range(0, n_g, 4096):  # chunked: k_query can be large
                    hi = min(lo + 4096, n_g)
                    d, j = tree.query(feats[lo:hi], k=k_query)
                    same = eps_g[j] == eps_g[lo:hi, None]  # (chunk, k_query)
                    # Stable argsort on the bool moves same-episode candidates to
                    # the back while keeping the distance order of the rest intact.
                    order = np.argsort(same, axis=1, kind="stable")
                    j_sel = np.take_along_axis(j, order, axis=1)[:, :k]
                    d_sel = np.take_along_axis(d, order, axis=1)[:, :k]
                    neighbor_ids[members[lo:hi]] = members[j_sel]  # local -> global
                    nn_dists[members[lo:hi]] = d_sel
            else:
                d, j = tree.query(feats, k=k + 1)
                neighbor_ids[members] = members[j[:, 1:]]  # drop self, local -> global
                nn_dists[members] = d[:, 1:]

        n_eps_in_split = int((np.bincount(sample_ep) > 0).sum())
        print(f"[KNN] neighbors: k={k}, {n_eps_in_split} episodes in split, "
              f"cross_episode_only={cross_only}, "
              f"groups={len(group_members)}{'' if sample_group is None else f' by {gkey!r}'}.")
        self._knn_k_effective = k
        # Kept for diagnostics / tests (e.g. asserting the cross-episode invariant).
        self.per_sample_neighbor_ids = neighbor_ids.astype(np.int64, copy=False)
        self.per_sample_ep = sample_ep

        # Per-neighbor contribution weights: closer neighbors weigh more.
        # knn_weight_temp=None → uniform (legacy: every neighbor equal). Otherwise a
        # softmax over -distance with PER-ROW ADAPTIVE bandwidth: each sample's
        # neighbor distances are normalized by that sample's own MEAN neighbor
        # distance, so "closer-than-typical" neighbors get more weight regardless of
        # local density / absolute scale (a fixed bandwidth goes uniform wherever
        # neighbors are all tightly/equally clustered, e.g. 2D deterministic
        # pre-fork). Used for BOTH target_spread (here) and start_spread (runtime).
        wtemp = getattr(self, "knn_weight_temp", None)
        if wtemp is not None:
            row_scale = nn_dists.mean(axis=1, keepdims=True) + 1e-9   # (N, 1) local scale
            logits = -nn_dists / (float(wtemp) * row_scale)          # (N, k)
            logits -= logits.max(axis=1, keepdims=True)
            nn_w = np.exp(logits)
            nn_w /= nn_w.sum(axis=1, keepdims=True)                   # (N, k), rows sum to 1
        else:
            nn_w = np.full((n_samples, k), 1.0 / max(k, 1), dtype=np.float32)
        self.per_sample_nn_weights = nn_w.astype(np.float32)

        # Compute per-sample, per-dim target_spread (normalized action space):
        # mean |sample_d − neighbor_d| over K neighbors and T future steps.
        # Per-dim so the policy's per-dim router weight can fire each dim
        # independently (signal dims push w_d up, noise dims keep it near 0).
        #
        # Per-neighbor valid length: each (sample, neighbor) pair uses its
        # own min(self_valid, neighbor_valid) — avoids one short-window
        # neighbor truncating the comparison for all 20.
        all_target_spread = np.zeros((n_samples, action_dim), dtype=np.float32)
        for i in range(n_samples):
            nn_ids = neighbor_ids[i]
            self_L = int(future_valid_len[i])
            if self_L <= 0:
                continue
            self_fut = future_chunks[i, :self_L, :]            # (T_self, D)
            # Per-neighbor: valid length = min(self_L, neighbor_L)
            per_nn_means = []
            per_nn_w = []
            for pos, j_idx in enumerate(nn_ids):
                pair_L = min(self_L, int(future_valid_len[j_idx]))
                if pair_L <= 0:
                    continue
                nn_fut_j = future_chunks[j_idx, :pair_L, :]    # (T_pair, D)
                diff = nn_fut_j - self_fut[:pair_L, :]          # (T_pair, D)
                per_nn_means.append(np.abs(diff).mean(axis=0))  # (D,) — mean over T_pair
                per_nn_w.append(nn_w[i, pos])
            if per_nn_means:
                M = np.stack(per_nn_means)                      # (V, D)
                wv = np.asarray(per_nn_w, dtype=np.float64)
                wv = wv / (wv.sum() + 1e-9)                     # renormalize over valid
                all_target_spread[i] = (wv[:, None] * M).sum(axis=0)  # distance-weighted

        self.per_sample_target_spread = all_target_spread  # (N, D)
        # Store neighbor history actions (normalized): (N, k, n_obs, D)
        self.per_sample_nn_histories = history_actions_norm[neighbor_ids]

        # Print per-dim summary so users can see which dims carry the bimodal
        # signal in the data.
        per_dim_mean = all_target_spread.mean(axis=0)   # (D,)
        per_dim_std = all_target_spread.std(axis=0)
        print(f"[KNN] per-dim target_spread (action_dim={action_dim}):")
        print(f"      mean per dim: {np.round(per_dim_mean, 4).tolist()}")
        print(f"      std per dim:  {np.round(per_dim_std, 4).tolist()}")
        print(f"      overall mean={all_target_spread.mean():.4f}, "
              f"max={all_target_spread.max():.4f}")

    def _build_image_keys(self, indices, grid=None):
        """KNN key = downsampled AGENT-isolating map of each sample's CURRENT frame.

        The current obs frame is window position n_obs-1 (the anchor's present),
        mapped to its absolute buffer index with the sampler's replicate-pad
        clamping.

        To avoid the small agent being averaged away by downsampling, we isolate
        it by COLOR (the agent is red: redness = R - (G+B)/2 ≈ 150 for the agent,
        ≈0 for the gray/dark background, walls and goal) and downsample with
        MAX-pooling (keeps the agent's peak in its cell instead of diluting it).
        The resulting grid×grid map is essentially "where is the agent", so KNN
        neighbors = same-position moments — robust regardless of agent size.
        """
        imgs = np.asarray(self.replay_buffer["head_camera"])  # (N, 3, H, W) uint8
        H, W = imgs.shape[2], imgs.shape[3]
        if grid is None:
            grid = int(getattr(self, "knn_image_grid", 16))
        n_obs = self.n_obs
        cur = np.empty(len(indices), dtype=np.int64)
        for i in range(len(indices)):
            buf_start = int(indices[i, 0])
            buf_end = int(indices[i, 1])
            sstart = int(indices[i, 2])  # head-pad length
            idx = buf_start + (n_obs - 1 - sstart)
            cur[i] = min(max(idx, buf_start), buf_end - 1)
        sel = imgs[cur].astype(np.float32)          # (M, 3, H, W)
        fh = max(1, H // grid); fw = max(1, W // grid)
        sh, sw = H // fh, W // fw
        mode = getattr(self, "knn_image_mode", "redness")
        if mode in ("rgb", "motion"):
            # downsampled COLOR (3 channels) — keeps color cues (e.g. which cube
            # is which in stack_cube). Key dim = 3 * sh * sw.
            feat = sel[:, :, : sh * fh, : sw * fw]                       # (M, 3, H', W')
            down = feat.reshape(len(cur), 3, sh, fh, sw, fw).mean(axis=(3, 5))  # (M, 3, sh, sw)
            if mode == "motion":
                # Keep only MOVING cells (task-relevant); drop near-static ones
                # (background/table/walls — zero info for distinguishing situations).
                # Per-cell temporal saliency = std over samples, summed over channels.
                cell_sal = down.std(axis=0).sum(axis=0).reshape(-1)     # (sh*sw,)
                keep_frac = float(getattr(self, "knn_motion_keep", 0.5))
                n_keep = max(1, int(round(keep_frac * cell_sal.size)))
                keep_cells = np.argsort(cell_sal)[::-1][:n_keep]        # top-motion cells
                down_flat = down.reshape(len(cur), 3, sh * sw)          # (M, 3, sh*sw)
                key = down_flat[:, :, keep_cells].reshape(len(cur), -1)  # (M, 3*n_keep), RGB kept
                print(f"[KNN] image mode=motion grid={grid}: kept {n_keep}/{sh*sw} "
                      f"moving cells (keep_frac={keep_frac}), dim={key.shape[1]}")
                return (key / 255.0).astype(np.float32)
        elif mode == "gray":
            # plain grayscale mean-pool — drops color; ok if the subject is large
            # and color isn't discriminative.
            feat = sel.mean(axis=1)                  # (M, H, W)
            feat = feat[:, : sh * fh, : sw * fw]
            down = feat.reshape(len(cur), sh, fh, sw, fw).mean(axis=(2, 4))
        else:
            # redness + MAX-pool — isolate & preserve a small red agent (2D env)
            feat = np.clip(sel[:, 0] - 0.5 * (sel[:, 1] + sel[:, 2]), 0.0, None)
            feat = feat[:, : sh * fh, : sw * fw]
            down = feat.reshape(len(cur), sh, fh, sw, fw).max(axis=(2, 4))
        return (down.reshape(len(cur), -1) / 255.0).astype(np.float32)

    def _current_frame_indices(self, indices):
        """Absolute buffer index of each sample's CURRENT obs frame (window pos
        n_obs-1), with the sampler's replicate-pad clamping. Shared by the
        image / resnet key builders so both anchor on the same frame.
        """
        n_obs = self.n_obs
        cur = np.empty(len(indices), dtype=np.int64)
        for i in range(len(indices)):
            buf_start = int(indices[i, 0])
            buf_end = int(indices[i, 1])
            sstart = int(indices[i, 2])  # head-pad length
            idx = buf_start + (n_obs - 1 - sstart)
            cur[i] = min(max(idx, buf_start), buf_end - 1)
        return cur

    def _build_resnet_keys(self, indices, proprio_norm_for_key):
        """KNN key = frozen pretrained ResNet-18 features of each sample's CURRENT
        frame (L2-normalized) ⊕ that frame's proprio (state).

        Rationale (see discussion): a hand-tuned pixel heuristic (redness/rgb +
        downsample) is brittle and task-specific. A frozen pretrained CNN maps the
        current frame into a SEMANTIC space where neighbor = same scene/situation,
        with no per-task tuning — the same recipe works for 2D and 3D. We append
        the current-frame proprio so the key also reflects the robot's own
        configuration (the scene alone can't disambiguate joint pose).

        Balancing: the image block is L2-normalized to a unit vector (each sample
        contributes magnitude 1 to the distance); the proprio block (STATE-
        normalized, ~[-1,1] per dim) is scaled by 1/sqrt(state_dim) so its overall
        magnitude is also ~1, then by knn_proprio_weight. This keeps the two
        modalities comparable regardless of the 512-d vs state_dim mismatch.
        """
        imgs = np.asarray(self.replay_buffer["head_camera"])  # (N, 3, H, W) uint8
        cur = self._current_frame_indices(indices)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Frozen image encoder for the KNN key. "resnet18" (ImageNet/r3m, 512-d) or
        # a DINOv2 ViT ("dinov2_vits14"=384-d, "dinov2_vitb14"=768-d) — DINOv2's
        # self-supervised features are stronger/more render-robust, which matters on
        # weak renders (e.g. mujoco) where ImageNet ResNet features degrade.
        backbone = getattr(self, "knn_img_backbone", "resnet18")
        layer = getattr(self, "knn_resnet_layer", "avgpool")
        pool = int(getattr(self, "knn_resnet_pool", 3))
        gray = bool(getattr(self, "knn_img_gray", False))
        is_dino = backbone.startswith("dinov2")
        mid_level = (not is_dino) and layer in ("layer2", "layer3")
        if is_dino:
            net = torch.hub.load("facebookresearch/dinov2", backbone)
            resize_to = 224  # DINOv2 needs H,W multiples of 14
        else:
            from ode_policy.utils.vision.model_getter import get_resnet
            net = get_resnet(name="resnet18", weights=self.knn_resnet_weights)
            if mid_level:
                # Truncate to a mid layer for its spatial feature map. torchvision
                # resnet children: conv1,bn1,relu,maxpool,layer1,layer2,layer3,...
                ncut = {"layer2": 6, "layer3": 7}[layer]
                net = torch.nn.Sequential(*list(net.children())[:ncut])
            resize_to = None
        net = net.to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)

        # ImageNet normalization (matches both ResNet and DINOv2 pretraining).
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        feats = []
        bs = 128 if is_dino else 256
        with torch.no_grad():
            for s in range(0, len(cur), bs):
                chunk = imgs[cur[s:s + bs]].astype(np.float32) / 255.0  # (b,3,H,W)
                t = torch.from_numpy(chunk).to(device)
                if gray:  # luminance, replicated to 3 channels (drops color)
                    g = (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]).unsqueeze(1)
                    t = g.expand(-1, 3, -1, -1)
                if resize_to is not None:
                    t = torch.nn.functional.interpolate(
                        t, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
                t = (t - mean) / std
                f = net(t)
                if mid_level:
                    # (b, C, h, w) → adaptive-pool to (pool,pool), keep layout → flatten
                    f = torch.nn.functional.adaptive_avg_pool2d(f, pool).flatten(1)
                f = torch.nn.functional.normalize(f, dim=1)  # unit norm per sample
                feats.append(f.cpu().numpy())
        img_feat = np.concatenate(feats, axis=0).astype(np.float32)  # (M, feat_dim)
        _desc = f"{backbone}:{layer}(pool{pool})" if mid_level else backbone
        print(f"[KNN] image encoder = {_desc}{' +gray' if gray else ''}, "
              f"feat_dim={img_feat.shape[1]}")

        # Free the backbone before the policy claims the GPU.
        del net
        if device == "cuda":
            torch.cuda.empty_cache()

        # knn_proprio_weight <= 0 → pure ResNet key (no proprio). Appending proprio
        # can split "same scene, different joint pose" samples apart and suppress
        # cross-mode stitching, so this is a first-class option.
        if self.knn_proprio_weight <= 0:
            return img_feat

        # Current-frame proprio (position-level, normalized), scaled for balance.
        prop = proprio_norm_for_key[cur].astype(np.float32)         # (M, proprio_dim)
        proprio_dim = prop.shape[1]
        prop_scaled = prop / np.sqrt(max(proprio_dim, 1)) * self.knn_proprio_weight

        return np.concatenate([img_feat, prop_scaled], axis=1).astype(np.float32)

    def get_validation_dataset(self):
        """Build a val-side dataset with its OWN KNN precompute.

        Shallow-copying without rerunning ``_precompute_knn`` would leave
        ``val_set.per_sample_target_spread`` / ``per_sample_nn_histories``
        keyed by the TRAIN sampler indices while ``val_set.sampler`` uses
        the VAL indices. ``__getitem__`` would then index the train-keyed
        arrays with val indices — silently wrong data, scrambling the
        diversity term in val_loss (and IndexError if val is larger).

        Re-running KNN on the val sampler is cheap (val_ratio is typically
        small, ~hundreds of samples → BallTree takes well under a second).

        We keep the batch-local scratch buffers (target_spread_buffer /
        nn_hist_buffer) shared with the train set: trainer loops are
        serial (train epoch → val pass), so the next ``__getitem__`` call
        overwrites the previous batch's scratch only after the previous
        batch has been moved to device by ``postprocess``.
        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        # Recompute KNN against val sampler.indices so per_sample_* arrays
        # are correctly keyed for val.__getitem__.
        val_set._precompute_knn()
        # The k-shaped scratch buffers stay shared with the train set only while
        # both splits ended up with the same neighbor count. Cross-episode
        # filtering caps k at (n_samples - largest demo), which a small val split
        # can hit, so give val its own buffers whenever the two disagree.
        if val_set._knn_k_effective != self._knn_k_effective:
            k_val = val_set._knn_k_effective
            print(f"[KNN] val split uses k={k_val} (train k={self._knn_k_effective}) "
                  f"— allocating separate neighbor buffers for val.")
            batch_size, _, n_obs, action_dim = self.nn_hist_buffer.shape
            val_set.nn_hist_buffer = np.zeros(
                (batch_size, k_val, n_obs, action_dim), dtype=np.float32
            )
            val_set.nn_hist_buffer_torch = torch.from_numpy(val_set.nn_hist_buffer)
            val_set.nn_weights_buffer = np.zeros((batch_size, k_val), dtype=np.float32)
            val_set.nn_weights_buffer_torch = torch.from_numpy(val_set.nn_weights_buffer)
        return val_set

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            # Gather pre-computed data for this batch
            self.target_spread_buffer[:] = self.per_sample_target_spread[idx]
            self.nn_hist_buffer[:] = self.per_sample_nn_histories[idx]
            self.nn_weights_buffer[:] = self.per_sample_nn_weights[idx]

            result = dict(self.buffers_torch)
            result["target_spread"] = self.target_spread_buffer_torch
            result["nn_histories"] = self.nn_hist_buffer_torch
            result["nn_weights"] = self.nn_weights_buffer_torch
            return result
        else:
            return super().__getitem__(idx)

    def postprocess(self, samples, device):
        result = super().postprocess(samples, device)
        if "target_spread" in samples:
            result["target_spread"] = samples["target_spread"].to(device, non_blocking=True)
        if "nn_histories" in samples:
            result["nn_histories"] = samples["nn_histories"].to(device, non_blocking=True)
        if "nn_weights" in samples:
            result["nn_weights"] = samples["nn_weights"].to(device, non_blocking=True)
        return result
