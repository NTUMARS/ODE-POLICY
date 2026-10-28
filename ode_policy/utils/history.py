"""A2A history normalization: single source of truth.

Contract
--------
The A2A family (A2A, A2A-MARS, A2A-MARS-Router, ...) denoises from a history
action *in action-normalizer space* toward the future action. That means
everything fed into the flow's start point — and anything compared against it
(e.g. diversity-loss spreads, pre-computed KNN buffers) — MUST live in the
same action-normalizer space.

Two legal sources for the history signal:

* ``past_action``  — real executed actions from the last T steps. Always valid.
* ``agent_pos``    — raw state used as an action surrogate. Only valid when
                     ``state_dim == action_dim`` AND state/action share
                     semantics (e.g. metasim joint_pos where both are joint
                     positions). In that case applying the action normalizer
                     to the state is well-defined.

This module is the only place that performs the "raw history → action-space"
conversion. Both the dataset's KNN precompute (neighbor histories) and the
policy's runtime history (flow start point) route through here, so the two
sides cannot drift out of sync.

Why it exists
-------------
Prior versions had two independent normalization paths: the dataset normalized
neighbor states with the **state** normalizer while the policy normalized its
own state-as-history with the **action** normalizer. When state and action
ranges differed, the per-dim diversity loss compared two spreads living in
different numerical spaces, producing a systematically biased learning signal.
Routing both through this helper eliminates that class of bug.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


HistorySource = Literal["past_action", "agent_pos", "measured_delta"]


def to_action_space_history_np(
    raw: np.ndarray,
    act_min: np.ndarray,
    act_max: np.ndarray,
    range_eps: float = 1e-4,
    output_min: float = -1.0,
    output_max: float = 1.0,
) -> np.ndarray:
    """NumPy path for dataset-side precompute.

    Mirrors ``LinearNormalizer`` with ``mode='limits'`` and ``fit_offset=True``
    EXACTLY, including the constant-range fallback. The mapping is:

        normal dim (range >= range_eps):
            y = (raw - input_min) / range * (output_max - output_min) + output_min

        ignore dim (range < range_eps):
            y = raw - input_min      (scale=1, offset=-input_min)

    Why we can't use a flat 1e-8 epsilon: when an action dim is constant
    (range == 0) the policy's LinearNormalizer takes the ignore-dim branch
    and outputs values near zero (raw - min); a 1e-8 fallback divides by
    1e-8 instead, producing huge numbers and breaking the start_self vs
    nn_start comparison in the diversity loss.

    The torch-side computation lives in ``utils/normalizer.py::_fit`` /
    ``_normalize``; this helper reproduces it for numpy buffers used by the
    dataset's KNN precompute.
    """
    act_range = act_max - act_min
    ignore = act_range < range_eps
    # Avoid the divide; both branches are computed and selected via np.where.
    safe_range = np.where(ignore, 1.0, act_range)
    normal = (raw - act_min) / safe_range * (output_max - output_min) + output_min
    constant = raw - act_min  # mirrors LinearNormalizer's ignore-dim branch
    # Broadcast `ignore` (D,) over leading dims of `raw`.
    return np.where(ignore, constant, normal).astype(raw.dtype, copy=False)


def assert_history_compatible(
    source: HistorySource,
    state_dim: int,
    action_dim: int,
    owner: str = "A2A policy",
) -> None:
    """Validate the history_source config against the task shapes.

    Fails fast at ``__init__`` so dim-mismatched tasks never reach training with
    a silently miscalibrated flow start. ``past_action`` is always valid;
    ``agent_pos`` requires ``state_dim == action_dim`` because state is being
    reused as a history-action surrogate.
    """
    if source in ("past_action", "measured_delta"):
        # Both are constructed action-shaped by definition, so no dim check is
        # needed. "measured_delta" is the motion the arm actually achieved,
        # derived from consecutive measured poses and expressed in the
        # controller's action units.
        return
    if source == "agent_pos":
        if state_dim != action_dim:
            raise AssertionError(
                f"{owner}: history_source='agent_pos' treats the state vector as a "
                f"history-action surrogate, which requires state_dim == action_dim "
                f"(e.g. metasim joint_pos). Got state_dim={state_dim}, "
                f"action_dim={action_dim}. Use history_source='past_action' for "
                f"tasks where state and action have different semantics "
                f"(e.g. LIBERO: 8-dim pose state vs 7-dim delta action)."
            )
        return
    raise ValueError(
        f"{owner}: unknown history_source={source!r}; "
        f"expected 'agent_pos', 'past_action' or 'measured_delta'."
    )


def warn_if_ranges_diverge(
    act_min: np.ndarray,
    act_max: np.ndarray,
    state_min: np.ndarray,
    state_max: np.ndarray,
    owner: str = "A2A dataset",
    threshold: float = 0.5,
) -> None:
    """When using ``agent_pos`` history, state is normalized with the ACTION
    min/max. If the two ranges barely overlap, the state values end up mostly
    outside [-1, 1] after normalization and the diversity loss becomes noisy.

    Emits a warning (not an error) so users on edge-case joint_pos setups
    where ranges differ slightly can still train, but are aware of the risk.
    """
    act_range = np.maximum(act_max - act_min, 1e-8)
    state_range = np.maximum(state_max - state_min, 1e-8)
    inter_min = np.maximum(act_min, state_min)
    inter_max = np.minimum(act_max, state_max)
    inter = np.maximum(inter_max - inter_min, 0.0)
    # Per-dim IoU (1D interval): union = a + b − intersection.
    union = act_range + state_range - inter
    overlap = (inter / np.maximum(union, 1e-8)).mean()
    if overlap < threshold:
        import warnings
        warnings.warn(
            f"{owner}: history_source='agent_pos' normalizes state with the "
            f"action normalizer, but state/action ranges overlap only {overlap:.1%} "
            f"(per-dim IoU). Diversity-loss spreads may be miscalibrated. "
            f"Consider history_source='past_action' for this task.",
            RuntimeWarning,
            stacklevel=2,
        )
