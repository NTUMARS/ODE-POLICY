# Adapted for ODE-POLICY public integration; see THIRD_PARTY_NOTICES.md.
"""Offline ODE prediction helpers. This file never opens a robot connection."""

from __future__ import annotations

import numpy as np


class ExecutedBodyCarry:
    """Carry only the last actually issued body token; previews never update it."""

    def __init__(self, horizon=16):
        if type(horizon) is not int or not 1 <= horizon <= 16:
            raise ValueError("An H16 issued prefix must contain at most 16 control steps")
        self.horizon = horizon
        self.reset()

    def reset(self):
        self._last_issued = None
        self.executed_steps = 0

    def normalized_initial_body(self, normalizer):
        if self._last_issued is None:
            # An internal learned-coordinate BOS, not a neutral robot command.
            return np.zeros(64, dtype=np.float32)
        action = np.concatenate([self._last_issued, np.zeros(12, dtype=np.float32)])
        return normalizer.normalize_actions_unclipped(action)[..., :64].copy()

    def commit_executed(self, issued_actions):
        values = np.asarray(issued_actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 76 or not np.isfinite(values).all():
            raise ValueError("Expected finite actually issued physical actions [K,76]")
        if len(values) > self.horizon:
            raise ValueError("Issued actions exceed the configured H16 prediction window")
        if len(values):
            self._last_issued = values[-1, :64].copy()
            self.executed_steps += len(values)


def decode_actions(normalized_actions, normalizer):
    actions = np.asarray(normalizer.denormalize_actions(normalized_actions), dtype=np.float32)
    if actions.ndim < 2 or actions.shape[-1] != 76 or not np.isfinite(actions).all():
        raise ValueError("Invalid decoded SONIC64 plus left6/right6 states")
    actions = actions.copy()
    actions[..., 64:] = np.clip(actions[..., 64:], 0.0, 1.0)
    return actions
