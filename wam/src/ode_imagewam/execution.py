"""Explicit reference-carry execution. This module never connects to a robot."""
from __future__ import annotations

import torch

from .head import reference_to_command, state_to_reference
from .sidecar import validate_geometry


class ReferenceExecutor:
    """Track an ODE target against each live observation; commit only executed rows.

    `command` is a pure calculation. The caller sends it to its environment and
    calls `commit` only after successful execution. Stopping midway therefore
    carries the last executed target, not an unexecuted chunk tail.
    """
    def __init__(self, geometry=None):
        self.geometry = validate_geometry(geometry)
        self.reference = None

    def reset(self):
        self.reference = None

    def initial_reference(self, raw_state):
        if self.reference is not None:
            return self.reference.clone()
        state = torch.as_tensor(raw_state, dtype=torch.float64)
        if state.shape != (8,) or not torch.isfinite(state).all():
            raise ValueError("Expected eight finite raw proprioceptive values")
        return state_to_reference(state, self.geometry)

    def command(self, target, raw_state, grip):
        target = torch.as_tensor(target, dtype=torch.float64).reshape(6)
        state = torch.as_tensor(raw_state, dtype=torch.float64).reshape(8)
        if not torch.isfinite(target).all() or not torch.isfinite(state).all() or float(grip) not in (0, 1):
            raise ValueError("Expected finite state/target and binary gripper (0 closed, 1 open)")
        delta = reference_to_command(target, state, self.geometry).clamp(-1, 1)
        return torch.cat((delta, delta.new_tensor([1 - 2 * float(grip)]))).float().numpy()

    def commit(self, target):
        target = torch.as_tensor(target, dtype=torch.float64).reshape(6)
        if not torch.isfinite(target).all():
            raise ValueError("Cannot carry a nonfinite target")
        self.reference = target.detach().clone()
