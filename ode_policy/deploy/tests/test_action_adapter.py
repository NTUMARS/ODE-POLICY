"""Unit tests for ActionAdapter: expand (N, 7) policy output → (N, 14) client format."""

import numpy as np
import pytest

from ode_policy.deploy.adapters import ActionAdapter


def test_action_adapter_shape_and_dtype():
    action_7 = np.arange(8 * 7, dtype=np.float32).reshape(8, 7)
    left_echo = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    out = ActionAdapter.adapt(action_7, left_echo, n_action_steps=8)
    assert out.shape == (8, 14)
    assert out.dtype == np.float32


def test_action_adapter_left_arm_echoed_constant():
    action_7 = np.zeros((8, 7), dtype=np.float32)
    left_echo = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    out = ActionAdapter.adapt(action_7, left_echo, n_action_steps=8)
    for t in range(8):
        np.testing.assert_allclose(out[t, 0:7], left_echo)


def test_action_adapter_right_arm_placement():
    action_7 = np.zeros((8, 7), dtype=np.float32)
    action_7[3, :] = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)  # one row marked
    left_echo = np.zeros(7, dtype=np.float32)
    out = ActionAdapter.adapt(action_7, left_echo, n_action_steps=8)
    # Right arm joints land in [7:13], gripper in [13]
    np.testing.assert_allclose(out[3, 7:13], np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
    assert out[3, 13] == pytest.approx(7.0)
    # Other rows must be zero
    other = np.delete(out, 3, axis=0)
    assert np.all(other == 0)


def test_action_adapter_handles_n_action_steps_mismatch():
    # If action_7 has fewer rows than n_action_steps, raise.
    action_7 = np.zeros((4, 7), dtype=np.float32)
    left_echo = np.zeros(7, dtype=np.float32)
    with pytest.raises((ValueError, AssertionError)):
        ActionAdapter.adapt(action_7, left_echo, n_action_steps=8)


def test_action_adapter_two_arm_rows_pass_through():
    # a two-arm policy already speaks the wire layout [L1..L6, L_grip, R1..R6, R_grip]: no echo
    action_14 = np.arange(8 * 14, dtype=np.float64).reshape(8, 14)
    out = ActionAdapter.adapt(action_14, np.full(7, -1.0, dtype=np.float32), n_action_steps=8)
    assert out.shape == (8, 14) and out.dtype == np.float32
    np.testing.assert_allclose(out, action_14)
    with pytest.raises(ValueError):
        ActionAdapter.adapt(np.zeros((8, 9), dtype=np.float32), np.zeros(7, dtype=np.float32), n_action_steps=8)
