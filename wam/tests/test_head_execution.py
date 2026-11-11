import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from ode_imagewam.execution import ReferenceExecutor
from ode_imagewam.head import (ActionODEHead, command_to_reference, reference_to_command,
                              rotvec_to_quat, state_to_reference)
from ode_imagewam.sidecar import DEFAULT_GEOMETRY, validate_geometry


def test_world_frame_geometry_matches_scipy_and_roundtrips():
    generator = np.random.default_rng(3)
    states = generator.normal(size=(30, 8)) * 0.3
    commands = generator.uniform(-0.7, 0.7, (30, 7))
    states[0, 3:6] = 0  # removable-singularity edge
    ref = command_to_reference(torch.tensor(states), torch.tensor(commands))
    expected_rotation = (Rotation.from_rotvec(commands[:, 3:6] * 0.5)
                         * Rotation.from_rotvec(states[:, 3:6])
                         * Rotation.from_quat(DEFAULT_GEOMETRY["canonical_quat"]).inv()).as_rotvec()
    np.testing.assert_allclose(ref[:, :3], states[:, :3] + 0.05 * commands[:, :3], atol=1e-12)
    np.testing.assert_allclose(ref[:, 3:], expected_rotation, atol=1e-12)
    np.testing.assert_allclose(reference_to_command(ref, torch.tensor(states)), commands[:, :6], atol=2e-12)


def test_geometry_backward_at_zero_and_configurable_scales():
    rotation = torch.zeros(4, 3, requires_grad=True)
    rotvec_to_quat(rotation).sum().backward()
    assert torch.isfinite(rotation.grad).all()
    geometry = {**DEFAULT_GEOMETRY, "position_scale": 0.1, "rotation_scale": 0.25}
    state = torch.zeros(8, dtype=torch.float64)
    action = torch.tensor([0.2, -0.3, 0.1, 0.1, 0, 0, 1], dtype=torch.float64)
    reference = command_to_reference(state, action, geometry)
    np.testing.assert_allclose(reference_to_command(reference, state, geometry), action[:6], atol=1e-12)


@pytest.mark.parametrize("invalid", [{"rotation_sacle": 0.8}, [0.5], {"rotation_scale": 0},
                                     {"position_scale": float("nan")}, {"canonical_quat": [0, 0, 0, 0]}])
def test_invalid_geometry_cannot_silently_use_defaults(invalid):
    with pytest.raises(ValueError):
        validate_geometry(invalid)


def test_h12_euler_fp32_and_backward():
    torch.manual_seed(2)
    head = ActionODEHead(feature_dim=8, cond_dim=8, width=16, layers=2).to(torch.bfloat16)
    assert head.horizon == 12
    with torch.no_grad():
        head.out.bias.fill_(0.25)
    initial = torch.randn(2, 6)
    feature = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    trajectory, grip = head(feature, torch.randn(2, 8), initial)
    expected = initial[:, None] + torch.arange(1, 13)[None, :, None] * (0.05 * 0.25)
    torch.testing.assert_close(trajectory, expected, atol=8e-7, rtol=1e-6)
    assert trajectory.dtype == grip.dtype == torch.float32
    (trajectory.square().mean() + grip.square().mean()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
    assert torch.isfinite(feature.grad).all()
    with pytest.raises(ValueError, match="horizon"):
        head(feature, torch.randn(2, 8), initial, 13)


def test_carry_only_committed_target_and_live_feedback():
    executor = ReferenceExecutor()
    state = np.zeros(8)
    initial = executor.initial_reference(state)
    torch.testing.assert_close(initial, state_to_reference(torch.tensor(state)))
    targets = initial.repeat(12, 1)
    targets[:, 0] = torch.arange(1, 13) * 0.01
    first = executor.command(targets[0], state, 1)
    state_after = state.copy()
    state_after[0] = 0.008
    adjusted = executor.command(targets[0], state_after, 1)
    assert first[0] == pytest.approx(0.2)
    assert adjusted[0] == pytest.approx(0.04)
    assert first[-1] == -1
    executor.commit(targets[2])  # interrupted after three rows, not twelve
    torch.testing.assert_close(executor.initial_reference(state_after), targets[2])
    assert not torch.equal(executor.reference, targets[-1])
    executor.reset()
    torch.testing.assert_close(executor.initial_reference(state_after), state_to_reference(torch.tensor(state_after)))
