"""ODEImagePolicy in joint space (R1 Lite: 6 joints + gripper, two cameras), on the CPU with a
tiny encoder. Guards the pose_dim == 6 trap: without joint_space the policy would parse the
7-D joint state as [pos(3), quat(4)]."""

import numpy as np
import pytest
import torch

from ode_policy.policies.ode.ode_image_policy import ODEImagePolicy, pose_signal_torch
from ode_policy.utils.normalize_util import get_image_range_normalizer
from ode_policy.utils.normalizer import LinearNormalizer
from ode_policy.utils.vision.model_getter import get_resnet
from ode_policy.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder

IMG = 32          # ResNet18 with GroupNorm works down to 32x32; keeps the CPU test fast
To = Ta = 8
H = To + Ta


def _shape_meta(state_dim=7, action_dim=7):
    return {"obs": {"head_cam": {"shape": [3, IMG, IMG], "type": "rgb"},
                    "wrist_cam": {"shape": [3, IMG, IMG], "type": "rgb"},
                    "agent_pos": {"shape": [state_dim], "type": "low_dim"}},
            "action": {"shape": [action_dim]}}


def _policy(**kw):
    sm = _shape_meta(kw.pop("state_dim", 7))
    enc = MultiImageObsEncoder(shape_meta=sm, rgb_model=get_resnet("resnet18", weights=None), resize_shape=None,
                               crop_shape=None, random_crop=True, use_group_norm=True, share_rgb_model=False,
                               imagenet_norm=True)
    defaults = dict(shape_meta=sm, obs_encoder=enc, horizon=H, n_action_steps=Ta, n_obs_steps=To,
                    hidden_dim=32, num_layers=2, cond_dim=32, solver="rk4", pose_dim=6, joint_space=True,
                    cond_use_s=False, perturb_prob=1.0, perturb_pos=0.02)
    defaults.update(kw)
    return ODEImagePolicy(**defaults)


def _normalizer(rng, pose_dim=6):
    """Fit on fake joint data: joints in [-1.5, 1.5] rad, gripper state in mm, ODE action = [target, +-1]."""
    joints = rng.uniform(-1.5, 1.5, size=(500, pose_dim)).astype(np.float32)
    state = np.concatenate([joints, rng.uniform(0, 100, size=(500, 1)).astype(np.float32)], axis=1)
    action = np.concatenate([joints + 0.01, rng.choice([-1.0, 1.0], size=(500, 1)).astype(np.float32)], axis=1)
    n = LinearNormalizer()
    n.fit({"action": action, "agent_pos": state}, last_n_dims=1, mode="limits")
    n["head_cam"] = get_image_range_normalizer()
    n["wrist_cam"] = get_image_range_normalizer()
    return n


def _batch(rng, B=2):
    joints = torch.as_tensor(rng.uniform(-1.0, 1.0, size=(B, 1, 6)).astype(np.float32))
    joints = joints + 0.005 * torch.arange(H, dtype=torch.float32)[None, :, None]   # a slow drift
    grip_state = torch.full((B, H, 1), 60.0)
    agent_pos = torch.cat([joints, grip_state], dim=-1)
    grip_cmd = torch.where(torch.arange(H)[None, :, None] < 10, 1.0, -1.0).expand(B, H, 1)
    action = torch.cat([joints + 0.01, grip_cmd], dim=-1)
    return {"obs": {"head_cam": torch.rand(B, H, 3, IMG, IMG), "wrist_cam": torch.rand(B, H, 3, IMG, IMG),
                    "agent_pos": agent_pos}, "action": action}


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    p = _policy()
    p.set_normalizer(_normalizer(np.random.default_rng(0)))
    return p


def test_live_pose_is_the_joint_vector(policy):
    ap = torch.arange(14, dtype=torch.float32).view(2, 7)
    torch.testing.assert_close(policy._live_pose(ap), ap[:, :6])
    # without the flag, pose_dim == 6 is still read as an end-effector pose (LIBERO behaviour)
    ee = _policy(joint_space=False, state_dim=8)
    ap8 = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, -1.0]])
    torch.testing.assert_close(ee._live_pose(ap8), pose_signal_torch(ap8))


def test_perturb_std_scalar_and_per_dim():
    assert _policy().perturb_std == [0.02] * 6
    assert _policy(perturb_pos=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]).perturb_std == [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    with pytest.raises(AssertionError):
        _policy(perturb_pos=[0.01, 0.02])
    # the LIBERO default is untouched
    assert _policy(joint_space=False, state_dim=8, perturb_pos=0.01, perturb_rot=0.03).perturb_std == [0.01] * 3 + [0.03] * 3


def test_compute_loss_is_finite_and_backprops(policy):
    batch = _batch(np.random.default_rng(1))
    policy.train()
    loss = policy.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    g = policy.field.inp.weight.grad
    assert g is not None and torch.isfinite(g).all()
    policy.zero_grad()


def test_predict_action_shapes_live_start_and_carry(policy):
    policy.eval()
    batch = _batch(np.random.default_rng(2))
    obs = batch["obs"]
    with torch.no_grad():
        out = policy.predict_action(obs)
    a = out["action"]
    assert a.shape == (2, Ta, 7) and torch.isfinite(a).all()
    assert set(a[..., 6].unique().tolist()) <= {-1.0, 1.0}
    # zero-initialised field: the first chunk stays at the live joints (row To-1 of agent_pos)
    live = obs["agent_pos"][:, To - 1, :6]
    assert (a[..., :6] - live[:, None]).abs().max() < 0.05
    # carried start: integrate from an explicit un-normalised a0 instead of the live joints
    a0 = live + 0.3
    with torch.no_grad():
        out2 = policy.predict_action(obs, a0=a0)
    assert (out2["action"][..., :6] - a0[:, None]).abs().max() < 0.05
    assert (out2["action"][..., :6] - a[..., :6]).abs().mean() > 0.1


def test_single_gripper_head_unchanged():
    # one gripper: the head still outputs n_action_steps logits, so the single-arm checkpoints load as before
    assert _policy().grip_head[-1].out_features == Ta


# ── two arms: a = [L1..L6, R1..R6], grippers [L, R] last; agent_pos in the wire order ─────────────────

CAMS3 = ("head_cam", "wrist_cam", "left_wrist_cam")
POSE_COLS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def _bi_policy(**kw):
    sm = {"obs": {**{k: {"shape": [3, IMG, IMG], "type": "rgb"} for k in CAMS3},
                  "agent_pos": {"shape": [14], "type": "low_dim"}},
          "action": {"shape": [14]}}
    enc = MultiImageObsEncoder(shape_meta=sm, rgb_model=get_resnet("resnet18", weights=None), resize_shape=None,
                               crop_shape=None, random_crop=True, use_group_norm=True, share_rgb_model=False,
                               imagenet_norm=True)
    defaults = dict(shape_meta=sm, obs_encoder=enc, horizon=H, n_action_steps=Ta, n_obs_steps=To,
                    hidden_dim=32, num_layers=2, cond_dim=32, solver="rk4", pose_dim=12, n_grippers=2,
                    state_pose_index=POSE_COLS, joint_space=True, cond_use_s=False, perturb_prob=1.0,
                    perturb_pos=0.02)
    defaults.update(kw)
    return ODEImagePolicy(**defaults)


def _bi_normalizer(rng):
    joints = rng.uniform(-1.5, 1.5, size=(500, 12)).astype(np.float32)
    widths = rng.uniform(0, 100, size=(500, 2)).astype(np.float32)
    state = np.concatenate([joints[:, :6], widths[:, :1], joints[:, 6:], widths[:, 1:]], axis=1)   # wire order
    action = np.concatenate([joints + 0.01, rng.choice([-1.0, 1.0], size=(500, 2)).astype(np.float32)], axis=1)
    n = LinearNormalizer()
    n.fit({"action": action, "agent_pos": state}, last_n_dims=1, mode="limits")
    for k in CAMS3:
        n[k] = get_image_range_normalizer()
    return n


def _bi_batch(rng, B=2, grip=(1.0, -1.0)):
    left = torch.as_tensor(rng.uniform(-1.0, 1.0, size=(B, 1, 6)).astype(np.float32))
    left = left + 0.005 * torch.arange(H, dtype=torch.float32)[None, :, None]
    right = left + 0.2
    width = torch.full((B, H, 1), 60.0)
    obs = {k: torch.rand(B, H, 3, IMG, IMG) for k in CAMS3}
    obs["agent_pos"] = torch.cat([left, width, right, width], dim=-1)                 # (B, H, 14) wire order
    g = torch.tensor(grip, dtype=torch.float32).expand(B, H, 2)
    return {"obs": obs, "action": torch.cat([left + 0.01, right + 0.01, g], dim=-1)}   # (B, H, 14) ODE order


@pytest.fixture(scope="module")
def bi_policy():
    torch.manual_seed(0)
    p = _bi_policy()
    p.set_normalizer(_bi_normalizer(np.random.default_rng(0)))
    return p


def test_two_arm_live_pose_skips_the_gripper_columns(bi_policy):
    ap = torch.arange(28, dtype=torch.float32).view(2, 14)
    torch.testing.assert_close(bi_policy._live_pose(ap), ap[:, POSE_COLS])
    with pytest.raises(AssertionError):
        _bi_policy(state_pose_index=list(range(6)))            # not pose_dim indices
    with pytest.raises(AssertionError):
        _bi_policy(state_pose_index=POSE_COLS[:-1] + [14])     # out of agent_pos
    with pytest.raises(AssertionError):
        _bi_policy(n_grippers=1)                               # action 14 != 12 + 1
    assert bi_policy.grip_head[-1].out_features == 2 * Ta


def test_two_arm_loss_and_prediction(bi_policy):
    batch = _bi_batch(np.random.default_rng(1))
    bi_policy.train()
    loss = bi_policy.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(bi_policy.field.inp.weight.grad).all()
    left_cam = list(bi_policy.obs_encoder.key_model_map["left_wrist_cam"].parameters())
    assert any(w.grad is not None and w.grad.abs().sum() > 0 for w in left_cam)   # the third camera is used
    bi_policy.zero_grad()
    bi_policy.eval()
    with torch.no_grad():
        a = bi_policy.predict_action(batch["obs"])["action"]
    assert a.shape == (2, Ta, 14) and torch.isfinite(a).all()
    assert set(a[..., 12:].unique().tolist()) <= {-1.0, 1.0}
    # zero-initialised field: both arms stay at their live joints, read from the joint columns of agent_pos
    live = batch["obs"]["agent_pos"][:, To - 1, POSE_COLS]
    assert (a[..., :12] - live[:, None]).abs().max() < 0.05


class _FixedGrip(torch.nn.Module):
    """Gripper logits +5 for the LEFT gripper and -5 for the RIGHT one at every step."""

    def forward(self, c):
        return torch.tensor([5.0, -5.0]).repeat(Ta).expand(c.shape[0], -1)


def test_two_arm_gripper_columns_are_not_swapped():
    torch.manual_seed(0)
    p = _bi_policy(gripper_loss_weight=0.1)
    p.set_normalizer(_bi_normalizer(np.random.default_rng(0)))
    p.grip_head = _FixedGrip()
    p.eval()
    batch = _bi_batch(np.random.default_rng(3), grip=(1.0, -1.0))   # left close, right open = what the head says
    act = batch["action"].clone()
    act[..., 12:] = -act[..., 12:]
    swapped_batch = {"obs": batch["obs"], "action": act}            # same frames, gripper targets swapped
    with torch.no_grad():
        a = p.predict_action(batch["obs"])["action"]
        assert torch.all(a[..., 12] == 1.0) and torch.all(a[..., 13] == -1.0)
        p.train()
        torch.manual_seed(5)
        match = p.compute_loss(batch)
        torch.manual_seed(5)
        swapped = p.compute_loss(swapped_batch)
    # same perturbation and pose loss; only the gripper BCE differs: 0.1 * (5.0067 - 0.0067)
    assert abs((swapped - match).item() - 0.5) < 1e-3
