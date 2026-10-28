"""ODEChunkedInference: the server-side chunk schedule, carried state and gripper mapping,
checked against a fake ODE policy (no checkpoint, no GPU)."""

import numpy as np
import pytest
import torch

from ode_policy.deploy.ode_inference import ODEChunkedInference

P, N = 6, 8


class _FakeODEPolicy:
    """Deterministic stand-in: row i of a chunk = start + (i + 1) * 0.01 on every joint, where
    start is a0 if given else the live joints (obs_dict['agent_pos'][:, -1, :P]); gripper +1 on
    the first four rows, -1 after. Records every a0 it was integrated from."""

    pose_dim = P
    n_action_steps = N
    side_k = 0

    def __init__(self):
        self.a0_calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_action(self, obs_dict, a0=None, **kw):
        self.a0_calls.append(None if a0 is None else a0[0].cpu().numpy().copy())
        start = obs_dict["agent_pos"][:, -1, :P] if a0 is None else a0
        steps = torch.arange(1, N + 1, dtype=torch.float32)[None, :, None] * 0.01
        traj = start[:, None, :] + steps
        grip = torch.tensor([1.0] * 4 + [-1.0] * 4)[None, :, None]
        return {"action": torch.cat([traj, grip], dim=-1)}


def _obs(live_joints):
    ap = torch.zeros(1, 8, 7)
    ap[:, :, :P] = torch.as_tensor(live_joints, dtype=torch.float32)
    ap[:, :, P] = 50.0
    return {"head_cam": torch.zeros(1, 8, 3, 8, 8), "wrist_cam": torch.zeros(1, 8, 3, 8, 8), "agent_pos": ap}


def _make(policy, **kw):
    return ODEChunkedInference(policy, n_obs_steps=8, n_action_steps=N, horizon=16, device=torch.device("cpu"), **kw)


def test_chunk_schedule_replan_every_8_and_carry_state():
    pol = _FakeODEPolicy()
    inf = _make(pol, replan_every=8, gripper_close_cmd=0.0, gripper_open_cmd=100.0)
    live = np.full(P, 0.5, dtype=np.float32)
    chunk0 = live[None] + (np.arange(1, N + 1)[:, None] * 0.01)   # what the fake integrates from the live pose

    outs = [inf.infer(_obs(live)) for _ in range(8)]
    assert len(pol.a0_calls) == 1 and pol.a0_calls[0] is None     # first chunk: from the live joints
    for k, out in enumerate(outs):
        a = out["action"]
        assert a.shape == (N, P + 1) and a.dtype == np.float32
        assert out["replanned"] == (k == 0) and out["chunk_index"] == k
        # row 0 at tick k is the k-th target of the chunk; the rest is the remaining chunk, tail held
        np.testing.assert_allclose(a[:, :P], np.concatenate([chunk0[k:], np.repeat(chunk0[-1:], k, axis=0)]), atol=1e-6)
    # gripper +1/-1 mapped to mm: rows 0..3 close (0), 4..7 open (100); shifted with the chunk
    np.testing.assert_allclose(outs[0]["action"][:, P], [0, 0, 0, 0, 100, 100, 100, 100])
    np.testing.assert_allclose(outs[3]["action"][:, P], [0, 100, 100, 100, 100, 100, 100, 100])
    np.testing.assert_allclose(outs[7]["action"][:, P], [100] * 8)

    # tick 8: a new chunk, integrated from the LAST TARGET EXECUTED (row 7 of chunk 0), not the live joints
    out8 = inf.infer(_obs(live))
    assert out8["replanned"] and out8["chunk_index"] == 0 and len(pol.a0_calls) == 2
    np.testing.assert_allclose(pol.a0_calls[1], chunk0[7], atol=1e-6)
    np.testing.assert_allclose(out8["action"][0, :P], chunk0[7] + 0.01, atol=1e-6)
    assert inf.n_replans == 2


def test_replan_every_1_is_closed_loop_with_carry():
    pol = _FakeODEPolicy()
    inf = _make(pol, replan_every=1)
    live = np.zeros(P, dtype=np.float32)
    o1 = inf.infer(_obs(live))
    o2 = inf.infer(_obs(live))
    o3 = inf.infer(_obs(live))
    assert len(pol.a0_calls) == 3 and all(o["replanned"] for o in (o1, o2, o3))
    # each chunk continues from the single row the client executed from the previous one
    np.testing.assert_allclose(pol.a0_calls[1], o1["action"][0, :P], atol=1e-6)
    np.testing.assert_allclose(pol.a0_calls[2], o2["action"][0, :P], atol=1e-6)
    np.testing.assert_allclose(o3["action"][0, :P], np.full(P, 0.03), atol=1e-6)


def test_no_carry_state_always_starts_from_live_joints():
    pol = _FakeODEPolicy()
    inf = _make(pol, replan_every=2, carry_state=False)
    for _ in range(5):
        inf.infer(_obs(np.zeros(P, dtype=np.float32)))
    assert len(pol.a0_calls) == 3 and all(a is None for a in pol.a0_calls)


def test_reset_episode_forgets_chunk_and_state():
    pol = _FakeODEPolicy()
    inf = _make(pol, replan_every=8)
    for _ in range(8):
        inf.infer(_obs(np.zeros(P, dtype=np.float32)))
    inf.reset_episode()
    assert pol.resets >= 2   # once at construction, once now
    out = inf.infer(_obs(np.ones(P, dtype=np.float32)))
    assert out["replanned"] and pol.a0_calls[-1] is None and inf.n_replans == 1
    np.testing.assert_allclose(out["action"][0, :P], np.full(P, 1.01), atol=1e-6)


def test_custom_gripper_values_and_sign():
    pol = _FakeODEPolicy()
    inf = _make(pol, gripper_close_cmd=12.5, gripper_open_cmd=87.5)
    a = inf.infer(_obs(np.zeros(P, dtype=np.float32)))["action"]
    np.testing.assert_allclose(a[:, P], [12.5] * 4 + [87.5] * 4)


def test_gripper_ramp_follows_the_head_from_the_measured_width():
    pol = _FakeODEPolicy()
    inf = _make(pol, replan_every=8, gripper_close_cmd=0.0, gripper_open_cmd=100.0, gripper_ramp_ticks=4)
    live = np.zeros(P, dtype=np.float32)
    obs = _obs(live)
    obs["agent_pos"][..., P] = 100.0              # measured width: fully open
    sent = [inf.infer(obs)["action"][0, P] for _ in range(8)]
    # head: close for rows 0..3 -> ramp 100 -> 75 -> 50 -> 25 -> 0; then open for rows 4..7 -> back up
    np.testing.assert_allclose(sent, [75, 50, 25, 0, 25, 50, 75, 100])
    # the rest of the returned chunk still carries the plain plateau targets
    a = inf.infer(obs)["action"]                  # tick 8: new chunk, head close on rows 0..3
    np.testing.assert_allclose(a[1:, P], [0, 0, 0, 100, 100, 100, 100])
    assert a[0, P] == 75.0                        # ramping down again from 100
    inf.reset_episode()
    assert inf._grip_cmd is None
    assert _make(pol, gripper_ramp_ticks=0).infer(obs)["action"][0, P] == 0.0   # no ramp: step


def test_non_finite_chunk_is_refused():
    class _NaNPolicy(_FakeODEPolicy):
        def predict_action(self, obs_dict, a0=None, **kw):
            out = super().predict_action(obs_dict, a0, **kw)
            out["action"][0, 2, 1] = float("nan")
            return out
    inf = _make(_NaNPolicy())
    with pytest.raises(RuntimeError, match="non-finite"):
        inf.infer(_obs(np.zeros(P, dtype=np.float32)))


def test_replan_every_bounds():
    with pytest.raises(ValueError):
        _make(_FakeODEPolicy(), replan_every=0)
    with pytest.raises(ValueError):
        _make(_FakeODEPolicy(), replan_every=N + 1)
    assert _make(_FakeODEPolicy(), replan_every=None).replan_every == N


# ── two arms: policy rows [L1..L6, R1..R6, L_grip, R_grip] -> reply rows in the wire order ─────────

POSE_COLS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


class _FakeBimanualPolicy(_FakeODEPolicy):
    """Row i = start + (i + 1) * 0.01 on all 12 joints (start = a0 or the live joints at POSE_COLS);
    LEFT gripper +1 on rows 0..3 and -1 after, RIGHT gripper -1 throughout."""

    pose_dim = 12
    n_grippers = 2
    state_pose_index = POSE_COLS

    def predict_action(self, obs_dict, a0=None, **kw):
        self.a0_calls.append(None if a0 is None else a0[0].cpu().numpy().copy())
        start = obs_dict["agent_pos"][:, -1, POSE_COLS] if a0 is None else a0
        traj = start[:, None, :] + torch.arange(1, N + 1, dtype=torch.float32)[None, :, None] * 0.01
        grip = torch.stack([torch.tensor([1.0] * 4 + [-1.0] * 4), -torch.ones(N)], dim=-1)[None]
        return {"action": torch.cat([traj, grip], dim=-1)}


def _obs14(left_width=100.0, right_width=30.0):
    ap = torch.zeros(1, 8, 14)
    ap[..., :6] = 0.5
    ap[..., 7:13] = -0.5
    ap[..., 6], ap[..., 13] = left_width, right_width
    return {k: torch.zeros(1, 8, 3, 8, 8) for k in ("head_cam", "wrist_cam", "left_wrist_cam")} | {"agent_pos": ap}


def test_two_arm_reply_is_in_the_wire_order_with_carry():
    pol = _FakeBimanualPolicy()
    inf = _make(pol, replan_every=8, gripper_close_cmd=0.0, gripper_open_cmd=100.0,
                camera_keys=("head_cam", "wrist_cam", "left_wrist_cam"))
    assert inf.state_dim == 14 and inf.pose_cols == POSE_COLS and inf.grip_cols == [6, 13]
    obs = _obs14()
    live = obs["agent_pos"][0, -1, POSE_COLS].numpy()
    chunk0 = live[None] + np.arange(1, N + 1)[:, None] * 0.01
    outs = [inf.infer(obs) for _ in range(8)]
    a = outs[0]["action"]
    assert a.shape == (N, 14) and a.dtype == np.float32
    np.testing.assert_allclose(a[:, POSE_COLS], chunk0, atol=1e-6)
    ramp = np.repeat(np.arange(1, N + 1)[:, None] * 0.01, 6, axis=1)
    np.testing.assert_allclose(a[:, :6], 0.5 + ramp, atol=1e-6)      # left joints
    np.testing.assert_allclose(a[:, 7:13], -0.5 + ramp, atol=1e-6)   # right joints
    np.testing.assert_allclose(a[:, 6], [0, 0, 0, 0, 100, 100, 100, 100])     # left gripper: close then open
    np.testing.assert_allclose(a[:, 13], [100] * 8)                           # right gripper: open
    np.testing.assert_allclose(outs[5]["action"][0, POSE_COLS], chunk0[5], atol=1e-6)
    inf.infer(obs)                                                            # tick 8: carried from row 7
    np.testing.assert_allclose(pol.a0_calls[1], chunk0[7], atol=1e-6)


def test_two_arm_per_gripper_values_and_ramps():
    pol = _FakeBimanualPolicy()
    inf = _make(pol, gripper_close_cmd=[1.0, 2.0], gripper_open_cmd=[90.0, 80.0], gripper_ramp_ticks=[4, 0])
    sent = [inf.infer(_obs14(left_width=90.0, right_width=30.0))["action"][0] for _ in range(8)]
    # left: ramps from the measured 90 toward close (1) by 89 / 4 per tick, then back toward open (90)
    np.testing.assert_allclose([s[6] for s in sent], [67.75, 45.5, 23.25, 1.0, 23.25, 45.5, 67.75, 90.0], atol=1e-4)
    # right: no ramp, straight to its own open value
    np.testing.assert_allclose([s[13] for s in sent], [80.0] * 8)
    with pytest.raises(ValueError, match="one per gripper"):
        _make(_FakeBimanualPolicy(), gripper_close_cmd=[0.0, 1.0, 2.0])
