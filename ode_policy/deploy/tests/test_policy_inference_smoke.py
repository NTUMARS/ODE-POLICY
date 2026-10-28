"""End-to-end smoke against a real ODE checkpoint (skipped without CUDA / checkpoint).

    ODE_DEPLOY_TEST_CKPT=il_outputs/ode/r1lite_insert_flower/checkpoints/100.ckpt \\
        python -m pytest ode_policy/deploy/tests/test_policy_inference_smoke.py -v

Single-arm (7-D, head + right wrist) and two-arm (14-D, + left wrist) checkpoints alike: the
observation is shaped from what the checkpoint was trained on.
"""

import os

import numpy as np
import pytest
import torch

CKPT = os.environ.get("ODE_DEPLOY_TEST_CKPT", "")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (CKPT and os.path.isfile(CKPT)), reason=f"checkpoint not found: {CKPT!r}"),
]


def test_ode_chunked_inference_on_checkpoint():
    from ode_policy.deploy.adapters import dummy_obs_dict
    from ode_policy.deploy.ode_inference import ODEChunkedInference

    inf = ODEChunkedInference.from_checkpoint(CKPT, device="cuda:0", gripper_close_cmd=0.0, gripper_open_cmd=100.0)
    assert inf.n_obs_steps == 8 and inf.n_action_steps == 8
    assert getattr(inf.policy, "joint_space", False), "checkpoint must be trained with joint_space=true"
    if inf.state_dim == 7:
        assert inf.pose_dim == 6 and inf.grip_cols == [6] and inf.camera_keys == ("head_cam", "wrist_cam")
    else:
        assert inf.state_dim == 14 and inf.pose_dim == 12 and inf.grip_cols == [6, 13]
        assert inf.camera_keys == ("head_cam", "wrist_cam", "left_wrist_cam")
    state = np.zeros(inf.state_dim, dtype=np.float32)
    state[inf.grip_cols] = 90.0
    flags = []
    for _ in range(10):
        out = inf.infer(dummy_obs_dict(inf, state=state))
        a = out["action"]
        assert a.shape == (8, inf.state_dim) and a.dtype == np.float32 and np.all(np.isfinite(a))
        assert set(np.unique(a[:, inf.grip_cols]).tolist()) <= {0.0, 100.0}
        flags.append(out["replanned"])
    assert flags == [True] + [False] * 7 + [True, False]
    assert inf.n_replans == 2
