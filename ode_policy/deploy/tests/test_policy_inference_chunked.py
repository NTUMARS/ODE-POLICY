"""PolicyInference.replan_every: chunk policies (fm_dit, ...) served with the same chunked
schedule as the ODE server, without a checkpoint (fake policy injected)."""

import numpy as np
import torch

from ode_policy.deploy.policy_inference import PolicyInference


class _FakeChunkPolicy:
    """Chunk i: every row r = 10 * i + r (all 7 dims), so shifts are easy to read."""

    def __init__(self):
        self.calls = 0

    def predict_action(self, obs_dict):
        base = 10.0 * self.calls
        self.calls += 1
        rows = base + torch.arange(8, dtype=torch.float32)[:, None].expand(8, 7)
        return {"action": rows[None].clone()}

    def reset(self):
        pass


def _make(replan_every):
    inf = object.__new__(PolicyInference)    # skip checkpoint loading
    inf.policy, inf.device, inf.policy_label = _FakeChunkPolicy(), torch.device("cpu"), "Fake"
    inf.n_obs_steps, inf.n_action_steps, inf.horizon = 8, 8, 16
    inf.replan_every = replan_every
    inf.reset_episode()
    return inf


def test_replan_every_1_predicts_every_request():
    inf = _make(1)
    outs = [inf.infer({}) for _ in range(3)]
    assert inf.policy.calls == 3 and all(o["replanned"] for o in outs)
    assert [o["action"][0, 0] for o in outs] == [0.0, 10.0, 20.0]


def test_replan_every_8_walks_through_the_chunk():
    inf = _make(8)
    outs = [inf.infer({}) for _ in range(10)]
    assert inf.policy.calls == 2
    assert [o["replanned"] for o in outs] == [True] + [False] * 7 + [True, False]
    assert [o["action"][0, 0] for o in outs] == [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
    a3 = outs[3]["action"]
    assert a3.shape == (8, 7) and a3.dtype == np.float32
    np.testing.assert_allclose(a3[:, 0], [3, 4, 5, 6, 7, 7, 7, 7])   # tail held
    inf.reset_episode()
    assert inf.infer({})["replanned"] and inf.policy.calls == 3
