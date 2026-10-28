"""Integration: start the server with a fake inference object, talk msgpack over WebSocket
and validate the exact reply the robot client expects (the ODE server
shares the handler, only `infer` differs)."""

import asyncio
import base64
import json
import socket
from contextlib import closing

import cv2
import numpy as np
import pytest
import torch
import websockets

from ode_policy.deploy import serve_policy as serve_mod
from ode_policy.deploy.msgpack_numpy import packb, unpackb


class _FakeInference:
    def __init__(self):
        self.n_obs_steps, self.n_action_steps, self.horizon = 8, 8, 16
        self.device = torch.device("cpu")
        self.policy_label = "FakeODE"
        self._reset_calls = 0
        self._tick = 0

    def reset_episode(self):
        self._reset_calls += 1
        self._tick = 0

    def infer(self, obs_dict):
        assert obs_dict["head_cam"].shape == (1, 8, 3, 256, 256)
        assert obs_dict["wrist_cam"].shape == (1, 8, 3, 256, 256)
        assert obs_dict["agent_pos"].shape == (1, 8, 7)
        action = np.full((8, 7), 0.123, dtype=np.float32)
        action[:, 6] = 100.0
        k = self._tick % 8
        self._tick += 1
        return {"action": action, "router_weight": None, "steps_used": None, "replanned": k == 0, "chunk_index": k}


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _b64_dummy_image(size=224):
    ok, buf = cv2.imencode(".png", np.zeros((size, size, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _obs_msg():
    return {"state": [float(i) for i in range(14)],
            "images": {"cam_head": _b64_dummy_image(), "cam_left_wrist": _b64_dummy_image(),
                       "cam_right_wrist": _b64_dummy_image()},
            "prompt": "insert flower"}


@pytest.mark.timeout(20)
def test_metadata_and_one_inference_roundtrip():
    port, fake = _free_port(), _FakeInference()

    async def run():
        server = await serve_mod.start_server(fake, "127.0.0.1", port, image_size=256)
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                meta = unpackb(await ws.recv())
                assert meta == {"action_dim": 14, "n_action_steps": 8, "horizon": 16, "policy": "FakeODE"}
                await ws.send(packb(_obs_msg()))
                resp = unpackb(await ws.recv())
                a = resp["actions"]
                assert a.shape == (8, 14) and a.dtype == np.float32
                np.testing.assert_allclose(a[:, 0:7], np.tile(np.arange(7, dtype=np.float32), (8, 1)))   # left arm echoed
                np.testing.assert_allclose(a[:, 7:13], 0.123)
                np.testing.assert_allclose(a[:, 13], 100.0)
                assert resp["state"].shape == (14,)
                assert "infer_ms" in resp["server_timing"] and "policy_timing" in resp
        finally:
            server.close()
            await server.wait_closed()
        assert fake._reset_calls == 1

    asyncio.run(run())


@pytest.mark.timeout(20)
def test_missing_wrist_image_closes_1003_and_server_survives():
    port, fake = _free_port(), _FakeInference()

    async def run():
        server = await serve_mod.start_server(fake, "127.0.0.1", port, image_size=256)
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                await ws.recv()
                bad = _obs_msg()
                del bad["images"]["cam_right_wrist"]
                await ws.send(packb(bad))
                with pytest.raises(websockets.exceptions.ConnectionClosed) as ei:
                    await ws.recv()
                code = getattr(ei.value, "code", None) or ei.value.rcvd.code
                assert code == 1003
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                assert unpackb(await ws.recv())["policy"] == "FakeODE"
        finally:
            server.close()
            await server.wait_closed()
        assert fake._reset_calls == 2

    asyncio.run(run())


@pytest.mark.timeout(30)
def test_episode_diag_logger_writes_jsonl_summary_png(tmp_path):
    port, fake = _free_port(), _FakeInference()
    log_root = tmp_path / "deploy_logs"
    n_ticks = 10

    async def run():
        server = await serve_mod.start_server(fake, "127.0.0.1", port, image_size=256, log_root=log_root,
                                              log_stem="my_run")
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                await ws.recv()
                for _ in range(n_ticks):
                    await ws.send(packb(_obs_msg()))
                    await ws.recv()
            for _ in range(100):
                if (log_root / "my_run_summary.json").exists():
                    break
                await asyncio.sleep(0.05)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())
    lines = (log_root / "my_run.jsonl").read_text().splitlines()
    assert len(lines) == n_ticks + 1 and json.loads(lines[0])["_meta"] is True
    rows = [json.loads(l) for l in lines[1:]]
    assert [r["tick"] for r in rows] == list(range(n_ticks))
    assert [r["replanned"] for r in rows] == [True] + [False] * 7 + [True, False]
    assert rows[0]["action0"][6] == 100.0 and len(rows[0]["state"]) == 7
    summary = json.loads((log_root / "my_run_summary.json").read_text())
    assert summary["n_ticks"] == n_ticks and summary["n_replans"] == 2
    assert (log_root / "my_run.png").stat().st_size > 1000


def test_resolve_log_root_and_stems(tmp_path):
    assert serve_mod._resolve_log_root("x.ckpt", None, False) is None
    assert serve_mod._resolve_log_root("x.ckpt", None, True) == serve_mod.REPO_ROOT / "deploy_logs"
    assert serve_mod._resolve_log_root("x.ckpt", str(tmp_path / "d"), False) == tmp_path / "d"
    ckpt_dir = tmp_path / "il_outputs" / "ode" / "r1lite_insert_flower" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "100.ckpt").touch()
    assert serve_mod._default_log_stem(str(ckpt_dir / "100.ckpt")).startswith("ode_r1lite_insert_flower_100_")
    assert serve_mod._unique_stem(tmp_path, "run") == "run"
    (tmp_path / "run.jsonl").write_text("first")
    assert serve_mod._unique_stem(tmp_path, "run") == "run_001"


class _FakeBimanualInference(_FakeInference):
    """A two-arm checkpoint: 14-D agent_pos, three cameras, 14-D action rows in the wire order."""

    state_dim = 14
    camera_keys = ("head_cam", "wrist_cam", "left_wrist_cam")

    def infer(self, obs_dict):
        assert set(obs_dict) == {"head_cam", "wrist_cam", "left_wrist_cam", "agent_pos"}
        assert obs_dict["left_wrist_cam"].shape == (1, 8, 3, 256, 256)
        assert obs_dict["agent_pos"].shape == (1, 8, 14)
        np.testing.assert_allclose(obs_dict["agent_pos"][0, -1].numpy(), np.arange(14, dtype=np.float32))
        action = np.full((8, 14), 0.25, dtype=np.float32)
        action[:, 0:6] = -0.75                     # the policy moves the LEFT arm too
        action[:, 6], action[:, 13] = 0.0, 100.0
        k = self._tick % 8
        self._tick += 1
        return {"action": action, "router_weight": None, "steps_used": None, "replanned": k == 0, "chunk_index": k}


@pytest.mark.timeout(30)
def test_two_arm_roundtrip_no_echo_and_left_wrist_required(tmp_path):
    port, fake = _free_port(), _FakeBimanualInference()
    log_root = tmp_path / "logs"

    async def run():
        server = await serve_mod.start_server(fake, "127.0.0.1", port, image_size=256, log_root=log_root,
                                              log_stem="bi")
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                assert unpackb(await ws.recv())["action_dim"] == 14
                for _ in range(3):
                    await ws.send(packb(_obs_msg()))
                    a = unpackb(await ws.recv())["actions"]
                    assert a.shape == (8, 14) and a.dtype == np.float32
                    np.testing.assert_allclose(a[:, 0:6], -0.75)     # NOT the client's left arm (0..5)
                    np.testing.assert_allclose(a[:, 6], 0.0)
                    np.testing.assert_allclose(a[:, 7:13], 0.25)
                    np.testing.assert_allclose(a[:, 13], 100.0)
            async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
                await ws.recv()
                bad = _obs_msg()
                del bad["images"]["cam_left_wrist"]
                await ws.send(packb(bad))
                with pytest.raises(websockets.exceptions.ConnectionClosed) as ei:
                    await ws.recv()
                assert (getattr(ei.value, "code", None) or ei.value.rcvd.code) == 1003
            for _ in range(100):
                if (log_root / "bi_summary.json").exists():
                    break
                await asyncio.sleep(0.05)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())
    lines = (log_root / "bi.jsonl").read_text().splitlines()
    meta = json.loads(lines[0])
    assert meta["action_dims"][6] == "L_grip" and meta["action_dims"][13] == "R_grip"
    row = json.loads(lines[1])
    assert len(row["state"]) == 14 and len(row["action0"]) == 14 and row["action0"][0] == -0.75
    assert (log_root / "bi.png").stat().st_size > 1000
