"""Pin the msgpack+numpy wire format against the openpi-client encoding."""

import base64

import cv2
import numpy as np

from ode_policy.deploy.msgpack_numpy import packb, unpackb


def test_ndarray_roundtrip_float32():
    arr = np.arange(8 * 14, dtype=np.float32).reshape(8, 14)
    out = unpackb(packb(arr))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (8, 14)
    np.testing.assert_allclose(out, arr)


def test_ndarray_roundtrip_float64():
    arr = np.linspace(0, 1, 16, dtype=np.float64)
    out = unpackb(packb(arr))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, arr)


def test_request_payload_roundtrip():
    """Pin the exact payload shape the openpi-client sends."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    payload = {
        "state": [0.0] * 14,
        "images": {"cam_head": b64, "cam_left_wrist": b64, "cam_right_wrist": b64},
        "prompt": "pick the peach",
    }
    out = unpackb(packb(payload))
    assert out["state"] == [0.0] * 14
    assert "cam_head" in out["images"]
    assert out["prompt"] == "pick the peach"


def test_response_payload_roundtrip():
    response = {
        "actions": np.zeros((8, 14), dtype=np.float32),
        "state": np.zeros((14,), dtype=np.float32),
        "server_timing": {"infer_ms": 12.3, "prev_total_ms": 0.0},
        "policy_timing": {"infer_ms": 12.3},
    }
    out = unpackb(packb(response))
    assert out["actions"].shape == (8, 14)
    assert out["state"].shape == (14,)
    assert out["server_timing"]["infer_ms"] == 12.3
