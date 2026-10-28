"""Unit tests for ObservationAdapter.

Tests are split into three groups:
  1. image decode (this file segment) — independent of full adapt()
  2. state extraction
  3. history buffering / adapt() integration
"""

import base64

import cv2
import numpy as np
import pytest
import torch

from ode_policy.deploy.adapters import ObservationAdapter


def _make_b64_png(img_bgr: np.ndarray) -> str:
    """Encode an OpenCV BGR uint8 image to base64 PNG (matches client format)."""
    ok, buf = cv2.imencode(".png", img_bgr)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── Image-decode tests ──────────────────────────────────────────────────────


def test_image_decode_resizes_to_target():
    """A 224×224 client image must be resized to 256×256 on the server."""
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)  # BGR ordering implied
    b64 = _make_b64_png(raw)
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    chw = adapter._decode(b64)
    assert chw.shape == (3, 256, 256)
    assert chw.dtype == np.float32
    assert 0.0 <= chw.min() and chw.max() <= 1.0


def test_image_decode_passes_through_when_already_target_size():
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    b64 = _make_b64_png(raw)
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    chw = adapter._decode(b64)
    assert chw.shape == (3, 256, 256)
    assert chw.dtype == np.float32
    assert 0.0 <= chw.min() and chw.max() <= 1.0


def test_image_decode_bgr_to_rgb():
    """A pure-red BGR image (B=0, G=0, R=255) must come out as RGB channel 0 high."""
    bgr = np.zeros((256, 256, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255  # red in BGR is the 3rd channel
    b64 = _make_b64_png(bgr)
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    chw = adapter._decode(b64)
    # After BGR→RGB and HWC→CHW: channel 0 is R, should be 1.0; G and B 0.0 (PNG is lossless).
    np.testing.assert_allclose(chw[0], 1.0)
    np.testing.assert_allclose(chw[1], 0.0)
    np.testing.assert_allclose(chw[2], 0.0)


def test_image_decode_bad_png_raises():
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        adapter._decode(base64.b64encode(b"not a png").decode("ascii"))


def test_image_decode_bad_base64_raises():
    """Invalid base64 (wrong padding) must surface as ValueError, not binascii.Error."""
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="not valid base64"):
        adapter._decode("abc")  # length 3: incorrect padding -> binascii.Error


def test_image_decode_empty_bytes_raises():
    """Zero-byte payload must surface as ValueError, not cv2.error."""
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        adapter._decode(base64.b64encode(b"").decode("ascii"))


# ── adapt() integration tests ──────────────────────────────────────────────


def _make_msg(state_14, prompt=None):
    """Synthesize a client request with placeholder images."""
    rng = np.random.default_rng(2)
    head = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    return {
        "state": list(state_14),
        "images": {
            "cam_head": _make_b64_png(head),
            "cam_left_wrist": _make_b64_png(np.zeros_like(head)),  # ignored
            "cam_right_wrist": _make_b64_png(wrist),
        },
        **({"prompt": prompt} if prompt is not None else {}),
    }


def test_adapt_state_splits_into_right_arm_and_left_echo():
    state = np.arange(14, dtype=np.float32)
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    obs_dict, left_echo = adapter.adapt(_make_msg(state))
    # Left echo is the first 7 entries
    np.testing.assert_allclose(left_echo, np.arange(7, dtype=np.float32))
    # agent_pos = state[7:13] + state[13] = arange(7,14)
    last_agent_pos = obs_dict["agent_pos"][0, -1].cpu().numpy()
    np.testing.assert_allclose(last_agent_pos, np.arange(7, 14, dtype=np.float32))


def test_adapt_rejects_wrong_state_length():
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        adapter.adapt(_make_msg(np.zeros(13, dtype=np.float32)))


def test_adapt_rejects_missing_images():
    msg = _make_msg(np.zeros(14, dtype=np.float32))
    del msg["images"]["cam_right_wrist"]
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="cam_right_wrist"):
        adapter.adapt(msg)


def test_adapt_rejects_missing_head_image():
    msg = _make_msg(np.zeros(14, dtype=np.float32))
    del msg["images"]["cam_head"]
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="cam_head"):
        adapter.adapt(msg)


def test_adapt_first_call_pads_history_forward():
    """First adapt() after reset() must fill the deque to length n_obs_steps
    by repeating the same triple."""
    state = np.arange(14, dtype=np.float32)
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    obs_dict, _ = adapter.adapt(_make_msg(state))
    assert obs_dict["head_cam"].shape == (1, 8, 3, 256, 256)
    assert obs_dict["wrist_cam"].shape == (1, 8, 3, 256, 256)
    assert obs_dict["agent_pos"].shape == (1, 8, 7)
    # All 8 timesteps should be identical
    for t in range(1, 8):
        torch.testing.assert_close(obs_dict["head_cam"][0, 0], obs_dict["head_cam"][0, t])
        torch.testing.assert_close(obs_dict["agent_pos"][0, 0], obs_dict["agent_pos"][0, t])


def test_adapt_subsequent_calls_shift_by_one():
    """After warm-up, each new call shifts the buffer by exactly 1."""
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    states = [np.full(14, fill_value=k, dtype=np.float32) for k in range(10)]
    obs_history = []
    for s in states:
        obs_dict, _ = adapter.adapt(_make_msg(s))
        obs_history.append(obs_dict["agent_pos"][0].cpu().numpy().copy())  # (8, 7)
    # The newest agent_pos at step k should equal the state value k
    for k, ap in enumerate(obs_history):
        last = ap[-1]
        # agent_pos is the latter 7 entries of state; here all entries == k
        np.testing.assert_allclose(last, np.full(7, float(k)))
    # After at least 8 steps, the buffer contains 8 strictly different values
    last = obs_history[-1]  # step 9
    expected_window = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    np.testing.assert_allclose(last[:, 0], expected_window)


def test_adapt_reset_clears_warm_up():
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    adapter.adapt(_make_msg(np.zeros(14, dtype=np.float32)))
    adapter.reset()
    assert not adapter.warmed
    assert len(adapter.buf) == 0


def test_adapt_logs_prompt_once_then_silent(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="ode_policy.deploy.adapters")
    adapter = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"))
    s = np.zeros(14, dtype=np.float32)
    adapter.adapt(_make_msg(s, prompt="pick the peach"))
    adapter.adapt(_make_msg(s, prompt="pick the peach"))
    adapter.adapt(_make_msg(s, prompt="pick the peach"))
    prompt_logs = [r for r in caplog.records if "prompt" in r.getMessage().lower()]
    assert len(prompt_logs) == 1


# ── two-arm policies: 14-D agent_pos (the wire state as is) and the left wrist camera ──────────────


def _make_msg3(state_14):
    rng = np.random.default_rng(3)
    imgs = {k: rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
            for k in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
    imgs["cam_left_wrist"][:] = 0
    imgs["cam_left_wrist"][:, :, 2] = 255           # pure red (BGR) on the left wrist only
    return {"state": list(state_14), "images": {k: _make_b64_png(v) for k, v in imgs.items()}}


def _bi_adapter():
    return ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"), state_dim=14,
                              camera_keys=("head_cam", "wrist_cam", "left_wrist_cam"))


def test_two_arm_adapter_reads_the_whole_state_and_three_cameras():
    state = np.arange(14, dtype=np.float32)
    obs, left_echo = _bi_adapter().adapt(_make_msg3(state))
    assert set(obs) == {"head_cam", "wrist_cam", "left_wrist_cam", "agent_pos"}
    assert obs["agent_pos"].shape == (1, 8, 14)
    np.testing.assert_allclose(obs["agent_pos"][0, -1].numpy(), state)        # wire order = training order
    assert obs["left_wrist_cam"].shape == (1, 8, 3, 256, 256)
    np.testing.assert_allclose(obs["left_wrist_cam"][0, -1, 0].numpy(), 1.0)  # cam_left_wrist -> left_wrist_cam
    np.testing.assert_allclose(obs["left_wrist_cam"][0, -1, 1:].numpy(), 0.0)
    np.testing.assert_allclose(left_echo, state[:7])


def test_two_arm_adapter_requires_the_left_wrist_image():
    msg = _make_msg3(np.zeros(14, dtype=np.float32))
    del msg["images"]["cam_left_wrist"]
    with pytest.raises(ValueError, match="cam_left_wrist"):
        _bi_adapter().adapt(msg)
    # a single-arm policy never needs it
    obs, _ = ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu")).adapt(msg)
    assert set(obs) == {"head_cam", "wrist_cam", "agent_pos"} and obs["agent_pos"].shape == (1, 8, 7)


def test_adapter_rejects_unknown_layouts():
    with pytest.raises(ValueError):
        ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"), state_dim=12)
    with pytest.raises(ValueError, match="front_cam"):
        ObservationAdapter(n_obs_steps=8, image_size=256, device=torch.device("cpu"),
                           camera_keys=("head_cam", "front_cam"))
