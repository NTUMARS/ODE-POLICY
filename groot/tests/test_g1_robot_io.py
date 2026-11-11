import json
import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from ode_g1.robot import HEADER_SIZE, RobotIO, command_message, planner_message, projected_gravity, token_message


def settings(**changes):
    return dict(
        command_endpoint="inproc://commands",
        state_endpoint="inproc://states",
        camera_host="camera.invalid",
        camera_port=12345,
        network_interface="",
        dds_domain=0,
        head_source="full",
        chest_rotate180=True,
        max_sensor_age_s=0.5,
        max_camera_skew_s=0.1,
        hand_speed=2.0,
        **changes,
    )


class Publisher:
    def __init__(self):
        self.messages = []
        self.fail = False
        self.closed = False

    def send(self, message):
        if self.fail:
            raise OSError("fake send failure")
        self.messages.append(message)

    def close(self, **kwargs):
        self.closed = True


class HandPublisher:
    def __init__(self):
        self.messages = []
        self.ok = True
        self.closed = False

    def Write(self, message):
        self.messages.append(message)
        return self.ok

    def Close(self):
        self.closed = True


def ready_io(**changes):
    config = settings()
    config.update(changes)
    io = RobotIO(config, ["front", "chest"])
    stamp = [100.0]
    io._clock = lambda: stamp[0]
    io._closed = False
    io._pub = Publisher()
    io._hand_pub = HandPublisher()
    io._cmd_type = lambda: SimpleNamespace(cmds=[])
    io._motor_type = lambda: SimpleNamespace(q=0.0)
    io._accept_body({"body_q": np.arange(29) / 100, "base_quat": [1, 0, 0, 0]})
    values = np.r_[np.full(6, 0.2), np.full(6, 0.4)]
    io._hand_callback(SimpleNamespace(states=[SimpleNamespace(q=float(v)) for v in values]))
    head = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    chest = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    io._accept_image(0, head)
    io._accept_image(1, chest)
    return io, stamp, head, chest


def header(message, topic):
    return json.loads(message[len(topic) : len(topic) + HEADER_SIZE].rstrip(b"\0"))


def test_pose_protocol_v4_matches_golden_wire_bytes():
    token = np.arange(64, dtype=np.float32)
    expected_header = (
        b'{"v":4,"endian":"le","count":1,"fields":'
        b'[{"name":"token_state","dtype":"f32","shape":[1,64]},'
        b'{"name":"frame_index","dtype":"i64","shape":[1]}]}'
    )
    expected = b"pose" + expected_header.ljust(1280, b"\0") + struct.pack("<64fq", *token.tolist(), 19)
    assert token_message(token, 19) == expected
    with pytest.raises(ValueError):
        token_message(token, -1)


def test_command_and_planner_fields_match_sonic_layout():
    message = command_message(start=True, planner=True)
    assert message[-3:] == b"\x01\x00\x01"
    assert header(message, b"command") == {
        "v": 1,
        "endian": "le",
        "count": 1,
        "fields": [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ],
    }
    upper = np.arange(17, dtype=np.float32) / 10
    message = planner_message(upper)
    assert [v["name"] for v in header(message, b"planner")["fields"]] == [
        "mode",
        "movement",
        "facing",
        "speed",
        "height",
        "upper_body_position",
        "upper_body_velocity",
    ]
    expected = struct.pack("<i42f", 0, *([0, 0, 0, 1, 0, 0, -1, -1] + upper.tolist() + [0] * 17))
    assert message[7 + 1280 :] == expected
    with pytest.raises(ValueError):
        command_message(start=True, stop=True)


def test_projected_gravity_matches_inverse_wxyz_rotation():
    np.testing.assert_allclose(projected_gravity([1, 0, 0, 0]), [0, 0, -1])
    np.testing.assert_allclose(projected_gravity([math.sqrt(0.5), math.sqrt(0.5), 0, 0]), [0, -1, 0], atol=2e-7)
    np.testing.assert_allclose(projected_gravity([math.sqrt(0.5), 0, math.sqrt(0.5), 0]), [1, 0, 0], atol=2e-7)
    with pytest.raises(ValueError):
        projected_gravity([0, 0, 0, 0])


@pytest.mark.parametrize("source", ["full", "left", "right"])
def test_snapshot_camera_conversion_and_dfx_feedback_order(source):
    io, stamp, head, chest = ready_io(head_source=source)
    observation = io.snapshot()
    selection = head if source == "full" else head[:, :2] if source == "left" else head[:, 2:]
    np.testing.assert_array_equal(observation.images["front"], selection[..., ::-1])
    np.testing.assert_array_equal(observation.images["chest"], chest[::-1, ::-1, ::-1])
    np.testing.assert_allclose(observation.state44[29:35], 0.4)
    np.testing.assert_allclose(observation.state44[35:41], 0.2)
    assert observation.captured_at == 100.0
    observation.images["front"][:] = 0
    observation.state44[:] = 0
    assert io.snapshot().state44[29] == pytest.approx(0.4)
    np.testing.assert_array_equal(io.snapshot().images["front"], selection[..., ::-1])


def test_repeated_camera_array_does_not_refresh_and_skew_is_rejected():
    io, stamp, head, chest = ready_io()
    stamp[0] = 100.4
    io._accept_image(0, head)
    io._accept_image(1, chest)
    assert io.snapshot().captured_at == 100.0
    io._accept_image(0, head.copy())
    with pytest.raises(TimeoutError, match="skew"):
        io.snapshot()
    stamp[0] = 100.6
    with pytest.raises(TimeoutError, match="stale"):
        io.snapshot()


def test_invalid_hand_callback_latches_error_instead_of_returning_old_state():
    io, stamp, head, chest = ready_io()
    io._hand_callback(SimpleNamespace(states=[SimpleNamespace(q=2.0)] * 12))
    with pytest.raises(RuntimeError, match="physical"):
        io.snapshot()
    io._hand_callback(SimpleNamespace(states=[SimpleNamespace(q=0.5)] * 12))
    with pytest.raises(RuntimeError):
        io.snapshot()


def test_send_action_limits_hands_and_uses_right_first_dds_layout():
    io, stamp, head, chest = ready_io()
    target = np.r_[np.arange(64) / 100, np.ones(12)].astype(np.float32)
    first = io.send_action(target, 0)
    np.testing.assert_allclose(first[64:70], 0.4)
    np.testing.assert_allclose(first[70:], 0.2)
    stamp[0] += 0.05
    actual = io.send_action(target, 1)
    np.testing.assert_allclose(actual[64:70], 0.5, atol=1e-6)
    np.testing.assert_allclose(actual[70:], 0.3, atol=1e-6)
    values = [cmd.q for cmd in io._hand_pub.messages[-1].cmds]
    np.testing.assert_allclose(values, np.r_[actual[70:], actual[64:70]], atol=1e-6)
    assert io._pub.messages[-1] == token_message(actual[:64], 1)
    with pytest.raises(ValueError):
        io.send_action(np.r_[np.zeros(64), np.ones(12) * 2], 2)


def test_partial_send_failures_do_not_return_fictional_actual_action():
    io, stamp, head, chest = ready_io()
    io._hand_pub.ok = False
    with pytest.raises(RuntimeError, match="DFX"):
        io.send_action(np.zeros(76), 0)
    assert not io._pub.messages
    io, stamp, head, chest = ready_io()
    io._pub.fail = True
    with pytest.raises(RuntimeError, match="after hand"):
        io.send_action(np.zeros(76), 0)
    assert len(io._hand_pub.messages) == 1
    with pytest.raises(RuntimeError, match="failed"):
        io.snapshot()


def test_context_is_lazy_sensor_worker_is_owned_and_close_stops_first(monkeypatch):
    io = RobotIO(settings(), ["front", "chest"])
    events = []
    assert io._context is None and io._pub is None
    publisher = Publisher()

    def open_transport():
        io._pub = publisher

    def open_hands():
        io._hand_pub = HandPublisher()

    def worker():
        events.append("worker-start")
        io._ready.set()
        io._stop.wait()
        events.append("worker-close")

    monkeypatch.setattr(io, "_open_transport", open_transport)
    monkeypatch.setattr(io, "_open_hands", open_hands)
    monkeypatch.setattr(io, "_sensor_loop", worker)
    with io:
        assert io._thread.is_alive()
    assert publisher.messages == [command_message(stop=True)]
    assert publisher.closed and events == ["worker-start", "worker-close"]
    io.close()
    assert len(publisher.messages) == 1
    with pytest.raises(RuntimeError):
        io.__enter__()


def test_pub_uses_bounded_queue_for_control_mode_bursts(monkeypatch):
    import sys

    socket = Publisher()
    options = {}
    socket.setsockopt = lambda key, value: options.__setitem__(key, value)
    endpoints = []
    socket.bind = endpoints.append
    context = SimpleNamespace(socket=lambda kind: socket)
    fake = SimpleNamespace(Context=lambda: context, PUB=1, LINGER=2, SNDHWM=3, SNDTIMEO=4)
    monkeypatch.setitem(sys.modules, "zmq", fake)
    io = RobotIO(settings(), ["front", "chest"])
    io._open_transport()
    assert options[fake.SNDHWM] == 32
    assert options[fake.SNDTIMEO] == 100
    assert options[fake.LINGER] == 100
    assert endpoints == ["inproc://commands"]
    # Both protocol messages can be queued in the same tick; this is no remote ack.
    io._publish(planner_message(np.zeros(17)))
    io._publish(command_message(start=True, planner=True))
    assert [message[:7] for message in socket.messages] == [b"planner", b"command"]


def test_camera_key_order_can_map_chest_then_head_without_swapping_physical_inputs():
    io, stamp, head, chest = ready_io(camera_sources=["chest", "head"])
    observation = io.snapshot()
    np.testing.assert_array_equal(observation.images["front"], chest[::-1, ::-1, ::-1])
    np.testing.assert_array_equal(observation.images["chest"], head[..., ::-1])
    with pytest.raises(ValueError, match="camera_sources"):
        config = settings()
        config["camera_sources"] = ["head", "head"]
        RobotIO(config, ["a", "b"])


def test_sensor_owner_decodes_msgpack_numpy_and_closes_own_resources(monkeypatch):
    import sys

    io = RobotIO(settings(), ["front", "chest"])
    io._closed = False
    closed = []

    class Subscriber:
        def setsockopt(self, *args):
            pass

        def setsockopt_string(self, *args):
            pass

        def connect(self, endpoint):
            assert endpoint == "inproc://states"

        def poll(self, timeout):
            return True

        def recv(self, *args):
            return b"g1_debugPAYLOAD"

        def close(self, **kwargs):
            closed.append("subscriber")

    io._context = SimpleNamespace(socket=lambda kind: Subscriber())
    io._zmq = SimpleNamespace(SUB=1, LINGER=2, CONFLATE=3, SUBSCRIBE=4, NOBLOCK=5)
    decode = lambda value: value

    def unpack(payload, **kwargs):
        assert payload == b"PAYLOAD" and kwargs == {"raw": False, "object_hook": decode}
        return {"body_q": [0] * 29, "base_quat": [1, 0, 0, 0]}

    class Client:
        def __init__(self, host, request_port):
            assert (host, request_port) == ("camera.invalid", 12345)

        def get_cam_config(self):
            return {"head_camera": {"enable_zmq": True}, "chest_camera": {"enable_zmq": True}}

        def get_head_frame(self):
            return np.zeros((2, 4, 3), np.uint8), 30

        def get_chest_frame(self):
            io._stop.set()
            return np.ones((2, 2, 3), np.uint8), 30

        def close(self):
            closed.append("camera")

    monkeypatch.setitem(sys.modules, "msgpack", SimpleNamespace(unpackb=unpack))
    monkeypatch.setitem(sys.modules, "msgpack_numpy", SimpleNamespace(decode=decode))
    monkeypatch.setitem(sys.modules, "teleimager", SimpleNamespace(ImageClient=Client))
    io._sensor_loop()
    assert io._error is None and io._body is not None and set(io._images) == {0, 1}
    assert closed == ["camera", "subscriber"]
