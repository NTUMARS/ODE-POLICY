"""Native SONIC/Teleimager/DFX I/O; imports and connections are explicit on enter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import threading
import time

import numpy as np

HEADER_SIZE = 1280


def _array(value, shape, name, *, unit=False):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if unit and np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must use physical [0,1] opening units")
    return result.copy()


def _packet(topic, fields, *, version):
    descriptions = []
    payload = []
    for name, value in fields:
        value = np.ascontiguousarray(value)
        dtype = {
            np.dtype("float32"): "f32",
            np.dtype("int64"): "i64",
            np.dtype("int32"): "i32",
            np.dtype("uint8"): "u8",
        }[value.dtype]
        descriptions.append({"name": name, "dtype": dtype, "shape": list(value.shape)})
        payload.append(value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes())
    header = json.dumps(
        {"v": version, "endian": "le", "count": 1, "fields": descriptions}, separators=(",", ":")
    ).encode()
    if len(header) > HEADER_SIZE:
        raise ValueError("SONIC header exceeds 1280 bytes")
    return topic.encode() + header.ljust(HEADER_SIZE, b"\0") + b"".join(payload)


def token_message(token64, frame_index):
    token = _array(token64, (64,), "motion token")
    if type(frame_index) is not int or not 0 <= frame_index < 2**63:
        raise ValueError("Nonnegative int64 frame index required")
    return _packet(
        "pose", [("token_state", token[None]), ("frame_index", np.asarray([frame_index], np.int64))], version=4
    )


def command_message(*, start=False, stop=False, planner=False):
    if any(type(value) is not bool for value in (start, stop, planner)) or start and stop:
        raise ValueError("Command flags must be booleans; start and stop are exclusive")
    return _packet(
        "command",
        [
            (name, np.asarray([value], np.uint8))
            for name, value in [("start", start), ("stop", stop), ("planner", planner)]
        ],
        version=1,
    )


def planner_message(upper17):
    upper = _array(upper17, (17,), "IsaacLab-order upper body")
    return _packet(
        "planner",
        [
            ("mode", np.asarray([0], np.int32)),
            ("movement", np.zeros(3, np.float32)),
            ("facing", np.asarray([1, 0, 0], np.float32)),
            ("speed", np.asarray([-1], np.float32)),
            ("height", np.asarray([-1], np.float32)),
            ("upper_body_position", upper),
            ("upper_body_velocity", np.zeros(17, np.float32)),
        ],
        version=1,
    )


def projected_gravity(quaternion_wxyz):
    q = _array(quaternion_wxyz, (4,), "base quaternion").astype(np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("Base quaternion has zero norm")
    w, x, y, z = q / norm
    return np.asarray([2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1], np.float32)


@dataclass(frozen=True)
class Observation:
    state44: np.ndarray
    images: dict[str, np.ndarray]
    captured_at: float


class RobotIO:
    """Native transports with local submission receipts, not controller/actuator acknowledgements.

    One owner thread samples cameras/body; DDS callbacks timestamp actual hand state.
    """

    def __init__(self, settings, camera_keys):
        self.settings = dict(settings)
        self.camera_keys = tuple(camera_keys)
        self.camera_sources = tuple(settings.get("camera_sources", ["head", "chest"]))
        if len(self.camera_sources) != 2 or set(self.camera_sources) != {"head", "chest"}:
            raise ValueError("camera_sources must map the two keys to head/chest exactly once")
        if (
            len(self.camera_keys) != 2
            or len(set(self.camera_keys)) != 2
            or any(not isinstance(k, str) or not k for k in self.camera_keys)
        ):
            raise ValueError("Two ordered camera keys required")
        for key in ("command_endpoint", "state_endpoint", "camera_host"):
            if not isinstance(settings.get(key), str) or not settings[key]:
                raise ValueError(f"Explicit {key} required")
        for key in ("max_sensor_age_s", "max_camera_skew_s", "hand_speed"):
            value = settings.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if settings.get("head_source") not in ("full", "left", "right"):
            raise ValueError("head_source must be full, left or right")
        if type(settings.get("camera_port")) is not int or not 0 < settings["camera_port"] < 65536:
            raise ValueError("Valid camera_port required")
        if type(settings.get("dds_domain")) is not int or settings["dds_domain"] < 0:
            raise ValueError("Nonnegative DDS domain required")
        if type(settings.get("chest_rotate180")) is not bool:
            raise ValueError("chest_rotate180 must be boolean")
        self._clock = time.monotonic
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._context = self._pub = self._hand_pub = self._hand_sub = None
        self._body = self._hands = None
        self._images = {}
        self._source_frames = {}
        self._error = None
        self._last_hands = None
        self._last_hand_time = None
        self._closed = True
        self._entered = False

    def _record_error(self, error):
        with self._lock:
            if self._error is None:
                self._error = error

    def _open_transport(self):
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._pub = self._context.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 100)
        self._pub.setsockopt(zmq.SNDHWM, 32)
        self._pub.setsockopt(zmq.SNDTIMEO, 100)
        self._pub.bind(self.settings["command_endpoint"])

    def _open_hands(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

        interface = self.settings.get("network_interface", "")
        if interface:
            ChannelFactoryInitialize(self.settings["dds_domain"], networkInterface=interface)
        else:
            ChannelFactoryInitialize(self.settings["dds_domain"])
        self._cmd_type = MotorCmds_
        self._motor_type = unitree_go_msg_dds__MotorCmd_
        self._hand_pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self._hand_pub.Init()
        self._hand_sub = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self._hand_sub.Init(handler=self._hand_callback)

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Create a new RobotIO for a new connection lifetime")
        self._entered = True
        self._closed = False
        try:
            self._open_transport()
            self._open_hands()
            self._thread = threading.Thread(target=self._sensor_loop, name="ode-g1-sensors", daemon=True)
            self._thread.start()
            if not self._ready.wait(10):
                raise TimeoutError("Sensor resources did not initialize")
            with self._lock:
                error = self._error
            if error is not None:
                raise RuntimeError(f"Sensor initialization failed: {error}") from error
            return self
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _hand_callback(self, message):
        try:
            if len(message.states) != 12:
                raise ValueError("DFX feedback requires exactly 12 motor states")
            values = _array([state.q for state in message.states], (12,), "DFX feedback", unit=True)
            with self._lock:
                self._hands = (values[6:].copy(), values[:6].copy(), self._clock())
        except Exception as error:
            self._record_error(error)

    def _accept_body(self, message):
        if not isinstance(message, dict):
            raise ValueError("g1_debug payload must be a mapping")
        body = _array(message["body_q"], (29,), "MuJoCo body feedback")
        gravity = projected_gravity(message["base_quat"])
        with self._lock:
            self._body = (body, gravity, self._clock())

    def _accept_image(self, index, frame):
        if frame is None or self._source_frames.get(index) is frame:
            return
        raw = np.asarray(frame)
        if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 3 or min(raw.shape[:2]) < 1:
            raise ValueError("Teleimager frames must be current uint8 BGR HWC")
        selected = raw
        if index == 0 and self.settings["head_source"] != "full":
            width = raw.shape[1] // 2
            if width < 1:
                raise ValueError("Stereo head frame is too narrow")
            selected = raw[:, :width] if self.settings["head_source"] == "left" else raw[:, width:]
        if index == 1 and self.settings["chest_rotate180"]:
            selected = selected[::-1, ::-1]
        rgb = np.ascontiguousarray(selected[..., ::-1])
        self._source_frames[index] = frame
        with self._lock:
            self._images[index] = (rgb, self._clock())

    def _sensor_loop(self):
        subscriber = client = None
        try:
            import msgpack
            import msgpack_numpy

            try:
                from teleimager import ImageClient
            except ImportError:
                from teleimager.image_client import ImageClient
            subscriber = self._context.socket(self._zmq.SUB)
            subscriber.setsockopt(self._zmq.LINGER, 0)
            subscriber.setsockopt(self._zmq.CONFLATE, 1)
            subscriber.setsockopt_string(self._zmq.SUBSCRIBE, "g1_debug")
            subscriber.connect(self.settings["state_endpoint"])
            client = ImageClient(host=self.settings["camera_host"], request_port=self.settings["camera_port"])
            config = client.get_cam_config()
            for key in ("head_camera", "chest_camera"):
                if not config.get(key, {}).get("enable_zmq", False):
                    raise RuntimeError(f"Teleimager {key} ZMQ is disabled")
            self._ready.set()
            while not self._stop.is_set():
                if subscriber.poll(timeout=0):
                    raw = subscriber.recv(self._zmq.NOBLOCK)
                    if not raw.startswith(b"g1_debug"):
                        raise ValueError("Unexpected body-state topic")
                    self._accept_body(
                        msgpack.unpackb(raw[len(b"g1_debug") :], raw=False, object_hook=msgpack_numpy.decode)
                    )
                self._accept_image(0, client.get_head_frame()[0])
                self._accept_image(1, client.get_chest_frame()[0])
                self._stop.wait(0.002)
        except Exception as error:
            self._record_error(error)
        finally:
            self._ready.set()
            if client is not None:
                try:
                    client.close()
                except Exception as error:
                    self._record_error(error)
            if subscriber is not None:
                subscriber.close(linger=0)

    def _fresh(self):
        if self._closed:
            raise RuntimeError("Robot I/O is closed")
        with self._lock:
            error = self._error
            body = self._body
            hands = self._hands
            images = dict(self._images)
        if error is not None:
            raise RuntimeError(f"Robot sensor/transport failed: {error}") from error
        if body is None or hands is None or set(images) != {0, 1}:
            raise RuntimeError("Waiting for body, DFX and both cameras")
        now = self._clock()
        stamps = [body[2], hands[2], images[0][1], images[1][1]]
        if any(
            not math.isfinite(stamp) or not 0 <= now - stamp <= self.settings["max_sensor_age_s"] for stamp in stamps
        ):
            raise TimeoutError("Body, hand or camera feedback is stale")
        if abs(images[0][1] - images[1][1]) > self.settings["max_camera_skew_s"]:
            raise TimeoutError("Current camera pair exceeds maximum skew")
        state = np.r_[body[0], hands[0], hands[1], body[1]].astype(np.float32)
        sources = {"head": 0, "chest": 1}
        return (
            state,
            {key: images[sources[source]][0] for key, source in zip(self.camera_keys, self.camera_sources)},
            min(stamps),
        )

    def snapshot(self):
        state, images, stamp = self._fresh()
        return Observation(state.copy(), {key: value.copy() for key, value in images.items()}, stamp)

    def _publish(self, message):
        if self._pub is None:
            raise RuntimeError("Robot command publisher is not open")
        try:
            self._pub.send(message)
        except Exception as error:
            self._record_error(error)
            raise

    def command(self, *, start=False, stop=False, planner=False):
        message = command_message(start=start, stop=stop, planner=planner)
        if not stop:
            self._fresh()
        self._publish(message)

    def planner(self, upper17):
        self._fresh()
        self._publish(planner_message(upper17))

    def send_hands(self, hands12):
        target = _array(hands12, (12,), "DFX target", unit=True)
        state, _, _ = self._fresh()
        now = self._clock()
        elapsed = 0 if self._last_hand_time is None else now - self._last_hand_time
        if elapsed < 0:
            raise RuntimeError("Hand command clock moved backwards")
        previous = self._last_hands
        if previous is None or elapsed > self.settings["max_sensor_age_s"]:
            previous = state[29:41]
            elapsed = 0
        delta = self.settings["hand_speed"] * elapsed
        sent = np.clip(target, previous - delta, previous + delta).astype(np.float32)
        if self._hand_pub is None:
            raise RuntimeError("DFX publisher is not open")
        message = self._cmd_type()
        message.cmds = [self._motor_type() for _ in range(12)]
        for index, value in enumerate(np.r_[sent[6:], sent[:6]]):
            message.cmds[index].q = float(value)
        try:
            if self._hand_pub.Write(message) is False:
                raise RuntimeError("DFX command submission failed")
        except Exception as error:
            self._record_error(error)
            raise
        self._last_hands = sent.copy()
        self._last_hand_time = now
        return sent.copy()

    def send_action(self, action76, frame_index):
        action = _array(action76, (76,), "SONIC/DFX action")
        body_message = token_message(action[:64], frame_index)
        sent_hands = self.send_hands(action[64:])
        try:
            self._publish(body_message)
        except Exception as error:
            self._record_error(error)
            raise RuntimeError("Body submission failed after hand submission; stop and reinitialize") from error
        return np.r_[action[:64], sent_hands].astype(np.float32)

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            try:
                if self._pub is not None:
                    self.command(stop=True)
            finally:
                self._closed = True
                self._stop.set()
                if self._thread is not None:
                    self._thread.join(timeout=2)
                    if self._thread.is_alive():
                        self._record_error(RuntimeError("Sensor owner did not stop within two seconds"))
                for resource in (self._hand_sub, self._hand_pub):
                    close = getattr(resource, "Close", None) or getattr(resource, "close", None)
                    if close:
                        try:
                            close()
                        except Exception:
                            pass
                if self._pub is not None:
                    self._pub.close(linger=100)
                    self._pub = None
                if self._context is not None:
                    # A stuck camera constructor must not make close block forever.
                    if self._thread is None or not self._thread.is_alive():
                        self._context.term()
                self._hand_sub = self._hand_pub = None

    def __exit__(self, *_):
        self.close()
