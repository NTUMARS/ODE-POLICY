"""Native policy execution; model inference never blocks the 30 Hz control loop."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import queue
import signal
import sys
import threading
import time

import numpy as np

UPPER_BODY_INDICES = np.asarray([12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
CONTROL_HZ = 30


@dataclass(frozen=True)
class InitialPose:
    body29: np.ndarray
    action76: np.ndarray
    instruction: str

    @classmethod
    def load(cls, path, binding):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding.get("initial_pose_sha256"):
            raise ValueError("Initial pose bytes differ from the trained policy binding")
        value = json.loads(raw)
        if value.get("schema") != "ode-g1-initial-pose-v1" or value.get("task") != binding["task"]:
            raise ValueError("Initial pose belongs to a different task/revision")
        body = np.asarray(value["body29"], np.float32)
        action = np.asarray(value["action76"], np.float32)
        instruction = value["instruction"]
        if body.shape != (29,) or action.shape != (76,) or not np.isfinite(np.r_[body, action]).all():
            raise ValueError("Invalid initial body29/action76")
        if np.any((action[64:] < 0) | (action[64:] > 1)) or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Invalid initial hands/instruction")
        return cls(body, action, instruction)


def initial_pose_document(dataset):
    episode = dataset.load_episode(dataset.episode_ids[0])
    return {
        "schema": "ode-g1-initial-pose-v1",
        "task": {"name": dataset.spec.name, "revision": dataset.spec.revision},
        "body29": episode.state[0, :29].tolist(),
        "action76": episode.actions[0].tolist(),
        "instruction": dataset.instruction,
    }


class AsyncPredictor:
    """One daemon worker with bounded queues; late results retain their episode ID."""

    def __init__(self, policy, instruction):
        self.policy, self.instruction = policy, instruction
        self.requests, self.results = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.pending = False
        self.thread = threading.Thread(target=self._work, daemon=True, name="ode-g1-inference")
        self.thread.start()

    def submit(self, generation, observation, initial_body, requested_at):
        if self.pending:
            raise RuntimeError("An inference request is already in flight")
        self.requests.put_nowait((generation, observation, initial_body.copy(), requested_at))
        self.pending = True

    def _work(self):
        while True:
            request = self.requests.get()
            if request is None:
                return
            generation, observation, r0, requested_at = request
            try:
                actions = self.policy.predict_rgb(
                    observation.images, self.instruction, observation.state44, initial_body=r0
                )
                result = (generation, requested_at, actions, None)
            except Exception as exc:
                result = (generation, requested_at, None, exc)
            self.results.put(result)

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.pending = False
        return result

    def close(self):
        # An unresponsive CUDA call cannot prevent the main thread sending STOP.
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass


class Deployment:
    def __init__(self, policy, robot, initial_pose, settings, *, predictor=None):
        self.policy, self.robot, self.pose, self.settings = policy, robot, initial_pose, settings
        self.predictor = predictor or AsyncPredictor(policy, initial_pose.instruction)
        self.phase = "paused"
        self.generation = 0
        self.frame = 0
        self.chunk = None
        self.latency = 2 / CONTROL_HZ
        self.requested_at = None

    def stop(self):
        try:
            self.robot.command(stop=True)
        finally:
            self.generation += 1
            self.phase, self.chunk = "paused", None
            self.policy.reset_episode()

    def handle(self, command, now):
        if command in ("p", "q"):
            self.stop()
            return
        if command == "i" and self.phase == "paused":
            observation = self.robot.snapshot()
            if (
                min(
                    np.min(observation.state44[29:34]),
                    np.min(observation.state44[35:40]),
                    np.min(self.pose.action76[64:69]),
                    np.min(self.pose.action76[70:75]),
                )
                < 0.8
            ):
                raise RuntimeError("Initialization requires empty, already-open hands and an open-hand start pose")
            self.start_upper = observation.state44[:29][UPPER_BODY_INDICES].copy()
            self.started_at = now
            self.robot.planner(self.start_upper)
            self.robot.command(start=True, planner=True)
            self.phase = "initializing"
        elif command == "r" and self.phase == "ready":
            if self.predictor.pending:
                raise RuntimeError("Previous inference is still exiting; keep the controller stopped")
            self.robot.snapshot()
            sent = self.robot.send_action(self.pose.action76, self.frame)
            self.frame += 1
            self.robot.command(start=True, planner=False)
            self.policy.reset_episode()
            self.policy.commit_sent_prefix(sent[None])
            self.phase, self.chunk = "running", None
            self._request(self.robot.snapshot(), now)

    def _request(self, observation, now):
        r0 = self.policy.carry.normalized_initial_body(self.policy.normalizer)
        self.predictor.submit(self.generation, observation, r0, now)
        self.requested_at = now

    def tick(self, now):
        result = self.predictor.poll()
        if result is not None and result[0] == self.generation and self.phase == "running":
            _, requested_at, actions, error = result
            if error is not None:
                raise RuntimeError("Policy inference failed") from error
            if now - self.requested_at >= self.settings["max_inference_s"]:
                raise RuntimeError("Policy inference timed out")
            actions = np.asarray(actions, np.float32)
            if actions.shape != (self.policy.head.config.horizon, 76) or not np.isfinite(actions).all():
                raise RuntimeError("Invalid policy action chunk")
            if now - requested_at >= len(actions) / CONTROL_HZ:
                raise RuntimeError("Policy result arrived after its control horizon")
            self.latency = now - requested_at
            self.chunk = {"actions": actions, "at": requested_at, "last_index": -1}
        if self.phase == "paused":
            return
        observation = self.robot.snapshot()
        if self.phase in ("initializing", "ready"):
            target = self.pose.body29[UPPER_BODY_INDICES]
            elapsed = now - self.started_at
            fraction = min(1.0, elapsed / self.settings["initialization_s"])
            blend = fraction * fraction * (3 - 2 * fraction)
            self.robot.planner(self.start_upper + blend * (target - self.start_upper))
            self.robot.send_hands(self.pose.action76[64:])
            reached = (
                np.max(np.abs(observation.state44[:29][UPPER_BODY_INDICES] - target)) <= 0.12
                and np.max(np.abs(observation.state44[29:41] - self.pose.action76[64:])) <= 0.15
            )
            if self.phase == "initializing" and fraction == 1:
                if reached:
                    self.phase = "ready"
                elif elapsed > self.settings["initialization_s"] + 2:
                    raise RuntimeError("Measured initial pose did not reach the target")
            elif self.phase == "ready" and not reached:
                raise RuntimeError("Robot moved away from the staged start pose")
            return
        if self.predictor.pending and now - self.requested_at >= self.settings["max_inference_s"]:
            raise RuntimeError("Policy inference timed out")
        if self.chunk is None:
            # Maintain the measured staging command only while the first request is valid.
            sent = self.robot.send_action(self.pose.action76, self.frame)
            self.frame += 1
            self.policy.commit_sent_prefix(sent[None])
            return
        chunk = self.chunk
        index = int((now - chunk["at"]) * CONTROL_HZ)
        if index < 0 or index >= len(chunk["actions"]):
            raise RuntimeError("Action horizon exhausted before the next prediction")
        if index > chunk["last_index"]:
            sent = self.robot.send_action(chunk["actions"][index], self.frame)
            self.frame += 1
            self.policy.commit_sent_prefix(sent[None])
            chunk["last_index"] = index
        remaining = (len(chunk["actions"]) - index - 1) / CONTROL_HZ
        if not self.predictor.pending and remaining <= self.latency + 2 / CONTROL_HZ:
            self._request(observation, now)

    def close(self):
        try:
            self.stop()
        finally:
            self.predictor.close()


def run_deployment(policy, initial_pose, settings, camera_keys):
    from .robot import RobotIO

    commands = queue.SimpleQueue()

    def read_commands():
        for line in sys.stdin:
            commands.put(line.strip().lower())
        commands.put("q")

    if not sys.stdin.isatty():
        raise RuntimeError("Native deployment requires an interactive operator terminal")
    threading.Thread(target=read_commands, daemon=True, name="ode-g1-operator").start()
    previous = {sig: signal.signal(sig, lambda *_: commands.put("q")) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with RobotIO(settings, camera_keys) as robot:
            runner = Deployment(policy, robot, initial_pose, settings)
            print("Paused. Enter i: initialize, r: run, p: stop, q: quit. Each command requires Enter.", flush=True)
            phase = runner.phase
            try:
                while True:
                    started = time.monotonic()
                    quit_requested = False
                    while not commands.empty():
                        command = commands.get()
                        runner.handle(command, started)
                        if command == "q":
                            quit_requested = True
                            break
                    if quit_requested:
                        break
                    runner.tick(started)
                    if runner.phase != phase:
                        phase = runner.phase
                        print(phase, flush=True)
                    time.sleep(max(0, 1 / CONTROL_HZ - (time.monotonic() - started)))
            finally:
                runner.close()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return {"status": "operator_stopped"}
