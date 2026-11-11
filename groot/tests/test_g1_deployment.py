import json
import hashlib
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ode_g1.carry import ExecutedBodyCarry
from ode_g1.config import DEFAULTS, validate_deployment_config
from ode_g1.deployment import Deployment, InitialPose
from ode_g1.runtime import ODEPolicy, RGBEncoder


class Normalizer:
    def normalize_actions_unclipped(self, value):
        return value

    def normalize_state(self, value):
        return value

    def denormalize_actions(self, value):
        return value


class Predictor:
    def __init__(self):
        self.pending = False
        self.result = None
        self.requests = []

    def submit(self, generation, observation, r0, requested_at):
        self.requests.append((generation, observation, r0.copy(), requested_at))
        self.pending = True

    def poll(self):
        result, self.result = self.result, None
        if result is not None:
            self.pending = False
        return result

    def close(self):
        pass


class Robot:
    def __init__(self):
        self.sent, self.commands = [], []
        self.state = np.r_[np.zeros(29), np.ones(12), [0, 0, -1]].astype(np.float32)
        self.now = 0

    def snapshot(self):
        return SimpleNamespace(state44=self.state.copy(), images={}, captured_at=self.now)

    def planner(self, upper):
        self.upper = upper.copy()

    def send_hands(self, hands):
        return hands.copy()

    def command(self, **values):
        self.commands.append(values)

    def send_action(self, action, frame):
        actual = action.copy()
        actual[64:] = 0.9  # Simulate the transport's actual bounded hand command.
        self.sent.append((actual, frame))
        return actual


def runner():
    policy = SimpleNamespace(
        head=SimpleNamespace(config=SimpleNamespace(horizon=16)), carry=ExecutedBodyCarry(), normalizer=Normalizer()
    )
    policy.reset_episode = policy.carry.reset
    policy.commit_sent_prefix = policy.carry.commit_executed
    robot, predictor = Robot(), Predictor()
    pose = InitialPose(np.zeros(29), np.r_[np.zeros(64), np.ones(12)].astype(np.float32), "move object")
    deployment = Deployment(policy, robot, pose, DEFAULTS["deployment"], predictor=predictor)
    return deployment, robot, predictor


def staged_runner():
    deployment, robot, predictor = runner()
    deployment.handle("i", 0)
    robot.now = 3
    deployment.tick(3)
    assert deployment.phase == "ready"
    deployment.handle("r", 3)
    return deployment, robot, predictor


def test_staging_requires_measured_open_hands_and_explicit_run():
    deployment, robot, _ = runner()
    robot.state[30] = 0.2
    with pytest.raises(RuntimeError, match="already-open"):
        deployment.handle("i", 0)
    assert not robot.sent and not robot.commands
    robot.state[30] = 1
    deployment.handle("i", 0)
    deployment.tick(1)
    assert deployment.phase == "initializing" and not robot.sent
    deployment.tick(3)
    assert deployment.phase == "ready" and not robot.sent
    deployment.handle("r", 3)
    assert deployment.phase == "running" and len(robot.sent) == 1
    assert deployment.policy.carry.executed_steps == 1


def test_async_elapsed_index_and_only_actual_sends_advance_carry():
    deployment, robot, predictor = staged_runner()
    actions = np.tile(np.r_[np.zeros(64), np.ones(12)], (16, 1)).astype(np.float32)
    actions[:, :64] = np.arange(16)[:, None]
    predictor.result = (deployment.generation, 3, actions, None)
    robot.now = 3.1
    deployment.tick(3.1)
    assert robot.sent[-1][0][0] == 3
    assert deployment.policy.carry.executed_steps == 2
    np.testing.assert_array_equal(deployment.policy.carry._last_issued, actions[3, :64])
    deployment.tick(3.1)
    assert len(robot.sent) == 2  # No duplicate index or unexecuted tail.
    deployment.close()
    assert robot.commands[-1] == {"stop": True}


def test_late_old_episode_result_is_discarded_after_stop():
    deployment, robot, predictor = staged_runner()
    old = deployment.generation
    deployment.handle("p", 3.05)
    predictor.result = (old, 3, None, RuntimeError("old request"))
    deployment.tick(3.1)
    assert deployment.phase == "paused" and len(robot.sent) == 1
    assert deployment.policy.carry.executed_steps == 0


def test_sensor_age_does_not_shift_the_r0_trajectory_origin():
    deployment, robot, predictor = runner()
    deployment.handle("i", 0)
    robot.now = 2.9  # Observation is older than the actual r0 snapshot.
    deployment.tick(3)
    deployment.handle("r", 3)
    assert predictor.requests[0][1].captured_at == 2.9
    assert predictor.requests[0][3] == 3
    actions = np.ones((16, 76), np.float32)
    actions[:, :64] = np.arange(16)[:, None]
    predictor.result = (deployment.generation, predictor.requests[0][3], actions, None)
    deployment.tick(3.01)
    assert robot.sent[-1][0][0] == 0


def test_result_arriving_after_timeout_is_rejected_even_inside_horizon():
    deployment, robot, predictor = staged_runner()
    deployment.settings = {**deployment.settings, "max_inference_s": 0.1}
    predictor.result = (deployment.generation, 3, np.ones((16, 76), np.float32), None)
    with pytest.raises(RuntimeError, match="timed out"):
        try:
            deployment.tick(3.2)
        finally:
            deployment.close()
    assert len(robot.sent) == 1


@pytest.mark.parametrize("mode", ["timeout", "late", "error", "exhausted"])
def test_invalid_inference_stops_without_executing_a_tail(mode):
    deployment, robot, predictor = staged_runner()
    actions = np.ones((16, 76), np.float32)
    if mode == "late":
        predictor.result = (deployment.generation, 3, actions, None)
    elif mode == "error":
        predictor.result = (deployment.generation, 3, None, RuntimeError("inference"))
    elif mode == "exhausted":
        deployment.chunk = {"actions": actions, "at": 3, "last_index": 15}
        predictor.pending = False
    with pytest.raises(RuntimeError):
        try:
            deployment.tick(3.6)
        finally:
            deployment.close()
    assert robot.commands[-1] == {"stop": True}
    assert len(robot.sent) == 1


def test_rgb_request_freezes_r0_before_encoding_even_if_execution_continues():
    class Head:
        config = SimpleNamespace(horizon=16)

        def eval(self):
            pass

        def predict_cached(self, batch, state, initial):
            self.initial = initial.clone()
            return {"action_pred": torch.cat([initial, torch.zeros(1, 12)], dim=-1)[:, None]}

    head = Head()
    policy = ODEPolicy(head, Normalizer())
    policy.initialize_last_sent_body(np.full(64, 2))

    def encoder(images, instruction):
        policy.commit_sent_prefix(np.full((1, 76), 9))
        return {}

    policy.rgb_encoder = encoder
    policy.predict_rgb({}, "move", np.r_[np.zeros(29), np.ones(12), [0, 0, -1]])
    np.testing.assert_array_equal(head.initial.numpy(), np.full((1, 64), 2))
    np.testing.assert_array_equal(policy.carry._last_issued, np.full(64, 9))


def test_rgb_encoder_rejects_reordered_manifest_cameras():
    model = SimpleNamespace(_ode_policy_manifest={"binding": {"camera_keys": ["head", "chest"]}})
    with pytest.raises(ValueError, match="Camera order"):
        RGBEncoder(model, camera_keys=["chest", "head"], device="cpu", eval_transform=lambda **x: x)


def test_initial_pose_task_binding_and_deploy_check_are_offline(tmp_path):
    pose = {
        "schema": "ode-g1-initial-pose-v1",
        "task": {"name": "task", "revision": "v1"},
        "body29": [0] * 29,
        "action76": [0] * 64 + [1] * 12,
        "instruction": "move",
    }
    path = tmp_path / "initial_pose.json"
    path.write_text(json.dumps(pose))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert InitialPose.load(path, {"task": pose["task"], "initial_pose_sha256": digest}).action76.shape == (76,)
    with pytest.raises(ValueError, match="task/revision"):
        InitialPose.load(path, {"task": {"name": "other", "revision": "v1"}, "initial_pose_sha256": digest})
    pose["action76"][0] = 99
    path.write_text(json.dumps(pose))
    with pytest.raises(ValueError, match="pose bytes"):
        InitialPose.load(path, {"task": pose["task"], "initial_pose_sha256": digest})
    config = tmp_path / "config.yaml"
    config.write_text(
        "deployment:\n  command_endpoint: tcp://command.invalid:5556\n"
        "  state_endpoint: tcp://state.invalid:5557\n"
        "  camera_host: camera.invalid\n  network_interface: robot_nic\n"
    )
    code = "from ode_g1.cli import main; import sys; main(['deploy','--check','--config',sys.argv[1]]); assert not any(x in sys.modules for x in ['torch','zmq','teleimager','unitree_sdk2py'])"
    result = subprocess.run([sys.executable, "-c", code, str(config)], check=True, capture_output=True, text=True)
    assert '"robot_connected": false' in result.stdout
    with pytest.raises(ValueError, match="Set deployment"):
        validate_deployment_config(DEFAULTS)


def test_native_cli_uses_the_shared_policy_and_rejects_wrong_pose_before_model_load(tmp_path, monkeypatch):
    from ode_g1 import cli, deployment, runtime

    pose = {
        "schema": "ode-g1-initial-pose-v1",
        "task": {"name": "task", "revision": "v1"},
        "body29": [0] * 29,
        "action76": [0] * 64 + [1] * 12,
        "instruction": "move",
    }
    path = tmp_path / "initial_pose.json"
    path.write_text(json.dumps(pose))
    binding = {
        "task": pose["task"],
        "camera_keys": ["head", "chest"],
        "initial_pose_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (tmp_path / "policy_manifest.json").write_text(json.dumps({"binding": binding}))
    cfg = deepcopy(DEFAULTS)
    cfg["inference"]["checkpoint_dir"] = str(tmp_path)
    cfg["deployment"].update(
        command_endpoint="tcp://command.invalid:5556",
        state_endpoint="tcp://state.invalid:5557",
        camera_host="camera.invalid",
        network_interface="robot_nic",
    )
    head, normalizer = SimpleNamespace(config=SimpleNamespace(horizon=16)), Normalizer()
    model = SimpleNamespace(_ode_policy_manifest={"binding": binding})
    loaded = []

    def load(config):
        loaded.append(config)
        return model, head, normalizer

    monkeypatch.setattr(cli, "_load_policy", load)
    encoder = object()
    monkeypatch.setattr(runtime, "RGBEncoder", lambda *a, **k: encoder)

    def run(policy, initial_pose, settings, cameras):
        assert policy.head is head and policy.normalizer is normalizer and policy.rgb_encoder is encoder
        assert cameras == ["head", "chest"] and initial_pose.instruction == "move"
        return {"status": "fake_hardware_verified"}

    monkeypatch.setattr(deployment, "run_deployment", run)
    assert cli.deploy(cfg, SimpleNamespace(check=False))["status"] == "fake_hardware_verified"
    loaded.clear()
    pose["body29"][0] = 7
    path.write_text(json.dumps(pose))
    with pytest.raises(ValueError, match="pose bytes"):
        cli.deploy(cfg, SimpleNamespace(check=False))
    assert not loaded
