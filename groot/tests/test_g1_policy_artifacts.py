import json
import shutil
from types import SimpleNamespace

import pytest
import torch

from ode_g1 import data, loader, training
from ode_g1.cli import _offload_cached_backbone
from ode_g1.features import combine_conditions
from test_g1_data_windows import fixture_episode, spec_for
from test_g1_model_contracts import SonicODEConfig, TinyNativeHead, install_sonic_ode
from test_g1_training import Dataset, TinyHead, conditions


@pytest.fixture
def policy_artifact(tmp_path):
    config = SonicODEConfig(horizon=4, feature_dim=8, cond_dim=8, width=16)
    head = install_sonic_ode(TinyNativeHead(), config).float().eval()
    normalizer = data.fit_normalization([fixture_episode()], spec_for([fixture_episode()]))
    (tmp_path / "ode_config.json").write_text(json.dumps(vars(config)))
    normalizer.save(tmp_path / "normalization.json")
    base = {"revision": "toy", "weights": {"toy.safetensors": {"sha256": "a" * 64, "bytes": 5}}}
    binding = loader.make_policy_binding(tmp_path, base_identity=base, camera_keys=["front", "side"])
    receipt = training.save_head_weights(
        tmp_path / "final_head.safetensors", head, final=True, policy_binding_sha256=loader._binding_digest(binding)
    )
    loader.write_policy_manifest(tmp_path, binding, receipt)
    return tmp_path, base, head


def load_fixture(monkeypatch, artifact, base=None, **kwargs):
    root, expected, head = artifact
    monkeypatch.setattr(
        loader,
        "load_base",
        lambda *a, **k: SimpleNamespace(action_head=TinyNativeHead(), _ode_base_identity=base or expected),
    )
    return loader.load_policy("toy", root, device="cpu", camera_keys=["front", "side"], **kwargs)


def test_bound_policy_loads_and_rejects_other_normalizer_or_base(monkeypatch, policy_artifact):
    model, head, normalizer = load_fixture(monkeypatch, policy_artifact)
    assert model._ode_policy_manifest["binding"]["camera_keys"] == ["front", "side"]
    with pytest.raises(ValueError, match="base-model"):
        load_fixture(monkeypatch, policy_artifact, base={"revision": "different"})
    root, _, _ = policy_artifact
    statistics = normalizer.statistics.copy()
    statistics["task"] = "other-task"
    (root / "normalization.json").write_text(json.dumps(statistics))
    with pytest.raises(ValueError, match="config/normalizer binding"):
        load_fixture(monkeypatch, policy_artifact)


def test_rejects_same_architecture_other_head_and_camera_order(monkeypatch, policy_artifact):
    root, base, head = policy_artifact
    with pytest.raises(ValueError, match="camera order"):
        loader.load_policy("unused", root, device="cpu", camera_keys=["side", "front"])
    identifier = json.loads((root / "policy_manifest.json").read_text())["binding_sha256"]
    training.save_head_weights(root / "final_head.safetensors", head, final=True, policy_binding_sha256=identifier)
    with pytest.raises(ValueError, match="head header binding"):
        load_fixture(monkeypatch, policy_artifact)


def test_bound_header_load_does_not_rehash_head_but_legacy_requires_it(monkeypatch, policy_artifact):
    root, _, head = policy_artifact
    original = training.sha256
    scanned = []

    def digest(path):
        scanned.append(str(path))
        return original(path)

    monkeypatch.setattr(training, "sha256", digest)
    load_fixture(monkeypatch, policy_artifact)
    assert not any(p.endswith("final_head.safetensors") for p in scanned)
    load_fixture(monkeypatch, policy_artifact, verify_head_sha256=True)
    assert any(p.endswith("final_head.safetensors") for p in scanned)
    manifest = json.loads((root / "policy_manifest.json").read_text())
    receipt = training.save_head_weights(root / "final_head.safetensors", head, final=True)
    loader.write_policy_manifest(root, manifest["binding"], receipt)
    scanned.clear()
    load_fixture(monkeypatch, policy_artifact)
    assert any(p.endswith("final_head.safetensors") for p in scanned)


def test_relocated_completed_training_run_can_resume_final_publication(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "moved"
    options = training.ODETrainingOptions(epochs=2, batch_size=2, device="cpu", checkpoint_milestone_epochs=(1, 2))
    result = training.train_ode(TinyHead(), Dataset(), conditions, old, options, identity={"toy": 1})
    shutil.move(old, new)
    restored = training.train_ode(
        TinyHead(), Dataset(), conditions, new, options, identity={"toy": 1}, resume=new / "final_training_state.pt"
    )
    assert restored["optimizer_steps"] == result["optimizer_steps"]
    assert restored["final_weights"]["sha256"] == result["final_weights"]["sha256"]
    assert restored["final_weights"]["path"] == "final_head.safetensors"
    for milestone in restored["milestone_checkpoints"]:
        assert (new / milestone["training_state"]["path"]).is_file()


def test_mid_epoch_resume_after_move_matches_uninterrupted_optimizer(tmp_path):
    torch.manual_seed(91)
    initial = TinyHead().state_dict()

    def fresh():
        head = TinyHead()
        head.load_state_dict(initial)
        return head

    options = training.ODETrainingOptions(epochs=2, batch_size=2, device="cpu", checkpoint_milestone_epochs=(1, 2))
    baseline = fresh()
    training.train_ode(baseline, Dataset(), conditions, tmp_path / "baseline", options, identity={"toy": 1})
    calls = []

    def stop():
        calls.append(1)
        return len(calls) == 4

    old = tmp_path / "old"
    new = tmp_path / "new"
    training.train_ode(fresh(), Dataset(), conditions, old, options, identity={"toy": 1}, stop_requested=stop)
    shutil.move(old, new)
    resumed = fresh()
    result = training.train_ode(
        resumed, Dataset(), conditions, new, options, identity={"toy": 1}, resume=new / "resume_latest.pt"
    )
    assert result["optimizer_steps"] == 6
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


def test_cache_only_training_offloads_real_backbone_and_never_calls_it(tmp_path):
    from ode_g1.features import FrozenFrameEncoder
    from test_g1_runtime import fake_model, identity_transform
    from test_g1_portable import source_fixture

    root = tmp_path / "source"
    cameras = source_fixture(root)
    model = fake_model()
    moves = []
    original_to = model.backbone.to

    def move(device):
        moves.append(str(device))
        return original_to(device)

    model.backbone.to = move

    class Reader:
        def read(self, path, frame):
            import numpy as np

            return np.ones((12, 18, 3), dtype=np.uint8)

    encoder = FrozenFrameEncoder(
        model,
        root,
        "cpu",
        eval_transform=identity_transform,
        frame_reader=Reader(),
        camera_keys=cameras,
        feature_cache_dir=tmp_path / "features",
        feature_cache_identity={"toy": 1},
    )
    encoder.encode_frames(Dataset.examples)
    encoder.require_cache_hits = True
    calls = model.backbone.calls
    _offload_cached_backbone(model, encoder)
    assert moves == ["cpu"] and next(model.backbone.parameters()).device.type == "cpu"

    # The tiny optimizer uses one scalar from each actually cached observation.
    def provider(keys):
        return combine_conditions(encoder.encode_frames(keys))["backbone_features"][:, 0, :1]

    options = training.ODETrainingOptions(epochs=1, batch_size=2, device="cpu")
    training.train_ode(TinyHead(), Dataset(), provider, tmp_path / "train", options, identity={})
    assert model.backbone.calls == calls
