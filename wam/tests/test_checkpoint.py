import pytest
import torch

from ode_imagewam.checkpoint import make_binding, save_weights, strict_load_checkpoint
from ode_imagewam.config import load_config


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mot = torch.nn.Linear(2, 3)
        self.proprio_encoder = torch.nn.Linear(8, 2)
        self.ode_binding = make_binding({"fixture_dataset": "task-a"}, load_config().model)


def test_full_checkpoint_and_missing_encoder_are_distinguished(tmp_path):
    model = Model()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "weights.pt"
    payload = {"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict(),
               "step": 123, "ode_binding": model.ode_binding}
    torch.save(payload, path)
    model.mot.weight.data.zero_()
    assert strict_load_checkpoint(model, path) == 123
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    torch.save({"mot": model.mot.state_dict(), "ode_binding": model.ode_binding}, path)
    with pytest.raises(ValueError, match="proprio encoder"):
        strict_load_checkpoint(model, path)


def test_partial_mot_checkpoint_is_rejected(tmp_path):
    model = Model()
    path = tmp_path / "partial.pt"
    torch.save({"mot": {"bias": model.mot.bias.detach()}, "ode_binding": model.ode_binding}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        strict_load_checkpoint(model, path)


def test_wrong_task_config_and_unbound_weights_fail_before_copying(tmp_path):
    source = Model()
    path = tmp_path / "weights.pt"
    save_weights(source, path, step=7)
    target = Model()
    before = target.mot.weight.detach().clone()
    target.ode_binding = make_binding({"fixture_dataset": "different-task"}, load_config().model)
    with pytest.raises(ValueError, match="binding"):
        strict_load_checkpoint(target, path)
    torch.testing.assert_close(target.mot.weight, before)
    changed = load_config(overrides=["model.ode.vel_scale=0.1"])
    target.ode_binding = make_binding({"fixture_dataset": "task-a"}, changed.model)
    with pytest.raises(ValueError, match="binding"):
        strict_load_checkpoint(target, path)
    torch.save({"mot": source.mot.state_dict()}, path)
    with pytest.raises(ValueError, match="binding"):
        strict_load_checkpoint(source, path)


def test_binding_is_portable_but_keeps_normalizer_and_method_identity():
    a, b = load_config(), load_config()
    a.paths.sidecar_dir = "/first/sidecar"
    b.paths.sidecar_dir = "/moved/sidecar"
    b.model.load_text_encoder = True
    metadata = {"normalization_stats_sha256": "same-stats", "datasets": ["ordered-content"]}
    assert make_binding(metadata, a.model) == make_binding(metadata, b.model)
    changed = {**metadata, "normalization_stats_sha256": "other-stats"}
    assert make_binding(changed, a.model) != make_binding(metadata, a.model)
