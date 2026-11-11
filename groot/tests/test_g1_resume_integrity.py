import json
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from ode_g1 import training
from test_g1_training import Dataset, TinyHead, conditions


def paused_run(directory):
    options = training.ODETrainingOptions(epochs=2, batch_size=2, device="cpu", log_every_steps=100)
    training.train_ode(
        TinyHead(),
        Dataset(),
        conditions,
        directory,
        options,
        identity={"content": "fixed"},
        stop_requested=lambda: True,
    )
    return options


def resume(directory, options):
    return training.train_ode(
        TinyHead(),
        Dataset(),
        conditions,
        directory,
        options,
        identity={"content": "fixed"},
        resume=directory / "resume_latest.pt",
    )


def rewrite_trusted_state(directory, modify):
    path = directory / "resume_latest.pt"
    document = torch.load(path, weights_only=False)
    modify(document)
    torch.save(document, path)
    training._append_state_footer(path)
    receipt_path = directory / "resume_latest.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(sha256=training.sha256(path), size_bytes=path.stat().st_size)
    receipt_path.write_text(json.dumps(receipt))


def test_resume_rejects_changed_full_state_bytes_before_deserialization(tmp_path):
    options = paused_run(tmp_path)
    with (tmp_path / "resume_latest.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA receipt"):
        resume(tmp_path, options)


def test_resume_rejects_same_length_log_metric_tampering(tmp_path):
    options = paused_run(tmp_path)
    path = tmp_path / "metrics.jsonl"
    row = json.loads(path.read_text())
    row["loss"] = row["loss"] + 0.01
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="prefix differs"):
        resume(tmp_path, options)


def test_resume_rejects_optimizer_parameter_order_even_with_valid_byte_receipt(tmp_path):
    options = paused_run(tmp_path)
    rewrite_trusted_state(tmp_path, lambda d: d["optimizer_parameter_names"].reverse())
    with pytest.raises(ValueError, match="name/order"):
        resume(tmp_path, options)


def test_resume_rejects_cursor_counter_disagreement(tmp_path):
    options = paused_run(tmp_path)
    rewrite_trusted_state(tmp_path, lambda d: d["progress"].update(next_batch=2))
    with pytest.raises(ValueError, match="cursor"):
        resume(tmp_path, options)


def test_sealed_state_is_torch_loadable_and_receipt_retains_full_file_sha(tmp_path):
    paused_run(tmp_path)
    path = tmp_path / "resume_latest.pt"
    data = path.read_bytes()
    magic, length, expected = training._STATE_FOOTER.unpack(data[-training._STATE_FOOTER.size :])
    assert magic == training._STATE_FOOTER_MAGIC
    assert length + training._STATE_FOOTER.size == len(data)
    assert hashlib.sha256(data[:length]).digest() == expected
    receipt = json.loads((tmp_path / "resume_latest.json").read_text())
    assert receipt["sha256"] == hashlib.sha256(data).hexdigest()
    assert receipt["size_bytes"] == len(data)
    document = torch.load(path, map_location="cpu", weights_only=False)
    assert document["container_format"] == training._STATE_CONTAINER
    assert document["progress"]["optimizer_steps"] == 1


@pytest.mark.parametrize("part", ["payload", "magic", "length", "digest"])
def test_corrupt_payload_or_footer_is_rejected_before_torch_load(tmp_path, part):
    options = paused_run(tmp_path)
    path = tmp_path / "resume_latest.pt"
    data = bytearray(path.read_bytes())
    positions = {"payload": len(data) // 3, "magic": -48, "length": -33, "digest": -1}
    data[positions[part]] ^= 1
    path.write_bytes(data)
    with patch.object(training.torch, "load", side_effect=AssertionError("corrupt state must not deserialize")):
        with pytest.raises(ValueError, match="footer|SHA receipt"):
            resume(tmp_path, options)


def test_new_format_without_footer_cannot_fall_back_to_a_rewritten_external_receipt(tmp_path):
    options = paused_run(tmp_path)
    path = tmp_path / "resume_latest.pt"
    path.write_bytes(path.read_bytes()[: -training._STATE_FOOTER.size])
    receipt_path = tmp_path / "resume_latest.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(sha256=training.sha256(path), size_bytes=path.stat().st_size)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="footer is missing"):
        resume(tmp_path, options)


def test_legacy_format_still_requires_the_original_external_full_file_sha(tmp_path):
    options = paused_run(tmp_path)
    path = tmp_path / "resume_latest.pt"
    document = torch.load(path, weights_only=False)
    document.pop("container_format")
    torch.save(document, path)
    receipt_path = tmp_path / "resume_latest.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(sha256=training.sha256(path), size_bytes=path.stat().st_size)
    receipt_path.write_text(json.dumps(receipt))
    assert not training._verify_state_footer(path)
    valid_bytes = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA receipt"):
        resume(tmp_path, options)
    path.write_bytes(valid_bytes)
    assert resume(tmp_path, options)["optimizer_steps"] == 6


@pytest.mark.parametrize("milestone", [False, True])
def test_replace_before_latest_json_interruption_still_resumes_exactly(tmp_path, monkeypatch, milestone):
    torch.manual_seed(519)
    initial = TinyHead().state_dict()

    def fresh():
        head = TinyHead()
        head.load_state_dict(initial)
        return head

    options = training.ODETrainingOptions(
        epochs=2, batch_size=2, device="cpu", checkpoint_milestone_epochs=(1, 2) if milestone else ()
    )
    baseline = fresh()
    training.train_ode(baseline, Dataset(), conditions, tmp_path / "baseline", options, identity={})
    output = tmp_path / "interrupted"
    if not milestone:
        training.train_ode(fresh(), Dataset(), conditions, output, options, identity={}, stop_requested=lambda: True)
        old_json = (output / "resume_latest.json").read_bytes()
    else:
        old_json = None
    original = training.atomic_json

    def fail_after_state_replace(path, value):
        if Path(path).name == "resume_latest.json":
            raise InterruptedError("process stopped after state replacement, before JSON replacement")
        return original(path, value)

    with monkeypatch.context() as context:
        context.setattr(training, "atomic_json", fail_after_state_replace)
        with pytest.raises(InterruptedError, match="after state replacement"):
            training.train_ode(
                fresh(),
                Dataset(),
                conditions,
                output,
                options,
                identity={},
                resume=None if milestone else output / "resume_latest.pt",
                stop_requested=None if milestone else lambda: True,
            )
    if milestone:
        state = output / "checkpoints/epoch_001/training_state.pt"
        assert state.stat().st_ino == (output / "resume_latest.pt").stat().st_ino
        assert not (output / "resume_latest.json").exists()
    else:
        assert (output / "resume_latest.json").read_bytes() == old_json
        assert torch.load(output / "resume_latest.pt", weights_only=False)["progress"]["optimizer_steps"] == 2
    restored = fresh()
    result = training.train_ode(
        restored, Dataset(), conditions, output, options, identity={}, resume=output / "resume_latest.pt"
    )
    assert result["optimizer_steps"] == 6
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
