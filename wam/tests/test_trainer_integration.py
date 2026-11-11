import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def test_native_trainer_two_objectives_and_exact_fullstate_resume(tmp_path):
    env = dict(os.environ)
    env.pop("IMAGEWAM_SOURCE", None)
    fixture = Path(__file__).with_name("trainer_fixture.py")
    complete, interrupted, resumed = (tmp_path / value for value in ("complete", "interrupted", "resumed"))
    def run(output, steps, resume="none"):
        result = subprocess.run([sys.executable, str(fixture), str(output), str(steps), str(resume)],
                                env=env, text=True, capture_output=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    run(complete, 4)
    run(interrupted, 2)
    state = interrupted / "checkpoints/state/step_000002"
    assert (state / "trainer_state.json").exists()
    assert (state / "optimizer.bin").exists()
    assert (state / "ode_binding.json").exists()
    wrong = subprocess.run([sys.executable, str(fixture), str(tmp_path / "wrong-task"), "4", str(state), "other-task"],
                           env=env, text=True, capture_output=True, timeout=90)
    assert wrong.returncode != 0 and "binding" in wrong.stderr
    assert not list((tmp_path / "wrong-task/checkpoints/weights").glob("*.pt"))
    run(resumed, 4, state)
    for key, value in torch.load(complete / "checkpoints/weights/step_000004.pt", weights_only=True)["mot"].items():
        actual = torch.load(resumed / "checkpoints/weights/step_000004.pt", weights_only=True)["mot"][key]
        torch.testing.assert_close(value, actual, atol=0, rtol=0)
    np.testing.assert_array_equal(np.load(complete / "next_rng.npy"), np.load(resumed / "next_rng.npy"))
    summary = json.loads((resumed / "assertions.json").read_text())
    assert summary["step"] == summary["batch_in_epoch"] == 4
    assert summary["optimizer_steps"] == [4] * 4  # two objectives, one AdamW update per batch
    assert summary["frozen_encoder_unchanged"]
