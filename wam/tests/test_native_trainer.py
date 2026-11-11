from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from accelerate.utils import DistributedType
import numpy as np
import pytest
import torch

from ode_imagewam.config import load_config
from ode_imagewam.sampling import ResumableEpochSampler
from ode_imagewam.trainer import PortableTrainer


class LossModel(torch.nn.Module):
    def __init__(self, count=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.count = count

    @property
    def dit(self):
        return self

    def iter_training_losses(self, _sample):
        yield (2 * self.weight - 3).square(), {}
        yield (self.weight - 2).square(), {}

    def training_objective_count(self):
        return self.count


class Engine:
    def __init__(self, model, accumulation=1, clipping=10):
        self.boundaries, self.backward_calls, self.step_calls = [], 0, 0
        self.micro_steps = self.optimizer_steps = self.scheduler_steps = 0
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.accumulation, self.clipping, self.norm = accumulation, clipping, None

    def set_gradient_accumulation_boundary(self, *, is_boundary):
        self.boundaries.append(is_boundary)

    def backward(self, loss):
        self.backward_calls += 1
        (loss / self.accumulation).backward()

    def step(self):
        self.step_calls += 1
        if self.boundaries[-1]:
            params = self.optimizer.param_groups[0]["params"]
            self.norm = torch.nn.utils.clip_grad_norm_(params, self.clipping, error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.optimizer_steps += 1
            self.scheduler_steps += 1
        self.micro_steps += 1

    def get_global_grad_norm(self):
        return self.norm


def trainer_with_engine(model, sync=True, accumulation=1, clipping=10):
    engine = Engine(model, accumulation, clipping)
    trainer = PortableTrainer.__new__(PortableTrainer)
    trainer.model = model
    trainer.cfg = SimpleNamespace(max_grad_norm=clipping)
    trainer.accelerator = SimpleNamespace(
        unwrap_model=lambda value: value, autocast=nullcontext, gather=lambda value: value,
        sync_gradients=sync, distributed_type=DistributedType.DEEPSPEED,
        optimizer_step_was_skipped=False, deepspeed_engine_wrapped=SimpleNamespace(engine=engine))
    return trainer, engine


def test_sequential_deepspeed_objectives_step_only_once_after_both_gradients():
    model = LossModel()
    trainer, engine = trainer_with_engine(model)
    loss, _ = trainer._backward_objectives({})
    assert engine.backward_calls == 2 and engine.step_calls == 0
    trainer._optimizer_step()
    assert engine.step_calls == engine.optimizer_steps == 1
    assert engine.boundaries == [False, True]
    torch.testing.assert_close(model.weight, torch.tensor(1.6))
    assert float(loss) == 2


def test_deepspeed_accumulation_and_wrong_objective_count_do_not_update():
    trainer, engine = trainer_with_engine(LossModel(), sync=False)
    trainer._backward_objectives({})
    trainer._optimizer_step()
    assert engine.boundaries == [False, False]
    assert engine.micro_steps == engine.step_calls == 1 and engine.optimizer_steps == 0
    trainer, engine = trainer_with_engine(LossModel(count=3))
    with pytest.raises(RuntimeError, match="objective count"):
        trainer._backward_objectives({})
    assert engine.step_calls == 0


def test_deepspeed_gas2_clips_once_and_preserves_every_microstep():
    model = LossModel()
    trainer, engine = trainer_with_engine(model, accumulation=2, clipping=0.5)
    trainer.accumulation = 2
    trainer.cfg = SimpleNamespace(max_grad_norm=0.5, log_every=100, save_every=0, save_at_end=False)
    trainer.train_loader = [{}, {}]
    trainer.global_step = trainer.epoch = trainer.batch_in_epoch = 0
    trainer.max_steps, trainer.save_steps, trainer._wandb = 1, frozenset(), None
    calls = []
    @contextmanager
    def accumulate(*models):
        # A ZeRO1 engine.no_sync context forbids Engine.step, even off-boundary.
        assert not models
        calls.append(True)
        trainer.accelerator.sync_gradients = len(calls) % 2 == 0
        yield
    trainer.accelerator.accumulate = accumulate
    trainer.train()
    assert engine.micro_steps == engine.step_calls == 2
    assert engine.optimizer_steps == engine.scheduler_steps == trainer.global_step == 1
    assert engine.backward_calls == 4 and engine.boundaries == [False, False, False, True]
    assert float(engine.norm) == 6
    torch.testing.assert_close(model.weight, torch.tensor(1.05), atol=1e-6, rtol=0)


def test_parameter_audit_runs_before_prepare_remaps_optimizer_parameters(tmp_path):
    model = LossModel()
    model.ode_binding = {"test": "binding"}
    trace = []
    class Accelerator:
        device, num_processes, is_main_process = torch.device("cpu"), 1, True
        distributed_type = DistributedType.NO
        @staticmethod
        def gather(value):
            return value
        @staticmethod
        def prepare(model, optimizer, loader, scheduler):
            assert trace == ["audit"]
            trace.append("prepare")
            masters = [torch.nn.Parameter(p.detach().clone().float())
                       for group in optimizer.param_groups for p in group["params"]]
            return model, torch.optim.AdamW(masters), loader, scheduler
    def audit(original, optimizer):
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert all(id(p) in optimized for p in original.parameters() if p.requires_grad)
        trace.append("audit")
        return {"all_parameters_optimized": True}
    cfg = load_config(overrides=["device=cpu", "mixed_precision=no", "num_workers=0"])
    cfg.paths.output_dir = str(tmp_path)
    dataset = torch.utils.data.TensorDataset(torch.arange(4))
    trainer = PortableTrainer(model, dataset, cfg=cfg, accelerator=Accelerator(), parameter_audit=audit)
    assert trace == ["audit", "prepare"]
    assert trainer.training_scope == {"all_parameters_optimized": True}
    after = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in after for p in model.parameters())


def test_epoch_sampler_resumes_shared_permutation_at_global_microbatch_offset():
    sampler = ResumableEpochSampler(range(19), 42, 2, 2)
    sampler.set_epoch_offset(3)
    sampler.set_resume_batch_offset(2)
    expected = torch.randperm(19, generator=torch.Generator().manual_seed(45)).tolist()
    padding = (-len(expected)) % 4
    padded = expected + expected[:padding]
    assert list(sampler) == padded[8:]
    sampler.clear_resume_batch_offset()
    sampler.set_epoch(1)
    expected = torch.randperm(19, generator=torch.Generator().manual_seed(46)).tolist()
    assert list(sampler) == expected + expected[:(-len(expected)) % 4]


def test_distributed_resume_retains_the_original_padded_tail():
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler
    for frames in range(1, 20):
        order = torch.randperm(frames, generator=torch.Generator().manual_seed(42)).tolist()
        for rank in range(3):
            expected = list(BatchSamplerShard(BatchSampler(order, 2, False), 3, rank))
            sampler = ResumableEpochSampler(range(frames), 42, 2, 3)
            actual = list(BatchSamplerShard(BatchSampler(sampler, 2, False), 3, rank))
            assert actual == expected
            sampler.set_resume_batch_offset(1)
            resumed = list(BatchSamplerShard(BatchSampler(sampler, 2, False), 3, rank))
            assert resumed == expected[1:]


@pytest.mark.parametrize("interrupted_steps", [2, 3])
def test_exact_accumulated_resume_across_short_epoch_tail(tmp_path, interrupted_steps):
    env = dict(os.environ, TEST_ACCUMULATION="2", TEST_FRAMES="10", TEST_BUDGET="5")
    env.pop("IMAGEWAM_SOURCE", None)
    fixture = Path(__file__).with_name("trainer_fixture.py")
    def run(name, steps, resume="none"):
        output = tmp_path / name
        process = subprocess.run([sys.executable, str(fixture), str(output), str(steps), str(resume)],
                                 env=env, capture_output=True, text=True, timeout=90)
        assert process.returncode == 0, process.stdout + process.stderr
        return output
    complete = run("complete", 5)
    interrupted = run("interrupted", interrupted_steps)
    state = interrupted / f"checkpoints/state/step_{interrupted_steps:06d}"
    if interrupted_steps == 3:
        progress = json.loads((state / "trainer_state.json").read_text())
        assert progress["epoch"] == 1 and progress["batch_in_epoch"] == 0
    resumed = run("resumed", 5, state)
    for key, expected in torch.load(complete / "checkpoints/weights/step_000005.pt", weights_only=True)["mot"].items():
        actual = torch.load(resumed / "checkpoints/weights/step_000005.pt", weights_only=True)["mot"][key]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(np.load(complete / "next_rng.npy"), np.load(resumed / "next_rng.npy"))
    assertions = json.loads((resumed / "assertions.json").read_text())
    assert assertions["optimizer_steps"] == [5] * 4
