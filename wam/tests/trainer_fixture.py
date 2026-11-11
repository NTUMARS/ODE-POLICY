"""A tiny stochastic model exercises the native trainer, without model weights."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from ode_imagewam.config import load_config
from ode_imagewam.trainer import PortableTrainer
from ode_imagewam.checkpoint import make_binding, save_weights


class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return int(os.environ.get("TEST_FRAMES", "12"))

    def __getitem__(self, index):
        return {"x": torch.tensor([index / 12, (12 - index) / 12]), "y": torch.tensor([index / 6])}


class Model(torch.nn.Module):
    def __init__(self, dataset_key="fixed-twelve-rows"):
        super().__init__()
        self.mot = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 1))
        self.frozen_encoder = torch.nn.Linear(2, 2)
        self.ode_binding = make_binding({"fixture_dataset": dataset_key}, load_config().model)

    @property
    def dit(self):
        return self.mot

    def iter_training_losses(self, sample):
        for factor in (0.5, 1.0):
            # Random noise makes RNG restoration necessary for an exact comparison.
            prediction = self.mot(sample["x"] + 0.01 * torch.randn_like(sample["x"]))
            loss = factor * (prediction - sample["y"]).square().mean()
            yield loss, {"loss_fixture": float(loss.detach())}

    def training_loss(self, sample):
        return sum(item[0] for item in self.iter_training_losses(sample)), {}

    def training_objective_count(self):
        return 2

    def save_checkpoint(self, path, optimizer=None, step=None):
        save_weights(self, path, step, optimizer)


def main():
    output, steps, resume = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    torch.manual_seed(12)
    model = Model(sys.argv[4] if len(sys.argv) > 4 else "fixed-twelve-rows")
    frozen = {key: value.clone() for key, value in model.frozen_encoder.state_dict().items()}
    cfg = load_config(overrides=["device=cpu", "mixed_precision=no", "batch_size=2", "num_workers=0",
                                 "num_epochs=2", "learning_rate=0.01", "lr_scheduler_total_steps=4",
                                 "warmup_steps=0", "save_at_end=true", "log_every=1"])
    cfg.gradient_accumulation_steps = int(os.environ.get("TEST_ACCUMULATION", "1"))
    cfg.lr_scheduler_total_steps = int(os.environ.get("TEST_BUDGET", "4"))
    cfg.paths.output_dir = output
    cfg.max_steps = steps
    cfg.resume = None if resume == "none" else resume
    trainer = PortableTrainer(model, Dataset(), cfg=cfg)
    trainer.train()
    assert "imagewam" not in sys.modules
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    assert all(torch.equal(value, frozen[key]) for key, value in unwrapped.frozen_encoder.state_dict().items())
    assert trainer.global_step == steps
    np.save(Path(output) / "next_rng.npy", np.array([float(torch.rand(())), float(np.random.rand())]))
    (Path(output) / "assertions.json").write_text(json.dumps({"step": trainer.global_step,
        "epoch": trainer.epoch, "batch_in_epoch": trainer.batch_in_epoch,
        "optimizer_steps": [int(value["step"]) for value in trainer.optimizer.state.values()],
        "frozen_encoder_unchanged": True}))


if __name__ == "__main__":
    main()
