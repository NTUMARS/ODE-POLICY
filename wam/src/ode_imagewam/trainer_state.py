"""Weights, optimizer/RNG state and the exact sampler cursor."""
import json
import re
import shutil
from pathlib import Path

from .checkpoint import require_binding


class TrainerState:
    def _state_binding(self):
        return {"model": self.accelerator.unwrap_model(self.model).ode_binding,
                "training": self._resume_training_config,
                "parameters": self.parameter_order}

    def save_checkpoint(self):
        tag = f"step_{self.global_step:06d}"
        state = self.state_dir / tag
        weights = self.weights_dir / f"{tag}.pt"
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).save_checkpoint(
                str(weights), optimizer=None, step=self.global_step)
        state.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(str(state))
        if self.accelerator.is_main_process:
            epoch, batch = self.epoch, self.batch_in_epoch
            if batch == self.microbatches_per_epoch:
                epoch, batch = epoch + 1, 0
            progress = {"global_step": self.global_step, "epoch": epoch,
                        "batch_in_epoch": batch}
            (state / "trainer_state.json").write_text(json.dumps(progress, indent=2) + "\n")
            (state / "ode_binding.json").write_text(json.dumps(self._state_binding(), indent=2) + "\n")
            self._prune_checkpoints(tag)
        self.accelerator.wait_for_everyone()
        return {"weights_path": str(weights), "state_path": str(state)}

    def _prune_checkpoints(self, current_tag):
        if bool(self.cfg.keep_latest_state_only):
            for path in self.state_dir.iterdir():
                if path.is_dir() and path.name != current_tag and re.fullmatch(r"step_\d+", path.name):
                    shutil.rmtree(path)
        keep = self.cfg.keep_latest_weights
        if keep is not None and int(keep) > 0:
            files = sorted(self.weights_dir.glob("step_*.pt"))
            for path in files[:-int(keep)]:
                path.unlink()

    def load_training_state(self, state_dir):
        state = Path(state_dir)
        binding = state / "ode_binding.json"
        if not binding.is_file():
            raise ValueError("Full-state checkpoint is missing its ODE data/config binding")
        require_binding(json.loads(binding.read_text()), self._state_binding())
        progress = json.loads((state / "trainer_state.json").read_text())
        step, epoch, batch = (progress[key] for key in ("global_step", "epoch", "batch_in_epoch"))
        if any(type(value) is not int or value < 0 for value in (step, epoch, batch)):
            raise ValueError("Checkpoint counters must be nonnegative integers")
        if batch >= self.microbatches_per_epoch:
            raise ValueError("Checkpoint sampler cursor is outside its epoch")
        expected = epoch * self.updates_per_epoch + (batch + self.accumulation - 1) // self.accumulation
        if step != expected or step > self.max_steps:
            raise ValueError("Checkpoint optimizer count and sampler cursor disagree")
        if batch and batch % self.accumulation:
            raise ValueError("Checkpoint cursor is not at an optimizer boundary")
        self.accelerator.load_state(str(state))
        self.global_step, self.epoch, self.batch_in_epoch = step, epoch, batch
        self.train_sampler.set_epoch_offset(epoch)
        self.train_sampler.set_resume_batch_offset(batch)
        self.accelerator.wait_for_everyone()
