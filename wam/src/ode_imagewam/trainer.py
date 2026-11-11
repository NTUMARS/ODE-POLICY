"""Native ImageWAM ODE training with sequential objectives and exact resume.

Sampler and sequential DeepSpeed handling derive from the MIT-licensed ImageWAM
implementation. See THIRD_PARTY_NOTICES.md for source attribution.
"""
import json
import logging
import math
from math import ceil
from pathlib import Path

import torch
from accelerate.utils import DistributedType
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .devices import training_accelerator
from .sampling import ResumableEpochSampler, seed_everything, seed_worker
from .trainer_state import TrainerState

logger = logging.getLogger(__name__)


class PortableTrainer(TrainerState):
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg, accelerator=None, parameter_audit=None):
        if getattr(model, "ode_binding", None) is None:
            raise ValueError("The trainer requires the model's ODE data/config binding")
        if val_dataset is not None or int(cfg.eval_every):
            raise ValueError("Use the separate LIBERO evaluation commands")
        self.cfg, self.model, self.train_dataset = cfg, model, train_dataset
        self.accelerator = accelerator or training_accelerator(cfg)
        self.seed = int(cfg.seed)
        self.batch_size, self.num_workers = int(cfg.batch_size), int(cfg.num_workers)
        self.accumulation = int(cfg.gradient_accumulation_steps)
        if min(len(train_dataset), self.batch_size, self.accumulation, int(cfg.num_epochs)) < 1:
            raise ValueError("Dataset, batch size, accumulation and epochs must be positive")
        lengths = self.accelerator.gather(torch.tensor([len(train_dataset)], device=self.accelerator.device))
        if not torch.all(lengths == lengths[0]):
            raise ValueError("Dataset lengths differ across ranks")
        seed_everything(self.seed)
        self._apply_train_mode(model)
        if cfg.resume:
            resume = Path(str(cfg.resume))
            if not resume.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
            if resume.is_file():
                model.load_checkpoint(str(resume), optimizer=None)
        parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters")
        self.parameter_order = [{"name": name, "shape": list(param.shape), "dtype": str(param.dtype)} for name, param in parameters]
        self.optimizer = torch.optim.AdamW([param for _, param in parameters], lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay), betas=(0.9, 0.95), foreach=bool(cfg.optimizer_foreach))
        self.training_scope = parameter_audit(model, self.optimizer) if parameter_audit is not None else None
        self.microbatches_per_epoch = ceil(len(train_dataset) / (self.batch_size * self.accelerator.num_processes))
        self.updates_per_epoch = ceil(self.microbatches_per_epoch / self.accumulation)
        self.max_steps = (int(cfg.max_steps) if cfg.max_steps is not None
                          else self.updates_per_epoch * int(cfg.num_epochs))
        budget = int(cfg.lr_scheduler_total_steps or self.max_steps)
        if not 0 < self.max_steps <= budget:
            raise ValueError("Training stop must fit the positive LR scheduler budget")
        warmup = int(cfg.warmup_steps) if cfg.warmup_steps is not None else int(budget * 0.05)
        self.scheduler = self._build_scheduler(str(cfg.lr_scheduler_type), budget, warmup)
        self._resume_training_config = {
            "batch_size": self.batch_size, "world_size": self.accelerator.num_processes,
            "dataset_frames": len(train_dataset), "num_workers": self.num_workers,
            "gradient_accumulation_steps": self.accumulation, "num_epochs": int(cfg.num_epochs),
            "seed": self.seed, "mixed_precision": str(cfg.mixed_precision),
            "learning_rate": float(cfg.learning_rate), "weight_decay": float(cfg.weight_decay),
            "max_grad_norm": float(cfg.max_grad_norm), "scheduler": str(cfg.lr_scheduler_type),
            "scheduler_total_steps": budget, "warmup_steps": warmup}
        self.train_sampler = ResumableEpochSampler(train_dataset, self.seed, self.batch_size,
                                                   self.accelerator.num_processes)
        loader_kwargs = {"batch_size": self.batch_size, "sampler": self.train_sampler,
                         "num_workers": self.num_workers, "worker_init_fn": seed_worker,
                         "pin_memory": self.accelerator.device.type == "cuda",
                         "generator": torch.Generator().manual_seed(self.seed)}
        if self.num_workers:
            loader_kwargs.update(persistent_workers=bool(cfg.persistent_workers))
            if cfg.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
        self.train_loader = DataLoader(train_dataset, **loader_kwargs)
        self.output_dir = Path(str(cfg.output_dir))
        self.weights_dir = self.output_dir / "checkpoints/weights"
        self.state_dir = self.output_dir / "checkpoints/state"
        for directory in (self.output_dir, self.weights_dir, self.state_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            model, self.optimizer, self.train_loader, self.scheduler)
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            engine = self.accelerator.deepspeed_engine_wrapped.engine
            if (float(engine.gradient_clipping()) != float(cfg.max_grad_norm)
                    or int(engine.gradient_accumulation_steps()) != self.accumulation
                    or int(engine.zero_optimization_stage()) != 1):
                raise ValueError("DeepSpeed must match configured clipping, accumulation and ZeRO stage 1")
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step = self.epoch = self.batch_in_epoch = 0
        self.save_steps = frozenset(int(step) for step in (cfg.save_steps or []))
        if any(step < 1 for step in self.save_steps):
            raise ValueError("save_steps must be positive")
        self._wandb = None
        if bool(cfg.wandb.enabled) and self.accelerator.is_main_process:
            import wandb
            self._wandb = wandb.init(project=str(cfg.wandb.project), name=cfg.wandb.name,
                                     group=cfg.wandb.group, mode=str(cfg.wandb.mode), dir=str(self.output_dir))
        if cfg.resume:
            path = Path(str(cfg.resume))
            if path.is_dir():
                self.load_training_state(path)
            elif not path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    @staticmethod
    def _apply_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train().requires_grad_(True)
        encoder = getattr(model, "proprio_encoder", None)
        if encoder is not None:
            encoder.train().requires_grad_(True)

    def _build_scheduler(self, kind, steps, warmup):
        warmup = min(max(warmup, 0), steps - 1)
        duration = max(steps - warmup, 1)
        if kind == "cosine":
            main = CosineAnnealingLR(self.optimizer, duration, eta_min=float(self.cfg.learning_rate) * 0.01)
        elif kind == "constant":
            main = ConstantLR(self.optimizer, factor=1, total_iters=duration)
        else:
            raise ValueError("lr_scheduler_type must be cosine or constant")
        if not warmup:
            return main
        ramp = LinearLR(self.optimizer, start_factor=1 / warmup, end_factor=1, total_iters=warmup)
        return SequentialLR(self.optimizer, [ramp, main], [warmup])

    def _backward_objectives(self, sample):
        model = self.accelerator.unwrap_model(self.model)
        expected = int(model.training_objective_count())
        if expected < 1:
            raise ValueError("Model must declare a positive training objective count")
        engine = (self.accelerator.deepspeed_engine_wrapped.engine
                  if self.accelerator.distributed_type == DistributedType.DEEPSPEED else None)
        total, metrics, count = None, {}, 0
        iterator = iter(model.iter_training_losses(sample))
        while True:
            with self.accelerator.autocast():
                try:
                    loss, values = next(iterator)
                except StopIteration:
                    break
            count += 1
            if loss.ndim or not bool(self.accelerator.gather(torch.isfinite(loss.detach()).reshape(1)).all()):
                raise FloatingPointError("Training objective must be a finite scalar")
            total = loss.detach().float() if total is None else total + loss.detach().float()
            for key, value in values.items():
                metrics[key] = metrics.get(key, 0) + float(value)
            if engine is None:
                self.accelerator.backward(loss)
            else:
                # Accelerator.backward would step DeepSpeed after EACH objective.
                engine.set_gradient_accumulation_boundary(
                    is_boundary=self.accelerator.sync_gradients and count == expected)
                engine.backward(loss)
        if count != expected:
            raise RuntimeError(f"Training objective count mismatch: expected {expected}, got {count}")
        return total, metrics

    def _optimizer_step(self):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            engine = self.accelerator.deepspeed_engine_wrapped.engine
            # Engine.step advances micro_steps every batch and updates only at
            # the boundary selected above. Its optimizer owns configured clipping.
            engine.step()
            if not self.accelerator.sync_gradients:
                return None
            grad_norm = engine.get_global_grad_norm()
        else:
            if not self.accelerator.sync_gradients:
                return None
            grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), float(self.cfg.max_grad_norm))
            if grad_norm is not None and not math.isfinite(float(grad_norm)):
                raise FloatingPointError("Training gradient norm must be finite")
            self.optimizer.step()
            if not self.accelerator.optimizer_step_was_skipped:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        if self.accelerator.optimizer_step_was_skipped:
            raise FloatingPointError("Optimizer skipped an update; no continuation state was saved")
        if grad_norm is not None and not math.isfinite(float(grad_norm)):
            raise FloatingPointError("Training gradient norm must be finite")
        return grad_norm

    def _record_metrics(self, loss, values, grad_norm):
        averaged = {}
        for key, value in values.items():
            tensor = torch.tensor(float(value), device=self.accelerator.device).reshape(1)
            averaged[key] = float(self.accelerator.gather(tensor).mean())
        metrics = {"loss": float(self.accelerator.gather(loss.reshape(1)).mean()), **averaged,
                   "step": self.global_step, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
                   "learning_rate": float(self.optimizer.param_groups[0]["lr"])}
        if grad_norm is not None:
            metrics["grad_norm"] = float(grad_norm)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Training metrics must be finite")
        if self.accelerator.is_main_process:
            with (self.output_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            logger.info("step=%d/%d loss=%.5f lr=%.3g", self.global_step, self.max_steps,
                        metrics["loss"], metrics["learning_rate"])
            if self._wandb is not None:
                self._wandb.log(metrics, step=self.global_step)

    def train(self):
        self._apply_train_mode(self.accelerator.unwrap_model(self.model))
        iterator = iter(self.train_loader)
        starting_step = self.global_step
        try:
            while self.global_step < self.max_steps:
                try:
                    sample = next(iterator)
                except StopIteration:
                    self.epoch += 1
                    self.batch_in_epoch = 0
                    self.train_sampler.clear_resume_batch_offset()
                    iterator = iter(self.train_loader)
                    continue
                self.batch_in_epoch += 1
                # ZeRO controls its own reductions. Entering engine.no_sync here
                # would forbid engine.step on a non-boundary microbatch.
                models = () if self.accelerator.distributed_type == DistributedType.DEEPSPEED else (self.model,)
                with self.accelerator.accumulate(*models):
                    loss, values = self._backward_objectives(sample)
                    grad_norm = self._optimizer_step()
                    if not self.accelerator.sync_gradients:
                        continue
                    self.global_step += 1
                    if self.global_step % int(self.cfg.log_every) == 0:
                        self._record_metrics(loss, values, grad_norm)
                    if self.global_step in self.save_steps or (int(self.cfg.save_every) > 0 and self.global_step % int(self.cfg.save_every) == 0):
                        self.save_checkpoint()
            if self.global_step > starting_step and bool(self.cfg.save_at_end) and not (self.global_step in self.save_steps or
                    (int(self.cfg.save_every) > 0 and self.global_step % int(self.cfg.save_every) == 0)):
                self.save_checkpoint()
        finally:
            if self._wandb is not None:
                self._wandb.finish()
