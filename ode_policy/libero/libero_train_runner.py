"""Training runner (LIBERO, nav2d, or any zarr in the same layout).

Training only: evaluation is a separate script (``libero_eval_runner.py`` /
``eval_ode_policy.py``) that drives LIBERO's own simulator.
"""
from __future__ import annotations

import copy
import os
import random
import time

import hydra
import numpy as np
import torch
import tqdm
from loguru import logger as log
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ode_policy.runners.base_runner import BaseRunner
from ode_policy.utils.json_logger import JsonLogger
from ode_policy.utils.lr_scheduler import get_scheduler
from ode_policy.utils.pytorch_util import optimizer_to


class _BatchSampler:
    """Yields arrays of sample indices; the datasets fill pinned buffers in one go.

    With world_size > 1 (torchrun / DDP) every rank draws the SAME permutation (same seed) and
    takes every world_size-th batch of it, so the shards are disjoint and of equal length -- DDP
    needs the same number of steps on every rank. world_size == 1 is the original behaviour.
    """

    def __init__(self, data_size: int, batch_size: int, shuffle: bool = False,
                 seed: int = 0, drop_last: bool = True, rank: int = 0, world_size: int = 1):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.rank, self.world_size = rank, world_size
        self.num_batch_total = data_size // batch_size
        self.num_batch = self.num_batch_total // world_size
        self.discard = data_size - batch_size * self.num_batch_total
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        perm = self.rng.permutation(self.data_size) if self.shuffle else np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[: -self.discard]
        perm = perm.reshape(self.num_batch_total, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i * self.world_size + self.rank]

    def __len__(self):
        return self.num_batch


class _LossModule(torch.nn.Module):
    """DistributedDataParallel hooks the gradient all-reduce on forward(); the policies compute
    their loss in compute_loss(), so this adapter makes that the forward."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.compute_loss(batch)


def _create_dataloader(dataset, *, batch_size, shuffle, num_workers, pin_memory,
                       persistent_workers, seed: int = 0, rank: int = 0, world_size: int = 1):
    batch_sampler = _BatchSampler(len(dataset), batch_size, shuffle=shuffle,
                                  seed=seed, drop_last=True, rank=rank, world_size=world_size)

    def collate(x):
        assert len(x) == 1
        return x[0]

    return DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        # The datasets already hand back pinned tensors; letting the loader pin
        # again would copy every batch a second time.
        pin_memory=False,
        persistent_workers=persistent_workers,
    )


class LiberoTrainRunner(BaseRunner):
    include_keys = ("global_step", "epoch")

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.train_config.training_params.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = hydra.utils.instantiate(cfg.policy_config)
        self.policy_name = cfg.policy_name

        self.ema_model = None
        if cfg.train_config.training_params.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.train_config.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def train(self):
        cfg = copy.deepcopy(self.cfg)

        # Multi-GPU via torchrun (one process per GPU, DistributedDataParallel over the loss).
        # Without torchrun WORLD_SIZE is unset and everything below is the single-process path.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        distributed = world_size > 1
        is_main = rank == 0
        if distributed:
            import torch.distributed as dist
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            # nccl on GPUs; gloo on CPU-only nodes (DDP_BACKEND overrides) so the DDP path can be
            # exercised in a CPU job
            use_cuda = torch.cuda.is_available()
            dist.init_process_group(os.environ.get("DDP_BACKEND", "nccl" if use_cuda else "gloo"))
            if use_cuda:
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
            else:
                device = torch.device("cpu")
            # the model was initialised identically on every rank (seeded in __init__, and DDP
            # broadcasts rank 0's weights anyway); the training noise should differ per rank
            torch.manual_seed(cfg.train_config.training_params.seed + rank)
            log.info(f"DDP rank {rank}/{world_size} on {device}; per-rank batch "
                     f"{cfg.train_config.dataloader.batch_size}, global batch "
                     f"{cfg.train_config.dataloader.batch_size * world_size}")
        else:
            device = torch.device(cfg.train_config.training_params.device)

        if cfg.train_config.training_params.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path is not None and latest_ckpt_path.is_file():
                log.info(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)
                if getattr(cfg.train_config.training_params, "reset_epoch_on_resume", False):
                    log.info(f"Resetting epoch counter from {self.epoch} to 0")
                    self.epoch = 0
                    self.global_step = 0

        dataset = hydra.utils.instantiate(cfg.dataset_config)
        train_dataloader = _create_dataloader(dataset, **cfg.train_config.dataloader,
                                             rank=rank, world_size=world_size)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = _create_dataloader(val_dataset, **cfg.train_config.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.train_config.training_params.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Optional dataset -> policy hand-offs (no-ops when the policy lacks the hook).
        for attr, setter in (
            ("bin_action_stds", "set_bin_action_stds"),
            ("bin_mean_actions", "set_bin_mean_actions"),
        ):
            if hasattr(dataset, attr) and hasattr(self.model, setter):
                getattr(self.model, setter)(getattr(dataset, attr))
                if cfg.train_config.training_params.use_ema:
                    getattr(self.ema_model, setter)(getattr(dataset, attr))
        if hasattr(dataset, "cross_demo_var") and hasattr(self.model, "set_gt_action_vars"):
            self.model.set_gt_action_vars(dataset.cross_demo_progress, dataset.cross_demo_var)
            if cfg.train_config.training_params.use_ema:
                self.ema_model.set_gt_action_vars(dataset.cross_demo_progress, dataset.cross_demo_var)

        lr_scheduler = get_scheduler(
            cfg.train_config.training_params.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.train_config.training_params.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.train_config.training_params.num_epochs
            ) // cfg.train_config.training_params.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema = None
        if cfg.train_config.training_params.use_ema:
            ema = hydra.utils.instantiate(cfg.train_config.ema, model=self.ema_model)

        wandb_run = None
        wandb_disabled = os.environ.get("WANDB_MODE", "").lower() == "disabled"
        if is_main and cfg.logging.mode == "online" and not wandb_disabled:
            logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            if logging_cfg.get("tags"):
                logging_cfg["tags"] = [t[:64] for t in logging_cfg["tags"]]
            try:
                import wandb
                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **logging_cfg,
                )
                wandb.config.update({"output_dir": self.output_dir})
            except Exception as e:
                log.warning(f"wandb init failed: {e}. Continuing without wandb.")
                wandb_run = None

        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Kept as a local, not an attribute: save_checkpoint() serialises every attribute with a
        # state_dict, and the wrapper would duplicate the whole model under "module.policy.*".
        train_module = None
        if distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            # ModuleAttrMixin registers an EMPTY `_dummy_variable` nn.Parameter on the policy and on
            # the obs encoder (device / dtype probes). They can never receive a gradient, which DDP
            # reports as "parameters that were not used in producing loss" -- freeze them. Any real
            # parameter a policy leaves unused needs DDP_FIND_UNUSED=1 instead.
            frozen = [n for n, p in self.model.named_parameters() if p.numel() == 0 and p.requires_grad]
            for n, p in self.model.named_parameters():
                if p.numel() == 0:
                    p.requires_grad_(False)
            if frozen:
                log.info(f"DDP: froze {len(frozen)} empty probe parameter(s): {frozen}")
            train_module = DDP(_LossModule(self.model),
                               device_ids=[device.index] if device.type == "cuda" else None,
                               find_unused_parameters=os.environ.get("DDP_FIND_UNUSED", "0") == "1")

        train_sampling_batch = None

        if cfg.train_config.training_params.debug:
            cfg.train_config.training_params.num_epochs = 2
            cfg.train_config.training_params.max_train_steps = 3
            cfg.train_config.training_params.max_val_steps = 3
            cfg.train_config.training_params.rollout_every = 1
            cfg.train_config.training_params.checkpoint_every = 1
            cfg.train_config.training_params.val_every = 1
            cfg.train_config.training_params.sample_every = 1

        _csv_step_rows = []
        # only rank 0 writes logs; the other ranks get a scratch file so nothing collides
        log_path = os.path.join(self.output_dir, "logs.json.txt" if is_main else f"logs.rank{rank}.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.train_config.training_params.num_epochs):
                step_log = dict()
                epoch_t0 = time.time()

                if cfg.train_config.training_params.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.train_config.training_params.tqdm_interval_sec,
                    disable=not is_main,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = train_module(batch) if distributed else self.model.compute_loss(batch)
                        loss = raw_loss / cfg.train_config.training_params.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.train_config.training_params.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.train_config.training_params.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        if hasattr(self.model, "_last_metrics"):
                            step_log.update(self.model._last_metrics)
                            _csv_step_rows.append({
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                                **self.model._last_metrics,
                            })

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            if wandb_run is not None:
                                wandb_run.log(step_log, step=self.global_step)
                            if is_main:
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.train_config.training_params.max_train_steps is not None) and \
                                batch_idx >= cfg.train_config.training_params.max_train_steps - 1:
                            break

                step_log["train_loss"] = float(np.mean(train_losses))
                step_log["epoch_time_s"] = time.time() - epoch_t0
                step_log["train_steps_in_epoch"] = len(train_losses)
                if device.type == "cuda":
                    step_log["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2 ** 30

                policy = self.ema_model if cfg.train_config.training_params.use_ema else self.model
                policy.eval()

                # validation, the sampling check and checkpoints: rank 0 only (no collectives in
                # them, so the other ranks simply wait at the next DDP step)
                if is_main and (self.epoch % cfg.train_config.training_params.val_every) == 0:
                    with torch.no_grad():
                        val_losses = []
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.train_config.training_params.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = val_dataset.postprocess(batch, device)
                                val_losses.append(self.model.compute_loss(batch))
                                if (cfg.train_config.training_params.max_val_steps is not None) and \
                                        batch_idx >= cfg.train_config.training_params.max_val_steps - 1:
                                    break
                        if val_losses:
                            step_log["val_loss"] = torch.mean(torch.tensor(val_losses)).item()

                if is_main and (self.epoch % cfg.train_config.training_params.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]
                        # Policies whose flow starts from the history read that
                        # signal from obs_dict at inference. There is no rollout
                        # here, so feed the batch's own copy.
                        hist_src = getattr(policy, "history_source", None)
                        if hist_src == "past_action":
                            obs_dict = dict(obs_dict)
                            obs_dict["past_action"] = gt_action[:, :policy.n_obs_steps, :]
                        elif hist_src == "measured_delta":
                            obs_dict = dict(obs_dict)
                            obs_dict["measured_delta"] = batch["measured_delta"]
                        rel = getattr(policy, "relative_dims", 0)
                        if rel:
                            # poses learned relative to the current one: anchor on the
                            # batch's current pose (row n_obs_steps - 1)
                            result = policy.predict_action(
                                obs_dict, a0=gt_action[:, policy.n_obs_steps - 1, :rel])
                        else:
                            result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        pred_len, gt_len = pred_action.shape[1], gt_action.shape[1]
                        if pred_len != gt_len:
                            start_idx = (gt_len - pred_len + 1) - 1
                            gt_action = gt_action[:, start_idx:start_idx + pred_len, :]
                        step_log["train_action_mse_error"] = torch.nn.functional.mse_loss(
                            pred_action, gt_action
                        ).item()

                ckpt_epochs = cfg.train_config.training_params.get("checkpoint_epochs")
                save_now = ((self.epoch + 1) in ckpt_epochs) if ckpt_epochs else \
                    ((self.epoch + 1) % cfg.train_config.training_params.checkpoint_every == 0)
                if is_main and (save_now or (self.epoch + 1 >= cfg.train_config.training_params.num_epochs)):
                    ckpt_path = os.path.join(
                        cfg.checkpoint.save_root_dir, "checkpoints", f"{self.epoch + 1}.ckpt"
                    )
                    self.save_checkpoint(ckpt_path)

                policy.train()
                if is_main:
                    json_logger.log(step_log)
                    log.info(f"epoch {self.epoch} done in {step_log['epoch_time_s']:.0f} s: "
                             f"train_loss {step_log['train_loss']:.4f}"
                             + (f", val_loss {step_log['val_loss']:.4f}" if "val_loss" in step_log else "")
                             + (f", gpu_mem {step_log['gpu_mem_gb']:.1f} GB" if "gpu_mem_gb" in step_log else ""))
                if wandb_run is not None:
                    wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1

                # Stop early WITHOUT shortening the LR schedule, so an early checkpoint
                # of a cut-short run is the same model a full run would have saved.
                stop_at = cfg.train_config.training_params.get("stop_at_epoch")
                if stop_at is not None and self.epoch >= stop_at:
                    log.info(f"stop_at_epoch={stop_at} reached; ending training early")
                    break

            if distributed:
                dist.barrier()
                dist.destroy_process_group()

            if _csv_step_rows:
                import csv
                csv_dir = os.path.join(cfg.checkpoint.save_root_dir, "csv_logs")
                os.makedirs(csv_dir, exist_ok=True)
                csv_path = os.path.join(csv_dir, "progress_tracking.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(_csv_step_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(_csv_step_rows)
                log.info(f"Saved progress tracking CSV: {csv_path} ({len(_csv_step_rows)} rows)")

    def run(self):
        if self.cfg.train_enable:
            self.train()
