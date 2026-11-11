# Adapted for ODE-POLICY public integration; see THIRD_PARTY_NOTICES.md.
"""Fixed-budget supervised SONIC ODE training, with exact resumable final state.

No validation selection, robot transport, or dataset preparation is performed
here. The caller supplies strictly current-frame conditions and measured states.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import time
import uuid

import numpy as np
import torch


_STATE_CONTAINER = "ode-g1-full-state-footer-v1"
_STATE_FOOTER_MAGIC = b"ODEG1S01"
_STATE_FOOTER = struct.Struct(">8sQ32s")


def _append_state_footer(path):
    """Seal all torch.save payload bytes in the same file, before its atomic rename."""
    path = Path(path)
    length = path.stat().st_size
    digest = bytes.fromhex(sha256(path))
    with path.open("ab") as stream:
        stream.write(_STATE_FOOTER.pack(_STATE_FOOTER_MAGIC, length, digest))
        stream.flush()
        os.fsync(stream.fileno())


def _verify_state_footer(path):
    """Return False only for legacy containers; never deserialize a bad sealed payload."""
    size = path.stat().st_size
    if size < _STATE_FOOTER.size:
        return False
    with path.open("rb") as stream:
        stream.seek(-_STATE_FOOTER.size, os.SEEK_END)
        magic, length, expected = _STATE_FOOTER.unpack(stream.read(_STATE_FOOTER.size))
        if magic != _STATE_FOOTER_MAGIC:
            return False
        if length <= 0 or length + _STATE_FOOTER.size != size:
            raise ValueError("Resume state footer payload length differs")
        stream.seek(0)
        digest = hashlib.sha256()
        remaining = length
        while remaining:
            block = stream.read(min(4 * 1024**2, remaining))
            if not block:
                raise ValueError("Resume state footer payload is truncated")
            digest.update(block)
            remaining -= len(block)
        if digest.digest() != expected:
            raise ValueError("Resume state footer payload SHA differs")
    return True


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ODETrainingOptions:
    epochs: int = 90
    batch_size: int = 112
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 17
    log_every_steps: int = 10
    checkpoint_every_epochs: int = 5
    checkpoint_milestone_epochs: tuple[int, ...] = (60, 70, 80, 90)
    device: str = "cuda"
    autocast_bfloat16: bool = True

    def __post_init__(self):
        for key in ("epochs", "batch_size", "log_every_steps", "checkpoint_every_epochs"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        milestones = self.checkpoint_milestone_epochs
        if (
            not isinstance(milestones, tuple)
            or any(type(epoch) is not int or epoch < 1 for epoch in milestones)
            or tuple(sorted(set(milestones))) != milestones
        ):
            raise ValueError("Milestone epochs must be a sorted unique tuple of positive integers")
        for key in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid learning-rate schedule")


def options_document(options):
    document = asdict(options)
    document["checkpoint_milestone_epochs"] = list(options.checkpoint_milestone_epochs)
    return document


def learning_rate(step, total, options):
    """Rate for a one-based optimizer update, reaching the floor at final update."""
    if not 1 <= step <= total:
        raise ValueError("Optimizer step is outside the fixed budget")
    warmup = min(total - 1, round(total * options.warmup_ratio))
    if warmup and step <= warmup:
        return options.learning_rate * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    scale = options.min_lr_ratio + (1 - options.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return options.learning_rate * scale


def example_identity(examples):
    payload = json.dumps([(int(ep), int(t)) for ep, t in examples], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def epoch_order(size, seed, epoch):
    """Independent deterministic permutation, so resume cannot change sample order."""
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(size)


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA training state requires CUDA for exact continuation")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _finite_weights(state):
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite checkpoint tensor: {name}")


def save_training_state(path, head, optimizer, progress, identity, options, *, logs_dir=None):
    """Write model+AdamW+RNG+cursor atomically; never call a partial run final."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _finite_weights(head.state_dict())
    for state in optimizer.state.values():
        _finite_weights({str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)})
    document = {
        "schema_version": 1,
        "container_format": _STATE_CONTAINER,
        "identity": identity,
        "options": options_document(options),
        "progress": progress,
        "head": head.state_dict(),
        "optimizer_parameter_names": [name for name, parameter in head.named_parameters() if parameter.requires_grad],
        "optimizer": optimizer.state_dict(),
        "rng": _rng_state(),
        "recorded_log_sha256": {
            name: sha256(Path(logs_dir or path.parent) / name)
            if (Path(logs_dir or path.parent) / name).exists()
            else None
            for name in ("metrics.jsonl", "epochs.jsonl")
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(document, temporary)
    _append_state_footer(temporary)
    os.replace(temporary, path)
    return {"path": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size}


def save_head_weights(path, head, *, final=False, policy_binding_sha256=None):
    """Save finite inference tensors atomically, without changing the training RNG."""
    from safetensors.torch import save_file

    path = Path(path)
    state = {key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()}
    _finite_weights(state)
    temporary = path.with_name(path.name + ".tmp")
    head_id = uuid.uuid4().hex
    metadata = {"method": "sonic_ode", "selection": "final" if final else "epoch_checkpoint", "head_id": head_id}
    if policy_binding_sha256 is not None:
        metadata["policy_binding_sha256"] = policy_binding_sha256
    save_file(
        state,
        str(temporary),
        metadata=metadata,
    )
    os.replace(temporary, path)
    return {"path": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size, "head_id": head_id}


def save_epoch_checkpoint(output, head, optimizer, progress, identity, options):
    """Publish a complete epoch milestone directory only after both payloads and SHA pass.

    This is a full continuation point, not best-checkpoint selection. The exact
    model, name-bound AdamW, Python/NumPy/Torch/CUDA RNG and next-epoch cursor
    are serialized by ``save_training_state`` inside the original allocation.
    """
    output = Path(output)
    epoch = progress["epoch_index"]
    if progress["next_batch"] != 0 or progress["epoch_samples"] != 0:
        raise ValueError("A milestone must be saved at an actual completed epoch boundary")
    destination = output / "checkpoints" / f"epoch_{epoch:03d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("An immutable milestone already exists; never replace its payloads")
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    head_receipt = save_head_weights(
        staging / "head.safetensors",
        head,
        final=epoch == options.epochs,
        policy_binding_sha256=identity.get("policy_binding_sha256"),
    )
    state_receipt = save_training_state(
        staging / "training_state.pt", head, optimizer, progress, identity, options, logs_dir=output
    )
    head_receipt["path"] = str(destination.relative_to(output) / "head.safetensors")
    state_receipt["path"] = str(destination.relative_to(output) / "training_state.pt")
    receipt = {
        "schema_version": 1,
        "status": "actual_epoch_milestone_saved",
        "completed_epoch": epoch,
        "optimizer_steps": progress["optimizer_steps"],
        "sample_exposures": progress["sample_exposures"],
        "progress": dict(progress),
        "identity": identity,
        "options": options_document(options),
        "optimizer_parameter_names": [name for name, parameter in head.named_parameters() if parameter.requires_grad],
        "head": head_receipt,
        "training_state": state_receipt,
        "finite_model_and_optimizer_tensors": True,
        "checkpoint_selection": "fixed_requested_epoch_no_validation_or_best",
    }
    atomic_json(staging / "RECEIPT.json", receipt)
    (staging / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(staging / name)}  {name}\n" for name in ("head.safetensors", "training_state.pt", "RECEIPT.json")
        )
    )
    os.replace(staging, destination)
    return receipt


def _atomic_hardlink(source, destination):
    """Expose an already verified payload at another owned path without copying bytes."""
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def restore_training_state(path, head, optimizer, identity, options, *, logs_dir=None):
    # This is an internally generated training state, not an arbitrary remote pickle.
    path = Path(path)
    if path.name not in ("resume_latest.pt", "training_state.pt", "final_training_state.pt"):
        raise ValueError("Resume requires a generated full-state filename and its SHA receipt")
    sealed = _verify_state_footer(path)
    # A valid new latest state is one atomic file. The JSON is a derived report:
    # it may still describe the previous state if a process stopped after rename.
    if not (sealed and path.name == "resume_latest.pt"):
        if path.name == "resume_latest.pt":
            receipt = json.loads(path.with_suffix(".json").read_text())
        elif path.name == "training_state.pt":
            receipt = json.loads((path.parent / "RECEIPT.json").read_text())["training_state"]
        else:
            receipt = json.loads((path.parent / "training_summary.json").read_text())["resumable_training_state"]
        if path.stat().st_size != receipt["size_bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError("Resume full-state bytes differ from their SHA receipt")
    document = torch.load(path, map_location="cpu", weights_only=False)
    container = document.get("container_format")
    if (sealed and container != _STATE_CONTAINER) or (not sealed and container is not None):
        raise ValueError("Resume state footer is missing, damaged or incompatible")
    if document["schema_version"] != 1 or document["identity"] != identity:
        raise ValueError("Resume dataset/model/normalization/config identity differs")
    if document["options"] != options_document(options):
        raise ValueError("Resume must keep the fixed training budget and schedule")
    names = [name for name, parameter in head.named_parameters() if parameter.requires_grad]
    if document["optimizer_parameter_names"] != names:
        raise ValueError("Resume optimizer parameter name/order binding differs")
    for name, expected in document["recorded_log_sha256"].items():
        log = Path(logs_dir or path.parent) / name
        actual = sha256(log) if log.exists() else None
        if expected != actual:
            raise ValueError("Recorded metrics/epoch prefix differs: automatic rollback/retraining is forbidden")
    _finite_weights(document["head"])
    for state in document["optimizer"]["state"].values():
        _finite_weights({str(k): v for k, v in state.items() if isinstance(v, torch.Tensor)})
    head.load_state_dict(document["head"], strict=True)
    optimizer.load_state_dict(document["optimizer"])
    _restore_rng(document["rng"])
    return document["progress"]


def verify_resume_logs(output, progress):
    """Reject rollback over any recorded updates; never silently rerun or erase them."""
    metrics_path = Path(output) / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    cursor = progress["optimizer_steps"]
    if [row["optimizer_step"] for row in rows] != list(range(1, cursor + 1)):
        raise ValueError(
            "Resume checkpoint does not match the complete recorded update prefix. "
            "Preserve logs and inspect recovery; automatic rollback/retraining is forbidden."
        )
    if sum(row["samples"] for row in rows) != progress["sample_exposures"]:
        raise ValueError("Recorded sample exposures differ from the resume checkpoint")
    epochs_path = Path(output) / "epochs.jsonl"
    epochs = (
        [json.loads(line) for line in epochs_path.read_text().splitlines() if line.strip()]
        if epochs_path.exists()
        else []
    )
    if [row["completed_epoch"] for row in epochs] != list(range(1, progress["epoch_index"] + 1)):
        raise ValueError("Recorded completed epochs differ from the resume checkpoint")


def stack_windows(windows, device):
    return {
        key: torch.from_numpy(np.stack([window[key] for window in windows])).to(device)
        for key in ("state", "r0", "actions", "action_mask")
    }


def _metrics(result):
    metrics = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = float(value.detach())
        elif not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite training metric {key}")
        metrics[key] = value
    if "loss" not in metrics:
        raise ValueError("ODE head did not supply a scalar loss")
    return metrics


def train_ode(
    head,
    dataset,
    condition_provider,
    output,
    options,
    *,
    identity,
    resume=None,
    stop_requested=None,
):
    """Visit every frame once per epoch. Stop requests save after a complete update.

    ``condition_provider`` receives only (episode, current_frame) pairs. It must
    return current frozen VLM features and masks, never future RGB or actions.
    A stopped run can resume the exact next minibatch, including RNG and AdamW.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    examples = list(dataset.examples)
    if not examples:
        raise ValueError("Empty training set")
    horizon = head.config.horizon
    if type(horizon) is not int or not 1 <= horizon <= 16:
        raise ValueError("The H16 trainer requires an explicit head horizon in 1..16")
    identity = {**identity, "example_sha256": example_identity(examples), "action_horizon": horizon}
    device = torch.device(options.device)
    head.to(device=device, dtype=torch.float32)
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable ODE parameters")
    optimizer = torch.optim.AdamW(parameters, lr=options.learning_rate, weight_decay=options.weight_decay)
    steps_per_epoch = math.ceil(len(examples) / options.batch_size)
    total = steps_per_epoch * options.epochs
    progress = {
        "epoch_index": 0,
        "next_batch": 0,
        "optimizer_steps": 0,
        "sample_exposures": 0,
        "epoch_metric_sums": {},
        "epoch_samples": 0,
    }
    if resume is None:
        if (output / "metrics.jsonl").exists() or (output / "final_head.safetensors").exists():
            raise FileExistsError("Training output exists; explicit verified resume is required")
        random.seed(options.seed)
        np.random.seed(options.seed)
        torch.manual_seed(options.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(options.seed)
    else:
        progress = restore_training_state(resume, head, optimizer, identity, options, logs_dir=output)
        verify_resume_logs(output, progress)
    expected_steps = progress["epoch_index"] * steps_per_epoch + progress["next_batch"]
    expected_samples = progress["epoch_index"] * len(examples) + min(
        progress["next_batch"] * options.batch_size, len(examples)
    )
    if progress["optimizer_steps"] != expected_steps or progress["sample_exposures"] != expected_samples:
        raise ValueError("Resume cursor and actual optimizer/sample counters disagree")
    plan = {
        "options": options_document(options),
        "identity": identity,
        "action_horizon": horizon,
        "checkpoint_milestone_epochs": [
            epoch for epoch in options.checkpoint_milestone_epochs if epoch <= options.epochs
        ],
        "examples_per_epoch": len(examples),
        "steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": total,
        "planned_sample_exposures": len(examples) * options.epochs,
        "validation_performed": False,
        "checkpoint_selection": "actual_final_epoch",
        "loss_scale": "plain_masked_normalized_MSE_no_flow_divisor",
        "epoch_metric_aggregation": "sample_weighted_minibatch_means_valid_value_counts_summed",
        "r0_training": "strict_previous_recorded_body_or_normalized_zero_BOS",
        "hand_target": "same_frame_measured_left6_right6_continuous_states",
    }
    previous_plan = output / "training_plan.json"
    if previous_plan.exists() and json.loads(previous_plan.read_text()) != plan:
        raise ValueError("Existing training plan differs")
    atomic_json(previous_plan, plan)
    started = time.monotonic()
    head.train()
    for epoch in range(progress["epoch_index"], options.epochs):
        order = epoch_order(len(examples), options.seed, epoch)
        initial_batch = progress["next_batch"] if epoch == progress["epoch_index"] else 0
        for batch_index in range(initial_batch, steps_per_epoch):
            chosen = order[batch_index * options.batch_size : (batch_index + 1) * options.batch_size]
            keys = [examples[int(index)] for index in chosen]
            windows = [dataset.sample(*key, horizon=horizon) for key in keys]
            values = stack_windows(windows, device)
            condition = condition_provider(keys)
            step = progress["optimizer_steps"] + 1
            lr = learning_rate(step, total, options)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=options.autocast_bfloat16 and device.type == "cuda",
            ):
                result = head.forward_cached(
                    condition,
                    values["state"],
                    values["r0"],
                    values["actions"],
                    values["action_mask"],
                )
            metrics = _metrics(result)
            result["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, options.max_grad_norm, error_if_nonfinite=True)
            if step == 1 and device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                free, total_memory = torch.cuda.mem_get_info(device)
                parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
                # AdamW first creates two moments; foreach can transiently
                # need another parameter-sized buffer. No optimizer yet.
                required = 3 * parameter_bytes + 2 * 1024**3
                atomic_json(
                    output / "first_backward_gpu_memory.json",
                    {
                        "finite_loss": True,
                        "finite_grad_norm": float(grad_norm),
                        "trainable_parameter_bytes": parameter_bytes,
                        "cuda_free_after_backward_bytes": free,
                        "cuda_total_bytes": total_memory,
                        "reserved_for_AdamW_moments_and_transient_bytes": required,
                        "optimizer_updates_before_this_gate": 0,
                        "passed": free >= required,
                    },
                )
                if free < required:
                    raise MemoryError("Insufficient actual GPU memory for AdamW after the first finite backward")
            optimizer.step()
            progress.update(
                optimizer_steps=step,
                sample_exposures=progress["sample_exposures"] + len(keys),
                next_batch=batch_index + 1,
                epoch_index=epoch,
                epoch_samples=progress["epoch_samples"] + len(keys),
            )
            for key, value in metrics.items():
                progress["epoch_metric_sums"][key] = progress["epoch_metric_sums"].get(key, 0.0) + value * (
                    1 if key.startswith("valid_") else len(keys)
                )
            row = {
                "event": "optimizer_update",
                "optimizer_step": step,
                "epoch_index": epoch,
                "batch_index": batch_index,
                "sample_exposures": progress["sample_exposures"],
                "samples": len(keys),
                "learning_rate": lr,
                "grad_norm": float(grad_norm),
                "finite": True,
                "elapsed_this_process_seconds": time.monotonic() - started,
                **metrics,
            }
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            atomic_json(output / "progress.json", {**progress, "status": "training"})
            if step == 1 or step % options.log_every_steps == 0 or step == total:
                print(json.dumps(row, allow_nan=False), flush=True)
            if stop_requested is not None and stop_requested() and step < total:
                receipt = save_training_state(output / "resume_latest.pt", head, optimizer, progress, identity, options)
                atomic_json(output / "resume_latest.json", {**receipt, "progress": progress})
                atomic_json(output / "progress.json", {**progress, "status": "paused_with_exact_resume"})
                return {
                    "status": "paused_with_exact_resume",
                    "progress": progress,
                    "checkpoint": receipt,
                }
        if progress["epoch_samples"] != len(examples):
            raise RuntimeError("A completed epoch did not visit every frame exactly once")
        epoch_row = {
            "completed_epoch": epoch + 1,
            "optimizer_steps": progress["optimizer_steps"],
            "sample_exposures": progress["sample_exposures"],
            "samples": progress["epoch_samples"],
            **{
                key: value if key.startswith("valid_") else value / progress["epoch_samples"]
                for key, value in progress["epoch_metric_sums"].items()
            },
        }
        with (output / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(epoch_row, allow_nan=False) + "\n")
        progress.update(epoch_index=epoch + 1, next_batch=0, epoch_metric_sums={}, epoch_samples=0)
        if epoch + 1 in options.checkpoint_milestone_epochs:
            milestone = save_epoch_checkpoint(output, head, optimizer, progress, identity, options)
            # Keep the latest exact state usable after an interruption between
            # milestone publication and final inference-file publication.
            _atomic_hardlink(output / milestone["training_state"]["path"], output / "resume_latest.pt")
            receipt = {**milestone["training_state"], "path": "resume_latest.pt"}
            atomic_json(output / "resume_latest.json", {**receipt, "progress": progress})
        elif (epoch + 1) % options.checkpoint_every_epochs == 0 and epoch + 1 < options.epochs:
            receipt = save_training_state(output / "resume_latest.pt", head, optimizer, progress, identity, options)
            atomic_json(output / "resume_latest.json", {**receipt, "progress": progress})
    if progress["optimizer_steps"] != total or progress["sample_exposures"] != len(examples) * options.epochs:
        raise RuntimeError("Fixed training budget was not completed")
    if options.epochs in options.checkpoint_milestone_epochs:
        final_milestone = json.loads(
            (output / "checkpoints" / f"epoch_{options.epochs:03d}" / "RECEIPT.json").read_text()
        )
        for key, filename in (
            ("head", "final_head.safetensors"),
            ("training_state", "final_training_state.pt"),
        ):
            source = final_milestone[key]
            source_path = output / source["path"]
            if sha256(source_path) != source["sha256"]:
                raise ValueError("Final milestone payload changed before final publication")
            _atomic_hardlink(source_path, output / filename)
        weights_receipt = {
            **final_milestone["head"],
            "path": "final_head.safetensors",
        }
        full_receipt = {
            **final_milestone["training_state"],
            "path": "final_training_state.pt",
        }
    else:
        weights_receipt = save_head_weights(
            output / "final_head.safetensors",
            head,
            final=True,
            policy_binding_sha256=identity.get("policy_binding_sha256"),
        )
        full_receipt = save_training_state(
            output / "final_training_state.pt", head, optimizer, progress, identity, options
        )
    milestones = [
        json.loads((output / "checkpoints" / f"epoch_{epoch:03d}" / "RECEIPT.json").read_text())
        for epoch in options.checkpoint_milestone_epochs
        if epoch <= options.epochs
    ]
    summary = {
        "status": "fixed_budget_final_saved",
        "completed_epochs": options.epochs,
        "action_horizon": horizon,
        "milestone_checkpoints": milestones,
        "optimizer_steps": total,
        "sample_exposures": progress["sample_exposures"],
        "final_weights": weights_receipt,
        "resumable_training_state": full_receipt,
        "finite_weights": True,
        "validation_performed": False,
        "task_success_tested": False,
        "identity": identity,
    }
    atomic_json(output / "training_summary.json", summary)
    atomic_json(output / "progress.json", {**progress, "status": summary["status"]})
    return summary
