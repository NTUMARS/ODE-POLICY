"""Offline planning; target execution belongs to a live-state reference tracker."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .config import resolve_user_paths, validate_config
from .sidecar import load_metadata, sha256_file
from .devices import device_and_dtype
from .checkpoint import make_binding, read_weights, strict_load_checkpoint

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


class OfflinePlanner:
    def __init__(self, cfg, checkpoint, *, online_text=False):
        self.cfg = resolve_user_paths(cfg)
        validate_config(cfg, runtime=True)
        import torch
        from hydra.utils import instantiate
        from .data.normalizer import load_dataset_stats_from_json

        self.device, self.dtype = device_and_dtype(cfg)
        self.metadata = load_metadata(Path(str(cfg.paths.sidecar_dir)) / "metadata.json")
        if sha256_file(cfg.paths.normalization_stats) != self.metadata["normalization_stats_sha256"]:
            raise ValueError("Inference normalization stats do not match the prepared ODE metadata")
        if online_text and not cfg.paths.qwen_model:
            raise ValueError("--online-text requires paths.qwen_model")
        if not online_text and not cfg.paths.text_cache_dir:
            raise ValueError("Supply paths.text_cache_dir or explicitly request --online-text")
        binding = make_binding(self.metadata, cfg.model)
        payload = read_weights(checkpoint, binding)
        cfg.model.load_text_encoder = bool(online_text)
        # A strict full MoT checkpoint replaces the complete action expert;
        # the separate pretrained ActionDiT initializer is only needed to train.
        cfg.model.action_dit_pretrained_path = None
        self.model = instantiate(cfg.model, model_dtype=self.dtype, device=self.device)
        self.checkpoint_step = strict_load_checkpoint(self.model, checkpoint, payload=payload)
        del payload
        self.model.eval()
        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(cfg.paths.normalization_stats)))
        self.online_text = online_text
        self.torch = torch

    def _text_context(self, prompt):
        if self.online_text:
            return None, None
        h = hashlib.sha256(prompt.encode()).hexdigest()
        length = int(self.cfg.model.qwen_context_len)
        path = Path(str(self.cfg.paths.text_cache_dir)) / f"{h}.qwen3_flux2_len{length}.pt"
        payload = self.torch.load(path, map_location="cpu", weights_only=True)
        context, mask = payload["text_hidden_states"], payload["text_attention_mask"].bool()
        if context.ndim != 2 or context.shape[0] != length or mask.shape != (length,):
            raise ValueError("Qwen context/mask dimensions do not match the configured context length")
        if not self.torch.isfinite(context).all():
            raise ValueError("Qwen context contains nonfinite values")
        return context[None].to(self.device), mask[None].to(self.device)

    def plan(self, primary_rgb, wrist_rgb, raw_state, instruction, reference=None):
        torch = self.torch
        raw_state = np.asarray(raw_state, dtype=np.float32)
        if raw_state.shape != (8,) or not np.isfinite(raw_state).all():
            raise ValueError("raw_proprio must contain eight finite state values")
        if not str(instruction).strip():
            raise ValueError("A nonempty task instruction is required")
        images = {}
        for image, meta in zip((primary_rgb, wrist_rgb), self.cfg.data.train.shape_meta.images):
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("Images must be uint8 HWC RGB arrays")
            images[str(meta.key)] = torch.from_numpy(image.copy()).permute(2, 0, 1)[None]
        if len(images) != 2 or self.cfg.data.train.concat_multi_camera != "horizontal":
            raise ValueError("Use two horizontally concatenated cameras")
        pixels = self.processor.process_images(images, expected_steps=1)
        image = self.processor.pack_video(pixels, self.cfg.data.train.video_size)[:, 0][None]
        image = image.to(device=self.device, dtype=self.dtype)
        state_batch = {"state": {"default": torch.from_numpy(raw_state)[None]}}
        state_batch = self.processor.action_state_transform(state_batch)
        proprio = self.processor.normalizer.forward(state_batch)["state"]["default"]
        prompt = PROMPT_TEMPLATE.format(task=instruction)
        context, mask = self._text_context(prompt)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        begin = time.perf_counter()
        output = self.model.infer_action(prompt=prompt, input_image=image,
            action_horizon=int(self.cfg.model.ode.horizon), proprio=proprio,
            raw_proprio=raw_state, ode_reference=reference,
            context=context, context_mask=mask)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - begin) * 1000
        targets = output["ode_targets"][0].detach().float().cpu().numpy()
        grip = output["ode_gripper"][0].detach().float().cpu().numpy()
        if not np.isfinite(targets).all() or not np.isin(grip, [0, 1]).all():
            raise ValueError("Planner produced nonfinite targets or nonbinary gripper values")
        return {"ode_targets": targets, "ode_gripper": grip,
                "initial_reference": output["ode_initial"][0].detach().float().cpu().numpy(),
                "model_ms": elapsed}


def infer_file(cfg, checkpoint, observation, output, instruction, online_text=False):
    from .execution import ReferenceExecutor
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with np.load(observation, allow_pickle=False) as data:
        primary, wrist, raw = data["agentview_rgb"], data["wrist_rgb"], data["raw_proprio"]
        reference = data["ode_reference"] if "ode_reference" in data else None
    planner = OfflinePlanner(cfg, checkpoint, online_text=online_text)
    result = planner.plan(primary, wrist, raw, instruction, reference)
    executor = ReferenceExecutor(planner.metadata["geometry"])
    result["first_live_state_command"] = executor.command(result["ode_targets"][0], raw, result["ode_gripper"][0])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Passing an open stream avoids silently appending .npz to a user path.
    with output.open("xb") as stream:
        np.savez(stream, **result)
    return {"output": str(output), "horizon": len(result["ode_targets"]), "model_ms": result["model_ms"]}
