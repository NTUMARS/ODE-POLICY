"""ImageWAM with a one-query action context reader and Euler reference rollout."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .backbones.base import ImageWAM
from .head import ActionODEHead, reference_to_command, state_to_reference
from .checkpoint import make_binding, save_weights, strict_load_checkpoint

logger = logging.getLogger(__name__)


class ODEImageWAM(ImageWAM):
    def save_checkpoint(self, path, optimizer=None, step=None):
        save_weights(self, path, step, optimizer)

    def load_checkpoint(self, path, optimizer=None):
        return strict_load_checkpoint(self, path, optimizer=optimizer)

    def _ode_tensor(self, key, like):
        # Keep geometry constants at FP32 precision through DeepSpeed bf16 casts.
        return torch.tensor(self.ode_metadata[key], device=like.device, dtype=torch.float32)

    def _normalize_reference(self, r):
        return (r.float() - self._ode_tensor("reference_center", r)) / self._ode_tensor("reference_half_range", r)

    def _denormalize_reference(self, r):
        return r.float() * self._ode_tensor("reference_half_range", r) + self._ode_tensor("reference_center", r)

    def _ode_query_pre(self, batch_size):
        a = self.action_expert
        tokens = a.ode_query.expand(batch_size, -1, -1)
        vec = a.ode_context_vec.expand(batch_size, -1)
        double = a.double_stream_modulation_img(vec)
        single, _ = a.single_stream_modulation(vec)
        ids = a.build_action_ids(batch_size, 1, device=tokens.device, dtype=tokens.dtype)
        return {"tokens": tokens, "ids": ids,
                "t_mod": {"vec": vec, "double_img": double, "single": single}}

    def _ode_context(self, ref_tokens, ref_ids, text, text_mask):
        b = ref_tokens.shape[0]
        empty = ref_tokens[:, :0]
        pre = self.video_expert.pre_dit(
            x=empty, timestep=ref_tokens.new_zeros(b), context=text,
            context_mask=text_mask, ref_image_hidden_states=ref_tokens,
            target_img_ids=ref_ids[:, :0], ref_img_ids=ref_ids)
        prefix_mask = self._build_mot_attention_mask_flux2(
            batch_size=b, txt_len=int(pre["txt_len"]), target_len=0,
            cond_len=int(pre["cond_len"]), action_len=0, device=ref_tokens.device,
            text_attention_mask=pre["text_mask"])

        def prefill():
            return self.mot.prefill_flux2_video_cache(
                video_tokens=pre["tokens"], video_freqs=pre["freqs"],
                video_t_mod=pre["t_mod"], attention_mask=prefix_mask)

        # Prefix gradients are retained: this experiment does not freeze the visual
        # backbone or introduce knowledge insulation into vanilla ImageWAM.
        cache = checkpoint(prefill, use_reentrant=False) if torch.is_grad_enabled() else prefill()
        query = self._ode_query_pre(b)
        mask = self._build_mot_attention_mask_flux2(
            batch_size=b, txt_len=int(pre["txt_len"]), target_len=0,
            cond_len=int(pre["cond_len"]), action_len=1, device=ref_tokens.device,
            text_attention_mask=pre["text_mask"])
        tokens = self.mot.forward_action_with_video_cache(
            action_tokens=query["tokens"], action_freqs=None,
            action_t_mod=query["t_mod"], action_context_payload={"ids": query["ids"]},
            video_kv_cache=cache, attention_mask=mask,
            video_seq_len=int(pre["txt_len"]) + int(pre["cond_len"]))
        return tokens[:, 0]

    def _ode_loss(self, sample, inputs):
        # No proprio token in c. The unchanged image FM objective still uses its
        # original proprio token; continuous s enters the action vector field only.
        text, text_mask = self._encode_flux2_text(sample)
        feature = self._ode_context(inputs["ref_image_latents"], inputs["ref_img_ids"], text, text_mask)
        r0 = self._normalize_reference(sample["ode_start"].to(self.device))
        target = self._normalize_reference(sample["ode_target"].to(self.device))
        s = sample["proprio"][:, 0].to(self.device).float()
        b, horizon, _ = target.shape
        if self.ode_perturb_prob:
            std = r0.new_tensor([self.ode_perturb_position_std] * 3 + [self.ode_perturb_rotation_std] * 3) / self._ode_tensor("reference_half_range", r0)
            on = (torch.rand(b, 1, device=r0.device) < self.ode_perturb_prob).float()
            delta = torch.randn_like(r0) * std * on
            r0 = r0 + delta
            decay = self.ode_perturb_decay ** torch.arange(1, horizon + 1, device=r0.device)
            target = target + delta[:, None] * decay[None, :, None]
        traj, grip = self.action_expert.ode_head(feature, s, r0, horizon)
        valid = (~sample["action_is_pad"].to(self.device)).float()
        denom = valid.sum().clamp_min(1)
        pose_loss = (((traj - target).square().mean(-1) * valid).sum() / denom
                     / self.action_expert.ode_head.vel_scale ** 2)
        grip_target = sample["ode_grip_target"].to(self.device).float()
        grip_loss = (F.binary_cross_entropy_with_logits(grip, grip_target, reduction="none") * valid).sum() / denom
        loss = self.loss_lambda_action * (pose_loss + self.ode_grip_weight * grip_loss)
        return loss, {"loss_action": float(loss.detach()), "loss_ode_path": float(pose_loss.detach()),
                      "loss_ode_grip": float(grip_loss.detach()),
                      "ode_mean_step": float((traj[:, 1:] - traj[:, :-1]).abs().mean().detach())}

    def iter_training_losses(self, sample, tiled=False):
        inputs = self.build_inputs_flux2(sample, tiled=tiled)
        # Preserve vanilla future-image FM, replacing only the action objective.
        yield super()._training_loss_flux2(sample, tiled=tiled, inputs=inputs)
        yield self._ode_loss(sample, inputs)

    def training_objective_count(self):
        return 2

    def training_loss(self, sample, tiled=False):
        terms = list(self.iter_training_losses(sample, tiled=tiled))
        metrics = {}
        for _, values in terms:
            for k, v in values.items():
                metrics[k] = metrics.get(k, 0) + v
        return sum(x[0] for x in terms), metrics

    @torch.no_grad()
    def infer_action(self, prompt, input_image, action_horizon, proprio=None,
                     context=None, context_mask=None, raw_proprio=None,
                     ode_reference=None, **kwargs):
        if raw_proprio is None:
            raise ValueError("ODE inference requires raw_proprio for the closed-loop reference tracker")
        if kwargs.get("action_init") is not None:
            raise ValueError("ODE uses reference carry, not an A2A action initializer")
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image[None]
        ref, ids = self._encode_flux2_image_tokens(input_image.to(self.device, dtype=self.torch_dtype), time_value=10.0)
        text, text_mask = self._prepare_flux2_infer_text(None if context is not None else prompt, context, context_mask)
        feature = self._ode_context(ref, ids, text, text_mask)
        raw = torch.as_tensor(raw_proprio, device=self.device, dtype=torch.float32).reshape(-1, 8)
        s = torch.as_tensor(proprio, device=self.device, dtype=torch.float32).reshape(-1, 8)
        initial = state_to_reference(raw, self.ode_metadata["geometry"]) if ode_reference is None else torch.as_tensor(ode_reference, device=self.device, dtype=torch.float32).reshape(-1, 6)
        traj, grip_logits = self.action_expert.ode_head(feature, s, self._normalize_reference(initial), action_horizon)
        targets = self._denormalize_reference(traj)
        grip = (grip_logits >= 0).float()
        # This normalized preview supports existing logging. The evaluator must
        # recompute each command against the live observation at execution time.
        cmd = reference_to_command(targets, raw[:, None].expand(-1, action_horizon, -1), self.ode_metadata["geometry"])
        raw_action = torch.cat((cmd, grip[..., None]), -1)
        amin = self._ode_tensor("action_min", raw_action)
        amax = self._ode_tensor("action_max", raw_action)
        normalized = 2 * (raw_action - amin) / (amax - amin) - 1
        return {"action": normalized, "ode_targets": targets, "ode_gripper": grip,
                "ode_initial": initial, "num_inference_steps": action_horizon}

    @torch.no_grad()
    def ode_env_command(self, reference, raw_state, grip):
        r = torch.as_tensor(reference, dtype=torch.float64).reshape(1, 6)
        s = torch.as_tensor(raw_state, dtype=torch.float64).reshape(1, 8)
        cmd = reference_to_command(r, s, self.ode_metadata["geometry"])[0].clamp(-1, 1)
        # Dataset gripper is 0=closed, 1=open; LIBERO expects +1=closed, -1=open.
        return torch.cat((cmd, cmd.new_tensor([1 - 2 * float(grip)]))).float().tolist()


def create_imagewam_ode(ode, **kwargs):
    from omegaconf import OmegaConf
    from .runtime import create_imagewam_flux2_klein
    cfg = OmegaConf.to_container(ode, resolve=True) if OmegaConf.is_config(ode) else dict(ode)
    for key in ("perturb_prob", "perturb_decay"):
        if not 0 <= float(cfg.get(key, 0.5 if key == "perturb_prob" else 0.7)) <= 1:
            raise ValueError(f"{key} must be between zero and one")
    for key in ("perturb_position_std", "perturb_rotation_std", "grip_weight"):
        if float(cfg.get(key, 0.01 if key == "perturb_position_std" else 0.03 if key == "perturb_rotation_std" else 0.1)) < 0:
            raise ValueError(f"{key} must be nonnegative")
    from .sidecar import load_metadata
    metadata_path = cfg.get("metadata_path") or str(Path(cfg["sidecar_dir"]) / "metadata.json")
    metadata = load_metadata(metadata_path)
    binding = make_binding(metadata, {**kwargs, "ode": cfg})
    model = create_imagewam_flux2_klein(**kwargs)
    if model.stack != "flux2" or model._flux2_action_uses_noise_context():
        raise ValueError("First ODE experiment requires vanilla FLUX2 prefix-only ImageWAM")
    before = sum(p.numel() for p in model.parameters())
    model.__class__ = ODEImageWAM
    model.ode_metadata = metadata
    model.ode_binding = binding
    model.ode_config = cfg
    model.ode_perturb_prob = float(cfg.get("perturb_prob", 0.5))
    model.ode_perturb_position_std = float(cfg.get("perturb_position_std", 0.01))
    model.ode_perturb_rotation_std = float(cfg.get("perturb_rotation_std", 0.03))
    model.ode_perturb_decay = float(cfg.get("perturb_decay", 0.7))
    model.ode_grip_weight = float(cfg.get("grip_weight", 0.1))
    a = model.action_expert
    from flux2.model import timestep_embedding
    with torch.no_grad():
        vec = a.time_in(timestep_embedding(torch.zeros(1, device=model.device), 256).to(model.torch_dtype)).detach()
    a.ode_query = nn.Parameter(torch.randn(1, 1, a.hidden_dim, device=model.device, dtype=model.torch_dtype) * 0.02)
    a.ode_context_vec = nn.Parameter(vec)
    # Retain all expert attention blocks and modulation weights, discard the unused
    # noisy-action input, flow-time embedder, and action velocity output head.
    a.action_encoder = nn.Identity()
    a.time_in = nn.Identity()
    a.head = nn.Identity()
    a.ode_head = ActionODEHead(feature_dim=a.hidden_dim, cond_dim=int(cfg.get("cond_dim", 512)),
                               width=int(cfg.get("width", 512)), layers=int(cfg.get("layers", 4)),
                               horizon=int(cfg.get("horizon", 12)),
                               vel_scale=float(cfg.get("vel_scale", 0.05))).to(model.device)
    after = sum(p.numel() for p in model.parameters())
    if abs(after / before - 1) > 0.02:
        raise ValueError(f"ODE total parameter difference exceeds 2%: {before} -> {after}")
    logger.info("ODE parameter audit: original=%d ODE=%d delta=%.4f%% head=%d", before, after,
                100 * (after / before - 1), sum(p.numel() for p in a.ode_head.parameters()))
    return model
