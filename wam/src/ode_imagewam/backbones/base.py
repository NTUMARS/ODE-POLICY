# Derived from the MIT-licensed ImageWAM implementation. See LICENSE.MIT.
"""Native FLUX.2 world-action model with a future-image flow objective."""
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conditioning import Flux2Inputs
from .text import Flux2Text
from .schedule import ContinuousFlowMatchScheduler


class ImageWAM(Flux2Inputs, Flux2Text, nn.Module):
    def __init__(self, video_expert, action_expert, mot, vae, *, text_dim,
                 proprio_dim=8, device="cpu", torch_dtype=torch.float32,
                 video_scheduler=None, loss=None, pack_proprio_after_text=True,
                 qwen_context_len=128, qwen_model=None):
        super().__init__()
        self.video_expert, self.action_expert, self.mot = video_expert, action_expert, mot
        self.dit = self.mot
        self.vae = vae.eval().requires_grad_(False)
        self.text_encoder = None
        self.text_dim, self.proprio_dim = int(text_dim), int(proprio_dim)
        self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        schedule = dict(video_scheduler or {})
        self.train_video_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=int(schedule.get("num_train_timesteps", 1000)),
            shift=float(schedule.get("train_shift", 5.0)))
        self.loss_lambda_video = float((loss or {}).get("lambda_video", 0.5))
        self.loss_lambda_action = float((loss or {}).get("lambda_action", 1.0))
        self.device, self.torch_dtype = torch.device(device), torch_dtype
        self.pack_proprio_after_text = bool(pack_proprio_after_text)
        self.qwen_context_len = int(qwen_context_len)
        self.flux2_qwen3_model_spec = qwen_model
        self.stack = "flux2"
        self.to(device=device, dtype=torch_dtype)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if getattr(self, "text_encoder", None) is not None:
            self.text_encoder.model.to(*args, **kwargs)
        return self

    def _flux2_action_uses_noise_context(self):
        return False

    def _training_loss_flux2(self, sample, tiled=False, inputs=None):
        inputs = self.build_inputs_flux2(sample, tiled) if inputs is None else inputs
        target = inputs["target_latent"]
        noise = torch.randn_like(target)
        schedule = self.train_video_scheduler
        timestep = schedule.sample_training_t(target.shape[0], self.device, target.dtype)
        noisy = schedule.add_noise(target, noise, timestep)
        pre = self.video_expert.pre_dit(
            x=noisy, timestep=timestep / float(schedule.num_train_timesteps),
            context=inputs["text_hidden_states"], context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
            target_img_ids=inputs["target_img_ids"], ref_img_ids=inputs["ref_img_ids"])
        mask = self._build_mot_attention_mask_flux2(
            batch_size=target.shape[0], txt_len=pre["txt_len"], target_len=pre["target_len"],
            cond_len=pre["cond_len"], action_len=0, device=target.device,
            text_attention_mask=pre["text_mask"])
        predicted = self.video_expert.post_dit(self._forward_flux2_video_only(pre, mask), pre)
        error = F.mse_loss(predicted.float(), (noise - target).float(), reduction="none").flatten(1).mean(1)
        loss = (error * schedule.training_weight(timestep)).mean() * self.loss_lambda_video
        return loss, {"loss_video": float(loss.detach()), "loss_action": 0.0}

    @torch.no_grad()
    def _build_mot_attention_mask_flux2(
        self,
        batch_size: int,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None = None,
        action_can_attend_target: bool = False,
    ) -> dict[str, torch.Tensor]:
        t0 = 0
        r0 = txt_len
        x0 = txt_len + cond_len
        a0 = txt_len + cond_len + target_len
        total = a0 + action_len
        mask = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)
        # Stable text/ref prefix cannot attend to noisy target/action tokens.
        mask[:, t0:r0, t0:x0] = True
        mask[:, r0:x0, t0:x0] = True
        # Target/noisy image uses stable prefix and target self-context.
        mask[:, x0:a0, t0:a0] = True
        # The action reader attends only to the current visual prefix.
        action_context_end = a0 if action_can_attend_target else x0
        mask[:, a0:total, t0:action_context_end] = True
        mask[:, a0:total, a0:total] = True
        if text_attention_mask is not None:
            if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (batch_size, txt_len):
                raise ValueError(
                    "`text_attention_mask` must be [B,txt_len], "
                    f"got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
                )
            text_valid = text_attention_mask.to(device=device, dtype=torch.bool)
            mask[:, :, t0:r0] &= text_valid[:, None, :]
        return {"double_joint": mask, "single": mask.clone()}

    def _forward_flux2_video_only(
        self,
        video_pre: dict[str, Any],
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        video_state = video_pre["tokens"]
        if not isinstance(video_state, dict):
            raise ValueError("FLUX.2 video tokens must be a dict with `txt` and `img` tensors.")

        txt = video_state["txt"]
        img = video_state["img"]
        video_freqs = video_pre["freqs"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        video_t_mod = video_pre["t_mod"]
        video_expert = self.video_expert

        def _flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])

        from flux2.model import apply_rope

        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            block = video_expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                video_t_mod["double_img"],
                video_t_mod["double_txt"],
            )
            q, k = apply_rope(q, k, pe_full)
            mixed = self.mot._mixed_attention(
                _flatten_heads(q),
                _flatten_heads(k),
                _flatten_heads(v),
                attention_mask["double_joint"],
            )
            txt_attn, img_attn = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = block._apply_residuals(img, txt, img_attn, txt_attn, mods)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            block = video_expert.single_blocks[layer_idx]
            q, k, v, mlp, gate = block._qkv(video_stream, video_t_mod["single"])
            q, k = apply_rope(q, k, stream_pe)
            mixed = self.mot._mixed_attention(
                _flatten_heads(q),
                _flatten_heads(k),
                _flatten_heads(v),
                attention_mask["single"],
            )
            video_stream = block._out(video_stream, mixed, mlp, gate)

        txt_len = int(txt.shape[1])
        txt, img = video_stream[:, :txt_len], video_stream[:, txt_len:]
        return {"txt": txt, "img": img}
