# Derived from the MIT-licensed ImageWAM implementation. See LICENSE.MIT.
from __future__ import annotations
from typing import Optional
import torch


class Flux2Inputs:
    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        if not getattr(self, "pack_proprio_after_text", False):
            proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
            return (
                torch.cat([context, proprio_token], dim=1),
                torch.cat([context_mask, proprio_mask], dim=1),
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[:2] != context_mask.shape:
            raise ValueError(
                f"`context/context_mask` leading dims must match, got {tuple(context.shape[:2])} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != proprio_token.shape[0]:
            raise ValueError(
                f"`proprio` batch size must match context batch size ({context.shape[0]}), got {proprio_token.shape[0]}"
            )

        context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        new_context = context.new_zeros(context.shape[0], context.shape[1] + 1, context.shape[2])

        # The packed text context uses a contiguous valid prefix, so the
        # proprio token must sit immediately after the valid text tokens.
        valid_counts = context_mask.sum(dim=1)
        valid_rank = context_mask.cumsum(dim=1) - 1
        invalid_mask = ~context_mask
        invalid_rank = invalid_mask.cumsum(dim=1) - 1
        target_indices = torch.where(
            context_mask,
            valid_rank,
            valid_counts[:, None] + 1 + invalid_rank,
        )
        new_context.scatter_(
            dim=1,
            index=target_indices[:, :, None].expand(-1, -1, context.shape[2]),
            src=context,
        )
        batch_indices = torch.arange(context.shape[0], device=context.device)
        new_context[batch_indices, valid_counts] = proprio_token[:, 0]
        positions = torch.arange(context.shape[1] + 1, device=context.device)
        new_context_mask = positions[None, :] <= valid_counts[:, None]

        return new_context, new_context_mask

    def _append_proprio_to_context_if_enabled(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "proprio_encoder", None) is None:
            return context, context_mask
        if proprio is None:
            raise ValueError(f"`{source}` requires `proprio` when `proprio_dim` is enabled.")

        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        elif proprio.ndim == 2:
            pass
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        else:
            raise ValueError(
                f"`{source}` `proprio` must be [B,T,D], [B,D], or [D], got shape {tuple(proprio.shape)}"
            )
        if proprio.shape[0] != context.shape[0]:
            raise ValueError(
                f"`{source}` `proprio` batch size must match context batch size "
                f"({context.shape[0]}), got {proprio.shape[0]}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`{source}` `proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        return self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
        )

    @torch.no_grad()
    def _encode_flux2_image_tokens(
        self,
        image: torch.Tensor,
        *,
        time_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .video import Flux2VideoExpert

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 16 != 0 or image.shape[-1] % 16 != 0:
            raise ValueError(f"FLUX.2 image spatial dims must be multiples of 16, got {tuple(image.shape[-2:])}")
        image = image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        latents = self.vae.encode(image).to(dtype=self.torch_dtype)
        tokens = Flux2VideoExpert.pack_latents(latents)
        _, _, latent_h, latent_w = latents.shape
        ids = Flux2VideoExpert.build_img_ids(
            batch_size=int(latents.shape[0]),
            token_height=int(latent_h),
            token_width=int(latent_w),
            time_value=float(time_value),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens, ids

    @torch.no_grad()
    def _decode_flux2_image_tokens(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        from .video import Flux2VideoExpert

        latent_h = int(height) // 16
        latent_w = int(width) // 16
        latents = Flux2VideoExpert.unpack_latents(tokens, latent_h, latent_w)
        image = self.vae.decode(latents.to(device=self.device, dtype=self.torch_dtype))
        return image.detach().float().clamp(-1, 1)

    def build_inputs_flux2(self, sample, tiled: bool = False):
        del tiled
        video = sample.get("video")
        if "target_latent" in sample and "target_img_ids" in sample:
            target_tokens = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            target_img_ids = sample["target_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("FLUX.2 stack sample requires `next_frame`, `target_image`, or target token fields.")
            target_tokens, target_img_ids = self._encode_flux2_image_tokens(next_frame, time_value=0.0)

        if "ref_image_latents" in sample and "ref_img_ids" in sample:
            ref_tokens = sample["ref_image_latents"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ref_img_ids = sample["ref_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("FLUX.2 stack sample requires `current_frame`, `input_image`, or ref token fields.")
            ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(current_frame, time_value=10.0)

        text_hidden_states, text_attention_mask = self._encode_flux2_text(sample)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="FLUX.2 training sample",
            )
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        past_action = sample.get("past_action")
        if past_action is not None:
            past_action = past_action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        past_action_is_pad = sample.get("past_action_is_pad")
        if past_action_is_pad is not None:
            past_action_is_pad = past_action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        result = {
            "target_latent": target_tokens,
            "target_img_ids": target_img_ids,
            "ref_image_latents": ref_tokens,
            "ref_img_ids": ref_img_ids,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "past_action": past_action,
            "action_is_pad": action_is_pad,
            "past_action_is_pad": past_action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }
        return result
