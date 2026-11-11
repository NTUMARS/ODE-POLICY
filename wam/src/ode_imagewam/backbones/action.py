# Derived from the MIT-licensed ImageWAM implementation. See LICENSE.MIT.
from __future__ import annotations
import torch
import torch.nn as nn
from .flux2_imports import ensure_flux2_importable

import math
from typing import Any, Dict
from .action_blocks import SlimFlux2DoubleBlock, SlimFlux2SingleBlock, Flux2ActionHead

class ActionDiTFlux2(nn.Module):
    """Slim action expert with FLUX.2-compatible attention dimensions."""

    block_protocol = "flux2"

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers_double: int = 5,
        num_layers_single: int = 20,
        mlp_ratio: float = 4.0,
        max_action_horizon: int = 64,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import MLPEmbedder, Modulation

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.double_layers = int(num_layers_double)
        self.single_layers = int(num_layers_single)
        self.max_action_horizon = int(max_action_horizon)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_dim, disable_bias=True)
        self.double_stream_modulation_img = Modulation(self.hidden_dim, double=True, disable_bias=True)
        self.single_stream_modulation = Modulation(self.hidden_dim, double=False, disable_bias=True)
        self.double_blocks = nn.ModuleList(
            [
                SlimFlux2DoubleBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    attn_head_dim=self.attn_head_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(self.double_layers)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SlimFlux2SingleBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    attn_head_dim=self.attn_head_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(self.single_layers)
            ]
        )
        self.head = Flux2ActionHead(self.hidden_dim, self.action_dim)

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)

    @classmethod
    def from_pretrained(cls, action_dit_config, action_dit_pretrained_path=None,
                        skip_dit_load_from_pretrain=False, zero_init_output_head=False,
                        output_head_init_scale=1.0, device="cuda", torch_dtype=torch.bfloat16):
        model = cls(**dict(action_dit_config)).to(device=device, dtype=torch_dtype)
        if action_dit_pretrained_path and not skip_dit_load_from_pretrain:
            payload = torch.load(action_dit_pretrained_path, map_location="cpu", weights_only=True)
            state = payload.get("state_dict", payload)
            missing, unexpected = model.load_state_dict(state, strict=False)
            required = [key for key in missing if not key.startswith(("action_encoder.", "head."))]
            if required or unexpected:
                raise ValueError(f"ActionDiT reader initialization differs: missing={required}, unexpected={unexpected}")
        scale = float(output_head_init_scale)
        if not math.isfinite(scale) or scale < 0 or (zero_init_output_head and scale != 1):
            raise ValueError("Invalid ActionDiT output initialization scale")
        with torch.no_grad():
            model.head.linear.weight.mul_(0 if zero_init_output_head else scale)
        return model

    @staticmethod
    def build_action_ids(
        batch_size: int,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
        ids[..., 0] = 2.0
        ids[..., 1] = torch.arange(seq_len, device=device, dtype=dtype)[None, :]
        return ids

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        del context, context_mask
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be [B,T,D], got {tuple(action_tokens.shape)}")
        batch_size, seq_len, action_dim = action_tokens.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")
        if seq_len > self.max_action_horizon:
            raise ValueError(f"Action length {seq_len} exceeds max_action_horizon={self.max_action_horizon}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be [B], got {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(f"`timestep` length must match batch size {batch_size}, got {timestep.shape[0]}")

        from flux2.model import timestep_embedding

        tokens = self.action_encoder(action_tokens)
        vec = self.time_in(timestep_embedding(timestep, 256)).to(dtype=tokens.dtype)
        double_mod_img = self.double_stream_modulation_img(vec)
        single_mod, _ = self.single_stream_modulation(vec)
        ids = self.build_action_ids(batch_size, seq_len, device=tokens.device, dtype=tokens.dtype)
        return {
            "tokens": tokens,
            "ids": ids,
            "t_mod": {"vec": vec, "double_img": double_mod_img, "single": single_mod},
            "context": None,
            "context_mask": None,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens, pre_state["t_mod"]["vec"])
