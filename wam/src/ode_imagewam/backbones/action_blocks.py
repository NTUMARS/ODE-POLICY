# Derived from the MIT-licensed ImageWAM implementation. See LICENSE.MIT.
from __future__ import annotations
import torch
import torch.nn as nn
from .flux2_imports import ensure_flux2_importable

class SlimFlux2SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int):
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import QKNorm

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.attn_dim, bias=False)
        self.norm = QKNorm(self.attn_head_dim)
        self.proj = nn.Linear(self.attn_dim, self.hidden_dim, bias=False)

class SlimFlux2DoubleBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import SiLUActivation

        self.hidden_size = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.mlp_hidden_dim = int(round(self.hidden_dim * float(mlp_ratio)))
        self.mlp_mult_factor = 2
        self.img_norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.img_attn = SlimFlux2SelfAttention(self.hidden_dim, self.num_heads, self.attn_head_dim)
        self.img_norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(self.mlp_hidden_dim, self.hidden_dim, bias=False),
        )

    def prepare_qkv(self, x: torch.Tensor, pe: torch.Tensor, mod_img):
        from einops import rearrange
        from flux2.model import apply_rope

        mod1, mod2 = mod_img
        mod1_shift, mod1_scale, mod1_gate = mod1
        mod2_shift, mod2_scale, mod2_gate = mod2
        x_mod = (1 + mod1_scale) * self.img_norm1(x) + mod1_shift
        qkv = self.img_attn.qkv(x_mod)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.img_attn.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return {
            "q": q.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "k": k.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "v": v.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "residual_x": x,
            "mod2_shift": mod2_shift,
            "mod2_scale": mod2_scale,
            "mod1_gate": mod1_gate,
            "mod2_gate": mod2_gate,
        }

    def apply_post(self, mixed_attn_out: torch.Tensor, state: dict) -> torch.Tensor:
        x = state["residual_x"] + state["mod1_gate"] * self.img_attn.proj(mixed_attn_out)
        x = x + state["mod2_gate"] * self.img_mlp(
            (1 + state["mod2_scale"]) * self.img_norm2(x) + state["mod2_shift"]
        )
        return x

class SlimFlux2SingleBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import QKNorm, SiLUActivation

        self.hidden_size = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.mlp_hidden_dim = int(round(self.hidden_dim * float(mlp_ratio)))
        self.mlp_mult_factor = 2
        self.linear1 = nn.Linear(
            self.hidden_dim,
            3 * self.attn_dim + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=False,
        )
        self.linear2 = nn.Linear(self.attn_dim + self.mlp_hidden_dim, self.hidden_dim, bias=False)
        self.norm = QKNorm(self.attn_head_dim)
        self.pre_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    def prepare_qkv(self, x: torch.Tensor, pe: torch.Tensor, mod):
        from einops import rearrange
        from flux2.model import apply_rope

        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.attn_dim, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return {
            "q": q.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "k": k.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "v": v.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "mlp": mlp,
            "gate": mod_gate,
            "residual_x": x,
        }

    def apply_post(self, mixed_attn_out: torch.Tensor, state: dict) -> torch.Tensor:
        output = self.linear2(torch.cat((mixed_attn_out, self.mlp_act(state["mlp"])), dim=2))
        return state["residual_x"] + state["gate"] * output

class Flux2ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, action_dim, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=False))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear((1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :])
