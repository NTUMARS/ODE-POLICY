# Derived from the MIT-licensed ImageWAM implementation. See LICENSE.MIT.
"""FLUX.2 mixture attention with a differentiable visual-prefix cache."""
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MoT(nn.Module):
    def __init__(self, mixtures, mot_checkpoint_mixed_attn=True,
                 gqa_implementation="repeat", force_flash_attention=False):
        super().__init__()
        if set(mixtures) != {"video", "action"}:
            raise ValueError("The model requires one visual and one action expert")
        self.mixtures = nn.ModuleDict(mixtures)
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        self.gqa_implementation = str(gqa_implementation)
        self.force_flash_attention = bool(force_flash_attention)
        video, action = mixtures["video"], mixtures["action"]
        self.num_heads = int(video.num_heads)
        self.num_kv_heads = int(video.num_kv_heads)
        self.attn_head_dim = int(video.attn_head_dim)
        self.num_layers = len(video.blocks)
        self.block_protocol = "flux2"
        for name in ("num_heads", "num_kv_heads", "attn_head_dim", "double_layers", "single_layers"):
            if getattr(video, name) != getattr(action, name):
                raise ValueError(f"Visual and action experts must share {name}")
        if self.gqa_implementation not in {"repeat", "sdpa"}:
            raise ValueError("gqa_implementation must be repeat or sdpa")

    def _mixed_attention(self, q_cat, k_cat, v_cat, attention_mask):
        batch, queries, _ = q_cat.shape
        keys = k_cat.shape[1]
        heads, kv_heads, width = self.num_heads, self.num_kv_heads, self.attn_head_dim
        mask = self._format_attention_mask(attention_mask, batch, queries, keys, q_cat.device)
        def attend(q_flat, k_flat, v_flat):
            q = q_flat.reshape(batch, queries, heads, width).transpose(1, 2)
            k = k_flat.reshape(batch, keys, kv_heads, width).transpose(1, 2)
            v = v_flat.reshape(batch, keys, kv_heads, width).transpose(1, 2)
            gqa = kv_heads != heads and self.gqa_implementation == "sdpa"
            if kv_heads != heads and not gqa:
                k = k.repeat_interleave(heads // kv_heads, dim=1)
                v = v.repeat_interleave(heads // kv_heads, dim=1)
            context = nullcontext()
            if self.force_flash_attention:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                context = sdpa_kernel([SDPBackend.FLASH_ATTENTION])
            with context:
                value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)
            return value.transpose(1, 2).reshape(batch, queries, heads * width)
        if self.mot_checkpoint_mixed_attn and self.training and torch.is_grad_enabled():
            return checkpoint(attend, q_cat, k_cat, v_cat, use_reentrant=False)
        return attend(q_cat, k_cat, v_cat)

    @staticmethod
    def _format_attention_mask(
        attention_mask: torch.Tensor,
        batch_size: int,
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = attention_mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 2:
            if tuple(mask.shape) != (query_len, key_len):
                raise ValueError(f"2D attention mask must be {(query_len, key_len)}, got {tuple(mask.shape)}")
            return mask.view(1, 1, query_len, key_len)
        if mask.ndim == 3:
            if mask.shape[0] != batch_size or tuple(mask.shape[1:]) != (query_len, key_len):
                raise ValueError(
                    f"3D attention mask must be {(batch_size, query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask.unsqueeze(1)
        if mask.ndim == 4:
            if mask.shape[0] not in (1, batch_size) or tuple(mask.shape[-2:]) != (query_len, key_len):
                raise ValueError(
                    f"4D attention mask must end with {(query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask
        raise ValueError(f"attention_mask must be 2D/3D/4D, got shape {tuple(mask.shape)}")

    def _flux2_flatten_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])

    def _flux2_video_single_io(self, block, x: torch.Tensor, pe: torch.Tensor, mod) -> dict:
        from flux2.model import apply_rope

        q, k, v, mlp, gate = block._qkv(x, mod)
        q, k = apply_rope(q, k, pe)
        return {
            "q": self._flux2_flatten_heads(q),
            "k": self._flux2_flatten_heads(k),
            "v": self._flux2_flatten_heads(v),
            "mlp": mlp,
            "gate": gate,
            "residual_x": x,
        }

    def prefill_flux2_video_cache(
        self,
        video_tokens: dict[str, torch.Tensor],
        video_freqs: dict[str, torch.Tensor],
        video_t_mod: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        if self.block_protocol != "flux2":
            raise ValueError("`prefill_flux2_video_cache` requires block_protocol='flux2'.")
        video_expert = self.mixtures["video"]
        txt = video_tokens["txt"]
        img = video_tokens["img"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        from flux2.model import apply_rope

        double_cache = []
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
            flat_q = self._flux2_flatten_heads(q)
            flat_k = self._flux2_flatten_heads(k)
            flat_v = self._flux2_flatten_heads(v)
            double_cache.append({"k": flat_k, "v": flat_v})
            mixed = self._mixed_attention(flat_q, flat_k, flat_v, attention_mask["double_joint"])
            txt_attn, img_attn = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = block._apply_residuals(img, txt, img_attn, txt_attn, mods)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        single_cache = []
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            block = video_expert.single_blocks[layer_idx]
            state = self._flux2_video_single_io(block, video_stream, stream_pe, video_t_mod["single"])
            single_cache.append({"k": state["k"], "v": state["v"]})
            mixed = self._mixed_attention(state["q"], state["k"], state["v"], attention_mask["single"])
            video_stream = block._out(state["residual_x"], mixed, state["mlp"], state["gate"])

        return {
            "double": double_cache,
            "single": single_cache,
            "txt_len": int(txt.shape[1]),
            "img_len": int(img.shape[1]),
            "final_video": {
                "txt": video_stream[:, : txt.shape[1]],
                "img": video_stream[:, txt.shape[1] :],
            },
        }

    def forward_action_with_video_cache(self, action_tokens, action_freqs, action_t_mod,
                                        action_context_payload, video_kv_cache,
                                        attention_mask, video_seq_len):
        del action_freqs
        video, expert = self.mixtures["video"], self.mixtures["action"]
        action = action_tokens
        ids = action_context_payload["ids"].to(device=action.device, dtype=action.dtype)
        positional = video.transformer.pe_embedder(ids)
        total = video_seq_len + action.shape[1]
        def action_mask(mask):
            return mask[..., video_seq_len:total, :total]
        for name, blocks, modulation in (
            ("double", expert.double_blocks, action_t_mod["double_img"]),
            ("single", expert.single_blocks, action_t_mod["single"]),
        ):
            mask = action_mask(attention_mask["double_joint" if name == "double" else "single"])
            caches = video_kv_cache[name]
            if len(caches) != len(blocks):
                raise ValueError("Visual prefix cache does not match action reader depth")
            for block, cache in zip(blocks, caches):
                state = block.prepare_qkv(action, positional, modulation)
                keys = torch.cat((cache["k"].to(state["k"].dtype), state["k"]), dim=1)
                values = torch.cat((cache["v"].to(state["v"].dtype), state["v"]), dim=1)
                mixed = self._mixed_attention(state["q"], keys, values, mask)
                action = block.apply_post(mixed, state)
        return action
