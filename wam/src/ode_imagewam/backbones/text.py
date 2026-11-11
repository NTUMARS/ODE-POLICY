# Derived from the MIT-licensed ImageWAM implementation. See LICENSE.MIT.
from __future__ import annotations
from typing import Optional, Sequence
import torch
from einops import rearrange

OUTPUT_LAYERS_QWEN3 = (9, 18, 27)


class Flux2Text:
    @torch.no_grad()
    def _encode_flux2_text(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        cached_text_hidden_states = sample.get("text_hidden_states")
        if cached_text_hidden_states is not None:
            text_hidden_states = cached_text_hidden_states.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            text_attention_mask = sample.get("text_attention_mask")
            if text_attention_mask is None:
                raise ValueError("FLUX.2 cached `text_hidden_states` must be paired with `text_attention_mask`.")
            else:
                text_attention_mask = text_attention_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            return text_hidden_states, text_attention_mask

        prompt = sample.get("instruction", sample.get("prompt", sample.get("task")))
        if prompt is None:
            raise ValueError(
                "FLUX.2 stack requires precomputed `text_hidden_states` or an `instruction`/`prompt`/`task`."
            )
        if self.text_encoder is None:
            raise ValueError("FLUX.2 online text encoding requires `load_text_encoder=true`.")
        video = sample.get("video")
        batch_size = int(video.shape[0]) if isinstance(video, torch.Tensor) and video.ndim == 5 else 1
        prompts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)
        return self._encode_flux2_prompts(prompts)

    @torch.no_grad()
    def _encode_flux2_prompts(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            self._load_flux2_text_encoder_for_inference()
        if not hasattr(self.text_encoder, "tokenizer") or not hasattr(self.text_encoder, "model"):
            text_hidden = self.text_encoder(list(prompts)).to(device=self.device, dtype=self.torch_dtype)
            text_mask = torch.ones(text_hidden.shape[:2], device=self.device, dtype=torch.bool)
            return text_hidden, text_mask

        ensure_flux2_importable_path = getattr(self, "model_paths", {}).get("flux2_src")
        from .flux2_imports import ensure_flux2_importable

        ensure_flux2_importable(ensure_flux2_importable_path)

        tokenizer = self.text_encoder.tokenizer
        model = self.text_encoder.model
        max_length = int(getattr(self.text_encoder, "max_length", 512))
        all_input_ids = []
        all_attention_masks = []
        for prompt in prompts:
            messages = [{"role": "user", "content": str(prompt)}]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            model_inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(model.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = torch.stack([outputs.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        hidden = rearrange(hidden, "b c l d -> b l (c d)")
        return (
            hidden.to(device=self.device, dtype=self.torch_dtype),
            attention_mask.to(device=self.device, dtype=torch.bool),
        )

    def _load_flux2_text_encoder_for_inference(self) -> None:
        from types import SimpleNamespace
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_spec = getattr(self, "flux2_qwen3_model_spec", None)
        if model_spec is None:
            model_spec = getattr(self, "model_paths", {}).get("qwen3")
        if model_spec is None:
            raise ValueError("Online text encoding requires a local Qwen3 snapshot")
        qwen3_model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype=self.torch_dtype, local_files_only=True,
        ).to(self.device).eval().requires_grad_(False)
        self.text_encoder = SimpleNamespace(
            model=qwen3_model,
            tokenizer=AutoTokenizer.from_pretrained(model_spec, local_files_only=True),
            max_length=int(getattr(self, "qwen_context_len", 512)),
        )

    @torch.no_grad()
    def _prepare_flux2_infer_text(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            return self._encode_flux2_prompts([prompt])
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )
